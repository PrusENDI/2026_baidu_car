import importlib.util
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


path = Path(__file__).parents[1] / "tools" / "safe_lane_test.py"
spec = importlib.util.spec_from_file_location("safe_lane_test", path)
safe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safe)


class SafetyTests(unittest.TestCase):
    def test_record_dir_argument_defaults_to_none(self):
        self.assertIsNone(safe.parse_args([]).record_dir)

    def test_argument_limits_allow_300_seconds_but_reject_longer(self):
        self.assertIsNone(safe.validate_limits(0.10, 300.0))
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.10, 300.1),
        )
        self.assertEqual(
            "speed 0..0.10; duration 0..300",
            safe.validate_limits(0.11, 15.0),
        )

    def test_rejects_non_finite_or_excessive_model_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([float("nan"), 0.0], 1.0))
        self.assertEqual("model_output_limit", safe.validate_output([1.1, 0.0], 1.0))
        self.assertIsNone(safe.validate_output([0.2, -0.3], 1.0))

    def test_rejects_wrong_length_lane_output(self):
        self.assertEqual("invalid_model_output", safe.validate_output([0.1], 1.0))
        self.assertEqual("invalid_model_output", safe.validate_output([0.1, 0.2, 0.3], 1.0))

    def test_wait_ready_times_out_without_server(self):
        class NeverReady:
            def ready(self, timeout_ms):
                return False

        self.assertFalse(safe.wait_ready(NeverReady(), timeout=0.0, sleep=lambda _: None))

    def test_cold_start_preflight_allows_five_seconds(self):
        class Client:
            timeout_ms = None

            def infer(self, frame, timeout_ms):
                self.timeout_ms = timeout_ms
                return [0.1, -0.2]

        client = Client()
        self.assertEqual([0.1, -0.2], safe.preflight_infer(client, object()))
        self.assertEqual(5000, client.timeout_ms)

    def test_loop_uses_cold_start_timeout_then_normal_timeout(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def __init__(self):
                self.timeouts = []

            def infer(self, frame, timeout_ms):
                self.timeouts.append(timeout_ms)
                return [0.1, -0.2]

        class Car:
            def set_velocity(self, speed, y, angle):
                pass

        lane = Lane()
        ticks = iter([0.0, 0.0, 0.0, 0.01, 0.02, 0.02, 0.03, 1.0])
        reason = safe.run_lane_loop(
            duration=0.1, speed=0.05, output_limit=1.0,
            cap=Cap(), stream=Stream(), lane=lane, car=Car(),
            py=lambda value: value, pa=lambda value: value,
            monotonic=lambda: next(ticks),
        )

        self.assertEqual("duration_elapsed", reason)
        self.assertEqual([5000, 1000], lane.timeouts)

    def test_initial_loop_inference_timeout_stops_before_velocity_command(self):
        expected_timeout = 5000

        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame, timeout_ms):
                if timeout_ms != expected_timeout:
                    raise AssertionError("expected cold-start timeout")
                return None

        class Car:
            def __init__(self):
                self.calls = []

            def set_velocity(self, speed, y, angle):
                self.calls.append((speed, y, angle))

        car = Car()
        ticks = iter([0.0, 0.0, 0.0, 0.01])
        reason = safe.run_lane_loop(
            duration=0.1, speed=0.05, output_limit=1.0,
            cap=Cap(), stream=Stream(), lane=Lane(), car=car,
            py=lambda value: value, pa=lambda value: value,
            monotonic=lambda: next(ticks),
        )

        self.assertEqual("inference_timeout", reason)
        self.assertEqual([], car.calls)

    def test_recorder_none_keeps_normal_duration_elapsed_behavior(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame, timeout_ms):
                return [0.1, -0.2]

        class Car:
            calls = 0

            def set_velocity(self, speed, y, angle):
                self.calls += 1

        ticks = iter([0.0, 0.0, 0.0, 0.0, 1.0])
        car = Car()
        reason = safe.run_lane_loop(
            duration=0.1,
            speed=0.05,
            output_limit=1.0,
            cap=Cap(),
            stream=Stream(),
            lane=Lane(),
            car=car,
            py=lambda value: value,
            pa=lambda value: value,
            monotonic=lambda: next(ticks),
            recorder=None,
        )
        self.assertEqual("duration_elapsed", reason)
        self.assertEqual(1, car.calls)


class LaneLoopRecordingTests(unittest.TestCase):
    def test_records_exact_chassis_commands_before_setting_velocity(self):
        events = []

        class Cap:
            def read(self):
                return "frame"

        class Stream:
            def update_frame(self, frame, name):
                events.append(("stream", frame, name))

        class Lane:
            def infer(self, frame, timeout_ms):
                events.append(("infer", frame))
                return [0.25, -0.25]

        class Recorder:
            def __init__(self):
                self.calls = []

            def record(self, *args):
                self.calls.append(args)
                events.append(("record", args))

        class Car:
            def __init__(self):
                self.calls = []

            def set_velocity(self, speed, vy, yaw):
                self.calls.append((speed, vy, yaw))
                events.append(("command", speed, vy, yaw))

        recorder = Recorder()
        car = Car()
        ticks = iter([0.0, 0.0, 0.0, 0.01, 1.0])

        reason = safe.run_lane_loop(
            duration=0.1,
            speed=0.05,
            output_limit=1.0,
            cap=Cap(),
            stream=Stream(),
            lane=Lane(),
            car=car,
            py=lambda value: value * 2,
            pa=lambda value: value * 3,
            monotonic=lambda: next(ticks),
            recorder=recorder,
        )

        self.assertEqual("duration_elapsed", reason)
        self.assertLess(
            next(index for index, event in enumerate(events) if event[0] == "record"),
            next(index for index, event in enumerate(events) if event[0] == "command"),
        )
        self.assertEqual((0.05, -0.5, 0.75), car.calls[0])
        recorded = recorder.calls[0]
        self.assertEqual(
            ("frame", 0.0, 0, 10.0, 0.0, 0.25, -0.25,
             -0.5, 0.75, 0.05, -0.5, 0.75),
            recorded,
        )


class CliRecordingLifecycleTests(unittest.TestCase):
    def _run_main(self, argv, *, recorder_create=None, loop=None, preflight_output=(0.0, 0.0)):
        events = []

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def __init__(self):
                events.append("driver.init")

            def stop(self):
                events.append("car.stop")

        messages = []
        patches = [
            patch.object(safe.subprocess, "Popen", return_value=Process()),
            patch.object(safe, "LaneClient", return_value=Lane()),
            patch.object(safe, "wait_ready", return_value=True),
            patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())),
            patch.object(safe, "preflight_infer", return_value=preflight_output),
            patch("builtins.print", side_effect=messages.append),
        ]
        if recorder_create is not None:
            patches.append(patch.object(safe.TestRecorder, "create", side_effect=recorder_create))
        if loop is not None:
            patches.append(patch.object(safe, "run_lane_loop", side_effect=loop))
        with ExitStack() as stack:
            for active_patch in patches:
                stack.enter_context(active_patch)
            safe.main(argv)
        return events, messages

    def test_preflight_only_record_dir_never_creates_recorder_or_driver(self):
        events, messages = self._run_main(
            ["--preflight-only", "--record-dir", "records"],
            recorder_create=AssertionError("recorder must not be created"),
        )

        self.assertNotIn("driver.init", events)
        self.assertIn("SAFE_STOP reason=preflight_complete", messages)

    def test_preflight_timeout_has_explicit_reason(self):
        events, messages = self._run_main(
            [], loop=AssertionError("loop must not run"), preflight_output=None,
        )

        self.assertNotIn("driver.init", events)
        self.assertEqual(["SAFE_STOP reason=preflight_inference_timeout"], messages)

    def test_recording_init_failure_prevents_driver_and_safely_stops(self):
        events, messages = self._run_main(
            ["--record-dir", "records"],
            recorder_create=OSError("disk unavailable"),
        )

        self.assertNotIn("driver.init", events)
        self.assertIn("SAFE_STOP reason=recording_init_failure:OSError", messages)

    def test_loop_recorder_failure_stops_car_before_closing_recorder(self):
        class Recorder:
            def close(self, reason):
                events.append(("recorder.close", reason))

        events = []
        recorder = Recorder()

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def stop(self):
                events.append("car.stop")

        messages = []
        with patch.object(safe.subprocess, "Popen", return_value=Process()), \
             patch.object(safe, "LaneClient", return_value=Lane()), \
             patch.object(safe, "wait_ready", return_value=True), \
             patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())), \
             patch.object(safe, "preflight_infer", return_value=[0.0, 0.0]), \
             patch.object(safe.TestRecorder, "create", return_value=recorder), \
             patch.object(safe, "run_lane_loop", side_effect=safe.RecordingWriteError(RuntimeError("write failed"))), \
             patch("builtins.print", side_effect=messages.append):
            safe.main(["--record-dir", "records"])

        self.assertIn("SAFE_STOP reason=recording_write_failure:RuntimeError", messages)
        self.assertLess(events.index("car.stop"), events.index(("recorder.close", "recording_write_failure:RuntimeError")))

    def test_final_stop_failure_does_not_skip_remaining_cleanup(self):
        events = []

        class Recorder:
            def close(self, reason):
                events.append(("recorder.close", reason))

        class Process:
            def terminate(self):
                events.append("process.terminate")

            def wait(self, timeout):
                events.append("process.wait")

        class Lane:
            def close(self):
                events.append("lane.close")

        class Cap:
            def read(self):
                return "frame"

            def close(self):
                events.append("cap.close")

        class Stream:
            def stop(self):
                events.append("stream.stop")

        class Driver:
            def __init__(self):
                self.stop_calls = 0

            def stop(self):
                self.stop_calls += 1
                events.append("car.stop")
                if self.stop_calls == 2:
                    raise RuntimeError("stop failed")

        messages = []
        with patch.object(safe.subprocess, "Popen", return_value=Process()), \
             patch.object(safe, "LaneClient", return_value=Lane()), \
             patch.object(safe, "wait_ready", return_value=True), \
             patch.object(safe, "hardware_components", return_value=(lambda *_: Cap(), lambda: Stream(), Driver, lambda *_, **__: object())), \
             patch.object(safe, "preflight_infer", return_value=[0.0, 0.0]), \
             patch.object(safe.TestRecorder, "create", return_value=Recorder()), \
             patch.object(safe, "run_lane_loop", return_value="duration_elapsed"), \
             patch("builtins.print", side_effect=messages.append):
            safe.main(["--record-dir", "records"])

        final_stop = len(events) - 1 - events[::-1].index("car.stop")
        self.assertEqual(
            [
                "car.stop",
                ("recorder.close", "duration_elapsed"),
                "cap.close",
                "stream.stop",
                "lane.close",
                "process.terminate",
                "process.wait",
            ],
            events[final_stop:],
        )
        self.assertIn("SAFE_STOP reason=duration_elapsed", messages)


class RecorderContractTests(unittest.TestCase):
    def test_session_name_is_utc_timestamp_with_random_hex_suffix(self):
        now = safe.datetime.datetime(2026, 7, 16, 9, 8, 7, tzinfo=safe.datetime.timezone.utc)

        name = safe.utc_session_name(now)

        self.assertRegex(name, r"^20260716T090807Z-[0-9a-f]{8}$")

    def test_create_record_session_creates_exclusive_child_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            session = safe.create_record_session(Path(directory), "session-1")

            self.assertEqual(Path(directory) / "session-1", session)
            self.assertTrue(session.is_dir())
            with self.assertRaises(FileExistsError):
                safe.create_record_session(Path(directory), "session-1")

    def test_open_video_writer_uses_mp4v_20fps_and_rejects_closed_writer(self):
        calls = []

        class Writer:
            def isOpened(self):
                return False

            def release(self):
                pass

        cv2 = SimpleNamespace(
            VideoWriter_fourcc=lambda *codec: calls.append(codec) or 123,
            VideoWriter=lambda path, fourcc, fps, size: calls.append((path, fourcc, fps, size)) or Writer(),
        )
        with patch.dict(sys.modules, {"cv2": cv2}):
            with self.assertRaisesRegex(RuntimeError, "^recording_video_open_failed$"):
                safe.open_video_writer("out.mp4", (320, 240))

        self.assertEqual(("m", "p", "4", "v"), calls[0])
        self.assertEqual(("out.mp4", 123, 20.0, (320, 240)), calls[1])

    def test_recorder_writes_annotated_video_csv_and_atomic_metadata(self):
        class Writer:
            def __init__(self):
                self.frames = []
                self.released = False

            def write(self, frame):
                self.frames.append(frame)

            def release(self):
                self.released = True

        writer = Writer()
        cv2 = SimpleNamespace(FONT_HERSHEY_SIMPLEX=0, putText=lambda *args: args[0])
        frame = SimpleNamespace(copy=lambda: "annotated-frame")
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.dict(sys.modules, {"cv2": cv2}):
            recorder = safe.TestRecorder.create(
                directory, "session-1", (320, 240), {"speed": 0.08},
            )
            recorder.record(frame, 1.25, 3, 12.5, 25.0, 0.1, -0.2, 0.3, -0.4, 0.08, 0.0, -0.1)
            recorder.close("duration_elapsed")
            recorder.close("ignored")

            session = Path(directory) / "session-1"
            self.assertEqual(["annotated-frame"], writer.frames)
            self.assertTrue(writer.released)
            self.assertEqual(
                ",".join(safe.CSV_FIELDS),
                (session / "frames.csv").read_text(encoding="utf-8").splitlines()[0],
            )
            row = (session / "frames.csv").read_text(encoding="utf-8").splitlines()[1].split(",")
            self.assertEqual("true", row[-1])
            self.assertEqual("3", row[1])
            self.assertFalse((session / "metadata.json.tmp").exists())
            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(1, metadata["format_version"])
            self.assertEqual({"speed": 0.08}, metadata["parameters"])
            self.assertEqual([320, 240], metadata["frame_size"])
            self.assertEqual("annotated.mp4", metadata["video_file"])
            self.assertEqual("frames.csv", metadata["csv_file"])
            self.assertEqual(1, metadata["frame_count"])
            self.assertEqual(1, metadata["video_frame_count"])
            self.assertEqual("duration_elapsed", metadata["stop_reason"])
            self.assertTrue(metadata["started_at_utc"])
            self.assertTrue(metadata["ended_at_utc"])

    def test_create_removes_empty_session_when_video_writer_initialization_fails(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", side_effect=RuntimeError("writer failed")):
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

            self.assertFalse((Path(directory) / "session-1").exists())

    def test_create_releases_writer_when_csv_initialization_fails(self):
        class Writer:
            released = False

            def release(self):
                self.released = True

        writer = Writer()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.object(Path, "open", side_effect=OSError("csv failed")):
            with self.assertRaisesRegex(OSError, "csv failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

        self.assertTrue(writer.released)

    def test_create_closes_csv_and_releases_writer_when_header_write_fails(self):
        class Writer:
            released = False

            def release(self):
                self.released = True

        class CsvHandle:
            closed = False

            def close(self):
                self.closed = True

        class HeaderWriter:
            def __init__(self, handle, fieldnames):
                pass

            def writeheader(self):
                raise OSError("header failed")

        writer = Writer()
        csv_handle = CsvHandle()
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(safe, "open_video_writer", return_value=writer), \
             patch.object(Path, "open", return_value=csv_handle), \
             patch.object(safe.csv, "DictWriter", HeaderWriter):
            with self.assertRaisesRegex(OSError, "header failed"):
                safe.TestRecorder.create(directory, "session-1", (320, 240), {})

        self.assertTrue(writer.released)
        self.assertTrue(csv_handle.closed)

    def test_close_publishes_metadata_when_release_and_csv_close_fail(self):
        class Writer:
            def release(self):
                raise RuntimeError("release failed")

        class CsvHandle:
            def close(self):
                raise RuntimeError("csv close failed")

        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory) / "session-1"
            session.mkdir()
            recorder = safe.TestRecorder(session, Writer(), CsvHandle(), object(), (320, 240), {})

            with self.assertRaisesRegex(RuntimeError, "^release failed$") as raised:
                recorder.close("safe_stop")

            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("safe_stop", metadata["stop_reason"])
            self.assertEqual(
                ("RuntimeError: csv close failed",),
                raised.exception.recorder_cleanup_errors,
            )


if __name__ == "__main__":
    unittest.main()
