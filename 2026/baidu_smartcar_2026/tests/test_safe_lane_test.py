import importlib.util
import json
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

    def test_normal_lane_loop_keeps_duration_elapsed_reason(self):
        class Cap:
            def read(self):
                return object()

        class Stream:
            def update_frame(self, frame, name):
                pass

        class Lane:
            def infer(self, frame):
                return [0.1, -0.2]

        class Car:
            calls = 0

            def set_velocity(self, speed, y, angle):
                self.calls += 1

        ticks = iter([0.0, 0.0, 1.0])
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
        )
        self.assertEqual("duration_elapsed", reason)
        self.assertEqual(1, car.calls)


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

            with self.assertRaises(Exception):
                recorder.close("safe_stop")

            metadata = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual("safe_stop", metadata["stop_reason"])


if __name__ == "__main__":
    unittest.main()
