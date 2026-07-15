"""Minimal, bounded lane-following test without importing the smartcar root package."""
import argparse, importlib.util, math, sys, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def validate_output(error_y, error_angle, limit):
    if not math.isfinite(error_y) or not math.isfinite(error_angle): return 'invalid_model_output'
    if abs(error_y) > limit or abs(error_angle) > limit: return 'model_output_limit'
    return None

def package(name, path):
    mod = types.ModuleType(name); mod.__path__ = [str(path)]; mod.__package__ = name; sys.modules[name] = mod

def load(name, path, package_name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); mod.__package__ = package_name; sys.modules[name] = mod; spec.loader.exec_module(mod); return mod

def components():
    for name, path in {
        'smartcar': ROOT/'smartcar', 'smartcar.whalesbot': ROOT/'smartcar/whalesbot',
        'smartcar.whalesbot.tools': ROOT/'smartcar/whalesbot/tools',
        'smartcar.whalesbot.vehicle': ROOT/'smartcar/whalesbot/vehicle',
        'smartcar.whalesbot.vehicle.driver': ROOT/'smartcar/whalesbot/vehicle/driver',
        'smartcar.paddlebaidu': ROOT/'smartcar/paddlebaidu',
        'smartcar.paddlebaidu.infer_cs': ROOT/'smartcar/paddlebaidu/infer_cs',
        'smartcar.paddlebaidu.infer_cs.base': ROOT/'smartcar/paddlebaidu/infer_cs/base',
    }.items(): package(name, path)
    tools = ROOT/'smartcar/whalesbot/tools'
    log = load('smartcar.whalesbot.tools.log_wrap', tools/'log_wrap.py', 'smartcar.whalesbot.tools')
    util = load('smartcar.whalesbot.tools.tools_class', tools/'tools_class.py', 'smartcar.whalesbot.tools')
    sys.modules['smartcar.whalesbot.tools'].logger = log.logger; sys.modules['smartcar.whalesbot.tools'].PID = util.PID; sys.modules['smartcar.whalesbot.tools'].CountRecord = util.CountRecord
    camera = load('smartcar.whalesbot.tools.camera', tools/'camera.py', 'smartcar.whalesbot.tools')
    streamer = load('smartcar.whalesbot.tools.streamer', tools/'streamer.py', 'smartcar.whalesbot.tools')
    mecanum = load('smartcar.whalesbot.vehicle.driver.mecanum', ROOT/'smartcar/whalesbot/vehicle/driver/mecanum.py', 'smartcar.whalesbot.vehicle.driver')
    client = load('smartcar.paddlebaidu.infer_cs.base.infer_front', ROOT/'smartcar/paddlebaidu/infer_cs/base/infer_front.py', 'smartcar.paddlebaidu.infer_cs.base')
    return camera.Camera, streamer.Streamer, mecanum.MecanumDriver, client.ClintInterface, util.PID

def main():
    p=argparse.ArgumentParser(); p.add_argument('--speed',type=float,default=.08); p.add_argument('--duration',type=float,default=15.0); p.add_argument('--output-limit',type=float,default=1.2); a=p.parse_args()
    if not 0<a.speed<=.10 or not 0<a.duration<=15: raise SystemExit('speed 0..0.10; duration 0..15')
    Camera, Streamer, Driver, Client, PID = components(); car=Driver(); cap=Camera(1,320,240); stream=Streamer(); lane=Client('lane'); py=PID(5,.1,0,setpoint=0,output_limits=(-.7,.7)); pa=PID(3,0,0,setpoint=0,output_limits=(-1.5,1.5)); reason='duration_elapsed'; start=time.monotonic()
    try:
        while time.monotonic()-start<a.duration:
            frame=cap.read()
            if frame is None: reason='camera_failure'; break
            stream.update_frame(frame,'cam1'); ey,ea=lane(frame); reason=validate_output(ey,ea,a.output_limit)
            if reason: break
            car.set_velocity(a.speed,py(-ey),pa(-ea))
    except KeyboardInterrupt: reason='keyboard_interrupt'
    except Exception as exc: reason='runtime_error:'+type(exc).__name__; print(reason)
    finally:
        car.stop(); cap.close(); stream.stop(); print('SAFE_STOP reason='+reason)

if __name__=='__main__': main()
