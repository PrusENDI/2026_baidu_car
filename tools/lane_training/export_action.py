from __future__ import annotations
import argparse,hashlib,json,tarfile
from pathlib import Path
import sys,numpy as np,paddle
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from lane_training.action_training import load_manifest
from lane_training.dataset import LaneDataset
from lane_training.model import CnnModel
def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): d.update(chunk)
    return d.hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--device",choices=("gpu","cpu"),default="gpu"); a=p.parse_args(); paddle.set_device("gpu:0" if a.device=="gpu" else "cpu")
    if a.output.exists(): raise SystemExit(f"output exists: {a.output}")
    rows=[r for r in load_manifest(a.manifest) if r.get("split")!="train"][:32] or load_manifest(a.manifest)[:32]; ds=LaneDataset(rows,training=False,return_target_mask=True); inputs=paddle.to_tensor(np.stack([ds[i][0] for i in range(len(ds))])); m=CnnModel(); m.set_state_dict(paddle.load(str(a.checkpoint))); m.eval()
    with paddle.no_grad(): dynamic=m(inputs).numpy()
    a.output.mkdir(parents=True); static=paddle.jit.to_static(m,input_spec=[paddle.static.InputSpec([None,3,128,128],dtype="float32")],full_graph=True); paddle.jit.save(static,str(a.output/"cnn_lane")); loaded=paddle.jit.load(str(a.output/"cnn_lane")); loaded.eval()
    with paddle.no_grad(): exported=loaded(inputs).numpy()
    difference=float(np.max(np.abs(dynamic-exported)))
    if not np.isfinite(difference) or difference>1e-5: raise RuntimeError(f"dynamic/static mismatch: {difference}")
    model_filename = "cnn_lane.pdmodel" if (a.output / "cnn_lane.pdmodel").exists() else "cnn_lane.json"
    metadata={"output_semantics":["speed_demand","kappa_action"],"control_profile":"kappa_action","lane_control":{"profile":"kappa_action","max_forward_speed":0.30,"min_forward_speed":0.12,"full_turn_curvature_m_inv":5.0,"speed_curve_exponent":1.5,"max_deceleration_mps2":0.194,"max_acceleration_mps2":0.129,"max_angular_speed":1.50},"input_size":[128,128],"source_image_size":[320,240],"model_filename":model_filename,"params_filename":"cnn_lane.pdiparams","checkpoint_sha256":sha256(a.checkpoint),"dynamic_static_max_abs_difference":difference,"tested_batch_size":len(rows)}; (a.output/"deployment.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    archive=a.output.parent/f"{a.output.name}.tgz"
    if archive.exists(): raise FileExistsError(f"archive exists: {archive}")
    with tarfile.open(archive,"w:gz") as stream: stream.add(a.output,arcname=a.output.name)
    metadata.update(archive=str(archive),archive_sha256=sha256(archive)); print(json.dumps(metadata,indent=2))
if __name__=="__main__": main()
