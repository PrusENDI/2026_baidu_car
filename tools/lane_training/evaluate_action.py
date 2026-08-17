from __future__ import annotations
import argparse,json
from pathlib import Path
import sys,paddle
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from lane_training.action_training import evaluate_model,load_manifest
from lane_training.model import CnnModel
def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--split",default="validation"); p.add_argument("--output",type=Path); p.add_argument("--device",choices=("gpu","cpu"),default="gpu"); a=p.parse_args(); paddle.set_device("gpu:0" if a.device=="gpu" else "cpu"); rows=[r for r in load_manifest(a.manifest) if r.get("split")==a.split]
    if not rows: raise SystemExit(f"no rows for split {a.split}")
    m=CnnModel(); m.set_state_dict(paddle.load(str(a.checkpoint))); result=evaluate_model(m,rows); result.update(checkpoint=str(a.checkpoint),output_semantics=["speed_demand","kappa_action"]); text=json.dumps(result,indent=2)+"\n"; print(text,end="")
    if a.output: a.output.write_text(text,encoding="utf-8")
if __name__=="__main__": main()
