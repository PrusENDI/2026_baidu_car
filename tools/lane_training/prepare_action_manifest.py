from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from lane_training.manifest import load_cv_action_session,load_manual_action_session

def main():
    p=argparse.ArgumentParser(); p.add_argument("--cv-json",action="append",default=[]); p.add_argument("--manual-json",action="append",default=[]); p.add_argument("--output",type=Path,required=True); a=p.parse_args(); rows=[]
    for kind,values,loader in (("cv",a.cv_json,load_cv_action_session),("manual",a.manual_json,load_manual_action_session)):
        for value in values:
            raw,sep,split=value.rpartition(":")
            if not sep or split not in {"train", "val", "validation", "test"}:
                raw,split=value,"train"
            path=Path(raw)
            for row in loader(path): row["split"]=split; row["source"]=kind; rows.append(row)
    if not rows: p.error("at least one --cv-json or --manual-json is required")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    print(f"rows={len(rows)} output={a.output}")
if __name__=="__main__": main()
