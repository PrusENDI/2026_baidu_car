from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import paddle
from .action_loss import masked_smooth_l1_loss
from .dataset import LaneDataset
from .model import CnnModel


def load_manifest(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows: raise ValueError(f"empty manifest: {path}")
    for row in rows:
        if not {"image_path", "speed_demand", "kappa_action"} <= row.keys():
            raise ValueError("manifest rows require image_path, speed_demand, kappa_action")
        row.setdefault("target_mask", [1.0, 1.0]); row.setdefault("split", "train")
    return rows


def _batch(dataset, indices, batch_size):
    images, targets, masks = [], [], []
    for index in indices:
        image, target, mask = dataset[index]; images.append(image); targets.append(target); masks.append(mask)
        if len(images) == batch_size:
            yield paddle.to_tensor(np.stack(images)), paddle.to_tensor(np.stack(targets)), paddle.to_tensor(np.stack(masks)); images, targets, masks = [], [], []
    if images: yield paddle.to_tensor(np.stack(images)), paddle.to_tensor(np.stack(targets)), paddle.to_tensor(np.stack(masks))


def predict_action(model, rows: list[dict], *, batch_size=64):
    dataset = LaneDataset(rows, training=False, return_target_mask=True); predictions=[]; targets=[]; masks=[]; model.eval()
    with paddle.no_grad():
        for images, target, mask in _batch(dataset, range(len(dataset)), batch_size):
            predictions.append(model(images).numpy()); targets.append(target.numpy()); masks.append(mask.numpy())
    if not predictions:
        raise ValueError("cannot predict an empty action row set")
    return np.concatenate(predictions), np.concatenate(targets), np.concatenate(masks)


def evaluate_model(model, rows: list[dict], *, batch_size=64):
    prediction, target, mask = predict_action(model, rows, batch_size=batch_size); valid=mask.sum(axis=0)
    mae=np.divide((np.abs(prediction-target)*mask).sum(axis=0), np.maximum(valid, 1e-6))
    mae=np.where(valid > 0, mae, np.nan)
    return {"count":len(rows),"valid_speed":int(valid[0]),"valid_kappa":int(valid[1]),"speed_mae":None if valid[0] == 0 else float(mae[0]),"kappa_mae":None if valid[1] == 0 else float(mae[1]),"masked_mae":float(np.sum(np.abs(prediction-target)*mask)/max(mask.sum(),1e-6))}


def train_action(*, manifest: Path, output: Path, initial_checkpoint: Path | None=None, epochs=20, batch_size=64, learning_rate=1e-5, kappa_weight=1.0, device="gpu", smoke=False):
    paddle.set_device("gpu:0" if device == "gpu" else "cpu"); rows=load_manifest(manifest)
    train_rows=[r for r in rows if r.get("split")=="train"]; val_rows=[r for r in rows if r.get("split") in {"val","validation"}]
    if not train_rows: raise ValueError("manifest has no train rows")
    if not val_rows: val_rows=train_rows
    model=CnnModel()
    if initial_checkpoint:
        state=paddle.load(str(initial_checkpoint))
        if set(state)!=set(model.state_dict()): raise ValueError("checkpoint keys do not match CnnModel")
        model.set_state_dict(state)
    optimizer=paddle.optimizer.Adam(learning_rate=learning_rate, parameters=model.parameters()); output.mkdir(parents=True,exist_ok=True)
    history=[]; best=math.inf; limit=2 if smoke else epochs; train_ds=LaneDataset(
        train_rows, training=True, return_target_mask=True,
        horizontal_flip_probability=0.0)
    for epoch in range(1,limit+1):
        train_ds.set_epoch(epoch); model.train(); losses=[]; order=np.random.default_rng(20260817+epoch).permutation(len(train_ds))
        for images,target,mask in _batch(train_ds,order,batch_size):
            loss,_=masked_smooth_l1_loss(model(images),target,mask,kappa_weight=kappa_weight); loss.backward(); optimizer.step(); optimizer.clear_grad(); losses.append(float(loss))
        metrics=evaluate_model(model,val_rows,batch_size=batch_size); metrics.update(epoch=epoch,train_loss=float(np.mean(losses))); history.append(metrics)
        paddle.save(model.state_dict(),str(output/f"epoch_{epoch}.pdparams"))
        if metrics["masked_mae"]<best: best=metrics["masked_mae"]; paddle.save(model.state_dict(),str(output/"best.pdparams"))
    (output/"training_report.json").write_text(json.dumps({"output_semantics":["speed_demand","kappa_action"],"history":history},indent=2)+"\n",encoding="utf-8"); return output


def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--initial-checkpoint",type=Path); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--learning-rate",type=float,default=1e-5); p.add_argument("--kappa-weight",type=float,default=1.0); p.add_argument("--device",choices=("gpu","cpu"),default="gpu"); p.add_argument("--smoke",action="store_true"); a=p.parse_args(); print(train_action(**vars(a)))
if __name__ == "__main__": main()
