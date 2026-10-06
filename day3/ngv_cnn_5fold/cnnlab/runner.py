from __future__ import annotations
import copy, hashlib, json, time, uuid, webbrowser
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, read_json, write_json, write_csv, set_seed, resolve_device, environment, project_lock, ids_hash, sha_file
from .data import load_bundle, nested_subset, folds_for, make_loader, CLASSES
from .models import build_model, NAMES
from .engine import fit_fold, refit, evaluate
from .reporting import variant_report, comparison, table, page

ALLOWED = {"domain","model","epochs","batch_size","learning_rate","weight_decay","dropout","width",
           "augmentation","seed","cv_seed","subset_seed","train_count","cpu_threads","name"}


def validate_config(cfg):
    unknown=set(cfg)-ALLOWED
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    if cfg["domain"] not in CLASSES or cfg["model"] not in NAMES:
        raise ValueError("Unsupported domain/model")
    if cfg["epochs"]<1 or cfg["batch_size"]<2 or cfg["learning_rate"]<=0 or cfg["weight_decay"]<0:
        raise ValueError("Invalid epochs/batch size/learning rate/weight decay")
    if not 0<=cfg["dropout"]<1 or cfg["width"]<2 or cfg["cpu_threads"]<1:
        raise ValueError("Invalid dropout, width, or thread count")
    if not cfg["name"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in cfg["name"]):
        raise ValueError("Variant name must use letters/digits/_/-")


def run_variant(cfg, bundle, out, device, env):
    validate_config(cfg)
    x,y,ids,meta=bundle
    full_pool=np.asarray(meta["split"]["train_pool"])
    test_idx=np.asarray(meta["split"]["test"])
    pool=nested_subset(full_pool,y,cfg["train_count"],cfg["subset_seed"])
    pairs=folds_for(pool,y,cfg["cv_seed"])
    out.mkdir(parents=True)
    write_json(out/"config.json",cfg)
    write_json(out/"environment.json",env)
    write_json(out/"indices.json",{"train_pool":pool.tolist(),"test":test_idx.tolist(),
        "source_test_ids":ids[test_idx].tolist(),
        "folds":[{"fold":i+1,"train":a.tolist(),"validation":b.tolist()} for i,(a,b) in enumerate(pairs)],
        "dataset_sha256":meta["npz_sha256"],"test_hash":meta["split"]["test_hash"]})
    histories,fold_rows,epochs=[],[],[]
    oof=np.full((len(y),len(CLASSES[cfg["domain"]])),np.nan,dtype=np.float32)
    parameters=None
    fit_start=time.perf_counter()
    cv_start=time.perf_counter()
    for fold,(tr,va) in enumerate(pairs,1):
        if set(tr)&set(va) or set(test_idx)&(set(tr)|set(va)):
            raise AssertionError("Data leakage in split")
        seed=cfg["seed"]+fold
        set_seed(seed,cfg["cpu_threads"])
        model=build_model(cfg["model"],x.shape[-1],len(CLASSES[cfg["domain"]]),cfg["width"],cfg["dropout"]).to(device)
        parameters=sum(p.numel() for p in model.parameters())
        training=make_loader(x,y,tr,cfg["domain"],cfg["batch_size"],True,seed,cfg["augmentation"],device.type=="cuda")
        validation=make_loader(x,y,va,cfg["domain"],cfg["batch_size"],False,seed,False,device.type=="cuda")
        hist,best_epoch,metrics,p,_=fit_fold(model,training,validation,device,cfg,fold)
        oof[va]=p
        histories.append(hist)
        epochs.append(best_epoch)
        fold_rows.append({"fold":fold,"n_train":len(tr),"n_validation":len(va),"best_epoch":best_epoch,
            "validation_loss":metrics["loss"],"validation_accuracy":metrics["accuracy"],
            "validation_macro_f1":metrics["macro_f1"],"train_seconds":sum(r["train_seconds"] for r in hist),
            "validation_seconds":sum(r["val_seconds"] for r in hist)})
        # One best checkpoint per fold, as CPU tensors; no optimizer state carried between folds.
        torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":cfg,"epoch":best_epoch},out/f"fold_{fold}_best.pt")
        write_csv(out/"history_cv.csv",[r for h in histories for r in h])
        write_csv(out/"fold_metrics.csv",fold_rows)
        del model,training,validation
    cv_total=time.perf_counter()-cv_start
    if not np.isfinite(oof[pool]).all():
        raise AssertionError("OOF prediction missing")
    # Median of 5 integer best epochs is an integer. No test information is used.
    selected=int(np.median(epochs))
    set_seed(cfg["seed"]+9000,cfg["cpu_threads"])
    model=build_model(cfg["model"],x.shape[-1],len(CLASSES[cfg["domain"]]),cfg["width"],cfg["dropout"]).to(device)
    training=make_loader(x,y,pool,cfg["domain"],cfg["batch_size"],True,cfg["seed"]+9000,cfg["augmentation"],device.type=="cuda")
    final_history=refit(model,training,device,cfg,selected)
    torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"config":cfg,
        "selected_epochs":selected,"test_hash":meta["split"]["test_hash"]},out/"final_model.pt")
    total_fit=time.perf_counter()-fit_start
    test_loader=make_loader(x,y,test_idx,cfg["domain"],cfg["batch_size"],False,0,False,device.type=="cuda")
    test,test_p,test_y=evaluate(model,test_loader,device,warmup=True)
    result={"name":cfg["name"],"domain":cfg["domain"],"model":cfg["model"],"model_description":NAMES[cfg["model"]],
        "n_train_pool":len(pool),"n_fold_train":len(pairs[0][0]),"n_fold_validation":len(pairs[0][1]),"n_test":len(test_idx),
        "folds":5,"cv_loss_mean":float(np.mean([r["validation_loss"] for r in fold_rows])),
        "cv_accuracy_mean":float(np.mean([r["validation_accuracy"] for r in fold_rows])),
        "cv_accuracy_std":float(np.std([r["validation_accuracy"] for r in fold_rows],ddof=1)),
        "selected_epochs":selected,"test_loss":test["loss"],"test_accuracy":test["accuracy"],
        "test_macro_f1":test["macro_f1"],"cv_total_seconds":cv_total,
        "refit_train_seconds":sum(r["train_seconds"] for r in final_history),"total_fit_seconds":total_fit,
        "test_seconds":test["seconds"],"test_ms_per_image":1000*test["seconds"]/len(test_idx),
        "parameters":parameters,"device":env["device_name"],"seed":cfg["seed"],"learning_rate":cfg["learning_rate"],
        "batch_size":cfg["batch_size"],"dropout":cfg["dropout"],"weight_decay":cfg["weight_decay"],
        "max_cv_epochs":cfg["epochs"],"augmentation":cfg["augmentation"],
        "dataset_sha256":meta["npz_sha256"],"train_pool_hash":ids_hash(ids[pool]),"test_hash":meta["split"]["test_hash"],
        "evaluation_protocol":f"internal_holdout{len(test_idx)}_after_5fold_cv_and_refit","pretrained":False}
    np.savez_compressed(out/"predictions.npz",test_indices=test_idx,test_y=test_y,test_probabilities=test_p,
                        oof_indices=pool,oof_y=y[pool],oof_probabilities=oof[pool])
    variant_report(out,cfg,result,fold_rows,histories,final_history,x[test_idx],test_y,test_p,y[pool],oof[pool],CLASSES[cfg["domain"]])
    result["status"]="complete"
    write_json(out/"metrics.json",result)
    print(f"[COMPLETE] {cfg['name']}: fixed test accuracy={test['accuracy']:.4f}, loss={test['loss']:.4f}",flush=True)
    return result


def run_plan(plan_name, device="auto", open_report=True, overrides=None, root=ROOT):
    root=Path(root)
    plan=read_json(root/"configs"/plan_name)
    base=plan["base"]
    if overrides:
        base.update(overrides)
    variants=[]
    for item in plan["variants"]:
        cfg={**base,**item}
        validate_config(cfg)
        variants.append(cfg)
    if len({c["name"] for c in variants}) != len(variants):
        raise ValueError("Variant names must be unique")
    bundle=load_bundle(root/"data",base["domain"])
    out=root/"results"/(datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+Path(plan_name).stem+"_"+uuid.uuid4().hex[:6])
    with project_lock(root/"results"):
        selected_device=resolve_device(device)
        env=environment(selected_device)
        out.mkdir(parents=True)
        write_json(out/"plan.json",plan)
        write_json(out/"environment.json",env)
        write_json(out/"source_hashes.json",{str(p.relative_to(ROOT)):sha_file(p) for p in (ROOT/"cnnlab").glob("*.py")})
        write_json(out/"status.json",{"status":"running"})
        try:
            rows=[]
            for cfg in variants:
                print(f"\n=== {cfg['name']} | {cfg['domain']} | {cfg['model']} | pool={cfg['train_count']} ===",flush=True)
                rows.append(run_variant(cfg,bundle,out/cfg["name"],selected_device,env))
                comparison(out,rows,plan["title"],data_size=plan.get("kind")=="data_size")
                write_json(out/"completed_variants.json",rows)
            if len({r["test_hash"] for r in rows}) != 1:
                raise AssertionError("Comparison test sets are different")
            write_json(out/"status.json",{"status":"complete","variants":len(rows)})
            write_json(root/"results"/f"latest_{Path(plan_name).stem}.json",{"path":str(out.resolve())})
        except BaseException as exc:
            write_json(out/"status.json",{"status":"interrupted_or_failed","message":str(exc)})
            raise
    print(f"\nReport: {out/'report.html'}",flush=True)
    if open_report:
        webbrowser.open((out/"report.html").resolve().as_uri())
    return out


def collect(root=ROOT,open_report=True):
    root=Path(root)
    rows=[]
    for p in sorted((root/"results").glob("*/*/metrics.json")):
        r=read_json(p)
        if r.get("status")=="complete":
            r["run_directory"]=str(p.parent.relative_to(root))
            rows.append(r)
    if not rows:
        raise RuntimeError("No completed experiments found")
    out=root/"results"/"summary"
    out.mkdir(exist_ok=True)
    write_csv(out/"all_results.csv",rows)
    page(out/"report.html","전체 실습 결과",table(rows,["domain","name","model","n_train_pool","cv_accuracy_mean",
        "test_accuracy","test_loss","cv_total_seconds","refit_train_seconds","test_seconds","run_directory"])+
        "<p>도메인·장치·데이터 hash가 다르면 직접적인 성능 순위를 만들지 마세요. 최신 실행만이 아니라 완료된 모든 실행이 들어 있습니다.</p>")
    if open_report:
        webbrowser.open((out/"report.html").resolve().as_uri())
    return out
