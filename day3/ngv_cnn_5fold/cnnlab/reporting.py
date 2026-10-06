from __future__ import annotations
import html, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from .common import write_csv

STYLE = '''body{font-family:Segoe UI,Malgun Gothic,sans-serif;max-width:1150px;margin:32px auto;padding:0 24px;line-height:1.65;background:#fafbfd;color:#172638}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border:1px solid #d4dce5;padding:8px;text-align:left}th{background:#edf2f7}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#edf2f7;padding:16px}.plots img{max-width:720px;width:100%;display:block;margin:20px 0}.gallery{display:flex;gap:16px;flex-wrap:wrap}.card{width:155px;border:1px solid #cdd6e1;padding:10px;background:white}.card img{width:144px;height:144px;object-fit:contain;image-rendering:auto}.note{border-left:4px solid #516b88;padding:10px 16px;background:#edf2f7}.scroll{overflow-x:auto}a{color:#174c81}'''


def table(rows, keys=None):
    if not rows:
        return "<p>No completed results.</p>"
    keys = keys or list(rows[0])
    def cell(v):
        if isinstance(v, float):
            return f"{v:.5g}"
        return html.escape(str(v))
    return "<div class='scroll'><table><thead><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in keys) + \
        "</tr></thead><tbody>" + "".join("<tr>"+"".join(f"<td>{cell(r.get(k,''))}</td>" for k in keys)+"</tr>" for r in rows) + "</tbody></table></div>"


def page(path, title, body):
    path.write_text(f"<!doctype html><html lang='ko'><meta charset='utf-8'><title>{html.escape(title)}</title>"
                    f"<style>{STYLE}</style><body><h1>{html.escape(title)}</h1>{body}</body></html>", encoding="utf-8")


def curve(path, histories, key, ylabel):
    values = np.asarray([[r[key] for r in hist] for hist in histories])
    epochs = np.arange(1, values.shape[1]+1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, vals in enumerate(values):
        ax.plot(epochs, vals, alpha=.28, linewidth=1, label=f"fold {i+1}")
    ax.errorbar(epochs, values.mean(0), yerr=values.std(0, ddof=1), marker="o", linewidth=2, label="5-fold mean +/- SD")
    ax.set(xlabel="Epoch", ylabel=ylabel, title=key.replace("_", " "))
    ax.set_xticks(epochs)
    if "accuracy" in key:
        ax.set_ylim(0, 1)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def confusion(path, y, pred, names, title, normalized=False):
    cm = confusion_matrix(y, pred, labels=np.arange(len(names)))
    data = cm / np.maximum(cm.sum(1, keepdims=True), 1) if normalized else cm
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(data)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(names)), yticks=np.arange(len(names)),
           xticklabels=names, yticklabels=names, xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(names)):
        for j in range(len(names)):
            text = f"{data[i,j]:.2f}" if normalized else str(cm[i,j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", alpha=.55))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return cm


def gallery(out, x, y, p, indices, names, stem, limit=18):
    out.mkdir(exist_ok=True)
    cards = []
    for j, idx in enumerate(indices[:limit]):
        im = x[idx]
        path = out / f"{stem}_{j:02d}.png"
        Image.fromarray(im[:, :, 0] if im.shape[-1]==1 else im).save(path)
        pred = int(p[idx].argmax())
        cards.append(f"<div class='card'><img src='examples/{path.name}' alt='sample'><br>"
                     f"True: {names[int(y[idx])]}<br>Pred: {names[pred]}<br>p: {p[idx,pred]:.3f}</div>")
    return "<div class='gallery'>"+"".join(cards)+"</div>" if cards else "<p>No misclassified samples in this evaluation.</p>"


def variant_report(out, cfg, metrics, fold_rows, histories, refit_history, test_x, test_y, test_p, oof_y, oof_p, names):
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out/"fold_metrics.csv", fold_rows)
    write_csv(out/"history_cv.csv", [r for h in histories for r in h])
    write_csv(out/"history_refit.csv", refit_history)
    images = []
    for key in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
        name = f"{key}.png"
        curve(out/name, histories, key, "Loss" if "loss" in key else "Accuracy")
        images.append(name)
    test_pred = test_p.argmax(1)
    cm = confusion(out/"confusion_test.png", test_y, test_pred, names, "Fixed holdout - final refit model")
    confusion(out/"confusion_test_normalized.png", test_y, test_pred, names, "Fixed holdout - row normalized", True)
    confusion(out/"confusion_oof.png", oof_y, oof_p.argmax(1), names, "Out-of-fold diagnostic predictions")
    write_csv(out/"confusion_test.csv", [{"true_class": names[i], **dict(zip(names, map(int,row)))} for i,row in enumerate(cm)])
    cr = classification_report(test_y, test_pred, labels=np.arange(len(names)), target_names=names, zero_division=0, output_dict=True)
    write_csv(out/"per_class.csv", [{"class":n, **cr[n]} for n in names])
    write_csv(out/"test_predictions.csv", [{"sample_index":i,"true":int(test_y[i]),"pred":int(test_pred[i]),
                                         **{f"p_{n}":float(test_p[i,j]) for j,n in enumerate(names)}} for i in range(len(test_y))])
    body = f"<p class='note'>선택된 학습 풀 {metrics['n_train_pool']}장 내부에서 5-fold CV. 고정 시험 {metrics['n_test']}장은 최종 재학습 뒤에만 평가합니다. " \
           "공식 전체 test benchmark가 아닌 교육용 내부 holdout입니다. Test를 반복 비교하며 추가 튜닝하면 독립 최종평가로 볼 수 없습니다.</p>"
    body += "<h2>실행 결과</h2>" + table([metrics])
    body += "<h2>5-fold 결과</h2>" + table(fold_rows)
    body += "<p>CV 표준편차는 fold 간 변동이며 test 정확도의 신뢰구간이 아닙니다. " \
            "OOF는 각 fold의 validation loss로 선택된 checkpoint의 진단 결과입니다.</p>"
    body += "<h2>Epoch별 학습·검증 곡선</h2><p>학습 loss에는 augmentation과 학습 모드가 적용되며, 검증은 평가 모드입니다.</p><div class='plots'>"
    body += "".join(f"<img src='{n}' alt='{n}'>" for n in images)+"</div>"
    body += "<h2>Confusion matrix</h2><div class='plots'>"+"".join(f"<img src='{n}' alt='{n}'>" for n in ["confusion_test.png","confusion_test_normalized.png","confusion_oof.png"])+"</div>"
    # Deterministic display selection, not a cherry-picked set of correct examples.
    selected = np.random.default_rng(15).permutation(len(test_y))
    body += "<h2>시험 이미지 예측</h2>"+gallery(out/"examples",test_x,test_y,test_p,selected,names,"sample")
    body += "<h2>실제 오분류</h2>"+gallery(out/"examples",test_x,test_y,test_p,np.flatnonzero(test_y != test_pred),names,"wrong")
    body += "<h2>실행 설정</h2><pre>"+html.escape(json.dumps(cfg,ensure_ascii=False,indent=2))+"</pre>"
    body += "<p><a href='../report.html'>전체 비교 보고서</a></p>"
    page(out/"report.html", f"{cfg['domain']} | {cfg['name']}", body)


def comparison(out, rows, title, data_size=False):
    keys = ["name","domain","model","n_train_pool","n_fold_train","n_test","cv_loss_mean","cv_accuracy_mean",
            "cv_accuracy_std","selected_epochs","test_loss","test_accuracy","test_macro_f1",
            "cv_total_seconds","refit_train_seconds","test_seconds","test_ms_per_image","parameters"]
    write_csv(out/"comparison.csv", rows)
    body = f"<p class='note'>같은 도메인의 고정 test_hash와 시험 이미지 {rows[0]['n_test'] if rows else '-'}장을 유지합니다. " \
           "모든 조건은 사전학습 없이 새 가중치로 시작합니다. 시간은 같은 장치에서 순차 실행한 값으로 비교하세요.</p>"
    body += table(rows,keys)
    for metric in ["test_accuracy","test_loss","cv_accuracy_mean","cv_loss_mean","refit_train_seconds","cv_total_seconds","test_seconds"]:
        fig, ax = plt.subplots(figsize=(8,4.6))
        xs = [r["n_train_pool"] for r in rows] if data_size else list(range(len(rows)))
        ys = [r[metric] for r in rows]
        ax.plot(xs,ys,marker="o")
        ax.set(xlabel="Training-pool size" if data_size else "Condition", ylabel=metric.replace("_"," "),title=metric)
        if not data_size:
            ax.set_xticks(xs, [r["name"] for r in rows], rotation=25, ha="right")
        else:
            ax.set_xticks(xs)
        if "accuracy" in metric:
            ax.set_ylim(0, 1)
        ax.grid(alpha=.25)
        fig.tight_layout()
        filename=f"compare_{metric}.png"
        fig.savefig(out/filename,dpi=130)
        plt.close(fig)
        body+=f"<div class='plots'><img src='{filename}' alt='{metric}'></div>"
    if rows:
        best = min(rows,key=lambda r:r["cv_loss_mean"])
        body+=f"<p>CV loss 기준 조건: <b>{html.escape(best['name'])}</b>. 이 선택에 test loss/accuracy는 사용하지 않았습니다.</p>"
    body+="<h2>개별 실행</h2>"+"".join(f"<p><a href='{html.escape(r['name'])}/report.html'>{html.escape(r['name'])}</a></p>" for r in rows)
    body+="<p>CSV는 UTF-8 BOM으로 저장되어 Excel에서 열 수 있습니다. 정확도는 0~1 비율입니다. " \
          "test_seconds는 이미 메모리에 읽힌 이미지의 변환·배치 전달·추론·출력 수집 시간을 포함하며, " \
          "모델 로딩·초기 2회 warm-up·그림 저장·브라우저 렌더링은 제외합니다.</p>"
    page(out/"report.html",title,body)
