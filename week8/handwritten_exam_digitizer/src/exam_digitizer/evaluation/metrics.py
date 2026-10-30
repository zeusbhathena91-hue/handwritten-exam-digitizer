"""Evaluation metrics (identical to the project notebook, §2.5): CER / WER, label-free CER, question-segmentation
P/R/F1, page accuracy, student-ID accuracy / wrong-ID rate / abstention, and a cluster bootstrap over booklets."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

_TOK = re.compile(r"\w+|[^\w\s]")


def norm_text(s: str) -> str:
    """NFKC, straight quotes, punctuation as separate tokens, single spaces (IAM ground truth is tokenised)."""
    s = unicodedata.normalize("NFKC", s or "")
    for a, b in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("—", "-"), ("–", "-")]:
        s = s.replace(a, b)
    return " ".join(_TOK.findall(s))


def cer_counts(ref: str, hyp: str):
    r, h = norm_text(ref), norm_text(hyp)
    return Levenshtein.distance(r, h), len(r)


def wer_counts(ref: str, hyp: str):
    r, h = norm_text(ref).split(), norm_text(hyp).split()
    return Levenshtein.distance(r, h), len(r)


def cer(ref: str, hyp: str) -> float:
    e, n = cer_counts(ref, hyp)
    return e / max(n, 1)


def eval_booklet(gt: dict, pred: dict):
    """gt: {"booklet_id", "student_id", "n_pages", "questions": [{"question","answer_text","page", "lines"?, "ood"?}]}
    pred: the contract dict. Returns (per-question rows, booklet-level counts)."""
    gt_labels = {q["question"] for q in gt["questions"]}
    ood_labels = {q["question"] for q in gt["questions"] if q.get("ood")}
    seen, fp, pred_by_label = set(), 0, {}
    for a in pred["answers"]:
        lab = a["question"]
        if lab in gt_labels and lab not in seen:
            seen.add(lab)
            pred_by_label[lab] = a
        else:
            fp += 1
    gtq = [q for q in gt["questions"] if not q.get("ood")]
    rows = []
    for q in gtq:
        a = pred_by_label.get(q["question"])
        hyp = a["answer_text"] if a else ""
        ce, cn = cer_counts(q["answer_text"], hyp)
        we, wn = wer_counts(q["answer_text"], hyp)
        rows.append(dict(booklet=gt["booklet_id"], question=q["question"], found=a is not None, ce=ce, cn=cn, we=we, wn=wn,
                         page_ok=(a is not None and a.get("page") == q["page"]), conf=(a or {}).get("confidence", np.nan),
                         ref=q["answer_text"], hyp=hyp, n_lines=len(q.get("lines", []))))
    ref_all = " ".join(q["answer_text"] for q in gtq)
    hyp_all = " ".join(a["answer_text"] for a in pred["answers"] if a["question"] not in ood_labels)
    lf_e, lf_n = cer_counts(ref_all, hyp_all)
    sid_gt, sid_pr = gt["student_id"], pred.get("student_id")
    bk = dict(booklet=gt["booklet_id"], tp=len(seen), fp=fp, fn=len(gt_labels) - len(seen), lf_e=lf_e, lf_n=lf_n,
              id_ok=(sid_pr == sid_gt), id_wrong=(sid_pr is not None and sid_pr != sid_gt), id_null=sid_pr is None,
              runtime=pred.get("runtime_s", np.nan), pages=gt.get("n_pages", pred.get("pages", 1)))
    return rows, bk


def aggregate(qdf: pd.DataFrame, bdf: pd.DataFrame) -> dict:
    tp, fp, fn = bdf.tp.sum(), bdf.fp.sum(), bdf.fn.sum()
    P = tp / max(tp + fp, 1)
    R = tp / max(tp + fn, 1)
    f = qdf[qdf.found]
    return {"CER": qdf.ce.sum() / max(qdf.cn.sum(), 1), "WER": qdf.we.sum() / max(qdf.wn.sum(), 1),
            "CER(found)": f.ce.sum() / max(f.cn.sum(), 1), "label-free CER": bdf.lf_e.sum() / max(bdf.lf_n.sum(), 1),
            "Seg P": P, "Seg R": R, "Seg F1": 2 * P * R / max(P + R, 1e-9),
            "Page acc": f.page_ok.mean() if len(f) else np.nan,
            "ID acc": bdf.id_ok.mean(), "wrong-ID": bdf.id_wrong.mean(), "ID null": bdf.id_null.mean(),
            "s/page": bdf.runtime.sum() / max(bdf.pages.sum(), 1)}


def evaluate(gts, preds: dict):
    """gts: list of ground-truth booklets; preds: {booklet_id: contract dict}. Returns (metrics, qdf, bdf)."""
    rows, bks = [], []
    for gt in gts:
        r, b = eval_booklet(gt, preds[gt["booklet_id"]])
        rows += r
        bks.append(b)
    qdf, bdf = pd.DataFrame(rows), pd.DataFrame(bks)
    return aggregate(qdf, bdf), qdf, bdf


def bootstrap_ci(qdf: pd.DataFrame, bdf: pd.DataFrame, n: int = 1000, seed: int = 0,
                 metrics=("CER", "WER", "Seg F1", "ID acc")):
    """95 % cluster-bootstrap CIs over booklets."""
    rng = np.random.default_rng(seed)
    ids = bdf.booklet.values
    qg = {k: v for k, v in qdf.groupby("booklet")}
    bg = {k: v for k, v in bdf.groupby("booklet")}
    out = {m: [] for m in metrics}
    for _ in range(n):
        s = rng.choice(ids, size=len(ids), replace=True)
        agg = aggregate(pd.concat([qg[i] for i in s]), pd.concat([bg[i] for i in s]))
        for m in metrics:
            out[m].append(agg[m])
    return {m: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for m, v in out.items()}
