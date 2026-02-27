# log_multiclass_latex.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Sequence, Optional
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from latex_logger import LatexLogger, fmt_float


def log_multiclass_result_to_latex(
    tex: LatexLogger,
    modelname: str,
    y_true: Optional[np.ndarray],
    y_pred: np.ndarray,
    fam_test_arr: Optional[np.ndarray] = None,
    original_test_counts: Optional[Counter] = None,   # só se quiseres “real vs detetado”
    is_kdd99_official: bool = False,
) -> None:
    tex.subsection(modelname)

    # -------------------------
    # UNLABELED: só distribuição
    # -------------------------
    if y_true is None:
        det = Counter(y_pred)
        rows = [(k, str(v), fmt_float(v / max(1, len(y_pred)), 4)) for k, v in det.most_common()]
        tex.table(
            headers=["attack", "count", "ratio"],
            rows=rows,
            caption="Distribuição detetada (unlabeled)",
            longtable=True,
            align="lrr"
        )
        return

    # -------------------------
    # Métricas globais
    # -------------------------
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    tex.kv_table(
        [
            ("Accuracy", fmt_float(acc, 6)),
            ("Precision (macro)", fmt_float(prec, 6)),
            ("Recall (macro)", fmt_float(rec, 6)),
            ("F1-score (macro)", fmt_float(f1, 6)),
        ],
        caption="Métricas globais"
    )

    # -------------------------
    # Classification report (tabela)
    # -------------------------
    rep: Dict[str, Dict[str, float]] = classification_report(
        y_true, y_pred, output_dict=True, zero_division=0
    )

    # Linhas por classe (exclui macro avg / weighted avg / accuracy)
    class_rows: List[Tuple[str, str, str, str, str]] = []
    for label, d in rep.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        class_rows.append((
            str(label),
            fmt_float(d.get("precision", 0.0), 4),
            fmt_float(d.get("recall", 0.0), 4),
            fmt_float(d.get("f1-score", 0.0), 4),
            str(int(d.get("support", 0))),
        ))

    # ordena por support desc
    class_rows.sort(key=lambda r: int(r[-1]), reverse=True)

    tex.table(
        headers=["class", "precision", "recall", "f1", "support"],
        rows=class_rows,
        caption="Classification report (por classe)",
        longtable=True,
        align="lrrrr"
    )

    # -------------------------
    # Confusion matrix (opcional)
    # -------------------------
    labels_sorted = sorted(set(map(str, np.unique(y_true))) | set(map(str, np.unique(y_pred))))
    cm = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels_sorted)

    # para não ficar gigante: mostra só top-N por suporte
    N = 15
    sup = Counter(map(str, y_true))
    top_labels = [k for k, _ in sup.most_common(N)]
    top_labels = [l for l in labels_sorted if l in top_labels]

    cm_top = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=top_labels)
    # tabela cm: primeira coluna label, depois colunas pred
    cm_rows = []
    for i, lab in enumerate(top_labels):
        cm_rows.append([lab] + [str(int(x)) for x in cm_top[i, :]])

    tex.table(
        headers=["true \\ pred"] + top_labels,
        rows=cm_rows,
        caption=f"Confusion matrix (top-{len(top_labels)} por suporte)",
        longtable=True,
        align="l" + "r" * len(top_labels)
    )

    # -------------------------
    # Real vs Detetado (só no teu modo “KDD official” se quiseres)
    # -------------------------
    if is_kdd99_official and original_test_counts is not None:
        det = Counter(map(str, y_pred))
        all_atks = sorted(set(map(str, original_test_counts.keys())) | set(det.keys()))
        rows = []
        for atk in all_atks:
            t = int(original_test_counts.get(atk, 0))
            p = int(det.get(atk, 0))
            rows.append((atk, str(t), str(p), str(p - t)))
        tex.table(
            headers=["attack", "true", "pred", "diff"],
            rows=rows,
            caption="Real vs Detetado (KDD official)",
            longtable=True,
            align="lrrr"
        )

    # -------------------------
    # Acertos/Erros por family (o que tu já fazes)
    # -------------------------
    if fam_test_arr is not None:
        stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "total": 0})
        for fam, yt, yp in zip(fam_test_arr, y_true, y_pred):
            fam = str(fam)
            stats[fam]["total"] += 1
            if yt == yp:
                stats[fam]["correct"] += 1
            else:
                stats[fam]["wrong"] += 1

        rows = []
        for fam, s in sorted(stats.items(), key=lambda kv: kv[1]["total"], reverse=True):
            total = s["total"]
            corr = s["correct"]
            wrong = s["wrong"]
            accf = (corr / total) if total else 0.0
            rows.append((fam, str(total), str(corr), str(wrong), fmt_float(accf, 4)))

        tex.table(
            headers=["family", "total", "correct", "wrong", "acc"],
            rows=rows,
            caption="Acertos/Erros por family",
            longtable=True,
            align="lrrrr"
        )
