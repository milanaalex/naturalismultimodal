from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


# ============================================================
# CONFIG
# ============================================================

n_boot = 1000
seed = 42

output_path = Path("bootstrap_confidence_intervals_thesis_ready.csv")

FILES = {
    "BioCLIP2": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/bioclip2_image_classifier/bioclip2_image_mlp_test_predictions.csv",
        "image_pred_label"
    ),

    "ModernBERT Raw": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/modernbert_text_classifier/metadata_full_record_realistic/modernbert_text_mlp_test_predictions.csv",
        "text_metadata_full_record_realistic_pred_label"
    ),

    "ModernBERT Order-Cleaned": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/modernbert_text_classifier/metadata_lower_taxonomy_realistic/modernbert_text_mlp_test_predictions.csv",
        "text_metadata_lower_taxonomy_realistic_pred_label"
    ),

    "ModernBERT Tax. Reduced": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/modernbert_text_classifier/metadata_strict_conservative/modernbert_text_mlp_test_predictions.csv",
        "text_metadata_strict_conservative_pred_label"
    ),

    "Gated Fusion Raw": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/multimodal_fusion/metadata_full_record_realistic/gated_embedding_fusion/gated_embedding_fusion_test_predictions.csv",
        "gated_metadata_full_record_realistic_pred_label"
    ),

    "Gated Fusion Order-Cleaned": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/multimodal_fusion/metadata_lower_taxonomy_realistic/gated_embedding_fusion/gated_embedding_fusion_test_predictions.csv",
        "gated_metadata_lower_taxonomy_realistic_pred_label"
    ),

    "Gated Fusion Tax. Reduced": (
        "/Users/milanaalexeeva/Desktop/CMollusca/experiments/order_supervised/results/multimodal_fusion/metadata_strict_conservative/gated_embedding_fusion/gated_embedding_fusion_test_predictions.csv",
        "gated_metadata_strict_conservative_pred_label"
    )
}


# ============================================================
# METRICS
# ============================================================

def fixed_label_balanced_accuracy(y_true, y_pred, labels):
    recalls = []

    for label in labels:
        mask = y_true == label

        if mask.sum() == 0:
            continue

        recall = (y_pred[mask] == label).sum() / mask.sum()
        recalls.append(recall)

    return np.mean(recalls)


def fixed_label_macro_f1(y_true, y_pred, labels):
    return f1_score(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0
    )


def bootstrap_metrics(y_true, y_pred, labels, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(y_true)

    bal_scores = []
    f1_scores = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        yt = y_true[idx]
        yp = y_pred[idx]

        bal_scores.append(
            fixed_label_balanced_accuracy(yt, yp, labels)
        )

        f1_scores.append(
            fixed_label_macro_f1(yt, yp, labels)
        )

    bal_ci = np.percentile(bal_scores, [2.5, 97.5])
    f1_ci = np.percentile(f1_scores, [2.5, 97.5])

    return bal_ci, f1_ci


# ============================================================
# RUN
# ============================================================

rows = []

for i, (model_name, (csv_path, pred_col)) in enumerate(FILES.items()):

    df = pd.read_csv(csv_path)

    y_true = df["label"].astype(str).values
    y_pred = df[pred_col].astype(str).values

    labels = np.unique(y_true)

    bal = fixed_label_balanced_accuracy(y_true, y_pred, labels)
    macro_f1 = fixed_label_macro_f1(y_true, y_pred, labels)

    bal_ci, f1_ci = bootstrap_metrics(
        y_true=y_true,
        y_pred=y_pred,
        labels=labels,
        n_boot=n_boot,
        seed=seed + i
    )

    rows.append({
        "Model": model_name,

        "Balanced Acc. (%, 95% CI)":
            f"{bal * 100:.1f} [{bal_ci[0] * 100:.1f}, {bal_ci[1] * 100:.1f}]",

        "Macro-F1 (95% CI)":
            f"{macro_f1:.3f} [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]",

        "n_test": len(y_true),
        "n_classes": len(labels)
    })


results = pd.DataFrame(rows)

print("\nThesis-ready bootstrap confidence intervals")
print("=" * 80)
print(results.to_string(index=False))

results.to_csv(output_path, index=False)

print(f"\nSaved: {output_path.resolve()}")