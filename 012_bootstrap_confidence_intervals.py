from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score
)

#configs
n_boot = 1000
seed = 42 #for reproducibility
rng = np.random.default_rng(seed)

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

#ignoring typical bootstrapping warning in multiclass problems
warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true"
)

#bootstrapping function
def bootstrap_metrics(y_true, y_pred, n_boot=1000):

    n = len(y_true)

    bal_acc_scores = []
    macro_f1_scores = []

    for _ in range(n_boot):

        idx = rng.integers(0, n, n)

        yt = y_true[idx]
        yp = y_pred[idx]

        bal_acc_scores.append(
            balanced_accuracy_score(yt, yp)
        )

        macro_f1_scores.append(
            f1_score(
                yt,
                yp,
                average="macro",
                zero_division=0
            )
        )

    return (
        np.percentile(bal_acc_scores, [2.5, 97.5]),
        np.percentile(macro_f1_scores, [2.5, 97.5])
    )

#running bootstrap CIs
rows = []

for model_name, (csv_path, pred_col) in FILES.items():

    df = pd.read_csv(csv_path)

    y_true = df["label"].astype(str).values
    y_pred = df[pred_col].astype(str).values

    bal = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    bal_ci, f1_ci = bootstrap_metrics(
        y_true,
        y_pred,
        n_boot
    )

    rows.append({
        "Model": model_name,

        "Balanced Accuracy":
            round(bal * 100, 1),

        "BalAcc 95% CI":
            f"[{bal_ci[0]*100:.1f}, {bal_ci[1]*100:.1f}]",

        "Macro F1":
            round(f1, 3),

        "MacroF1 95% CI":
            f"[{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]"
    })

results = pd.DataFrame(rows)

print(results)

results.to_csv(
    "bootstrap_confidence_intervals.csv",
    index=False
)

print("\nSaved bootstrap_confidence_intervals.csv")