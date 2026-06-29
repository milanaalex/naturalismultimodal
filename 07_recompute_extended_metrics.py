#extended evaluation metrics

#imports
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)

#configs
project_root = Path(__file__).resolve().parents[2]
results_root = project_root / "experiments/order_supervised/results"

image_dir = results_root / "bioclip2_image_classifier"
text_root = results_root / "modernbert_text_classifier"

out_path = results_root / "extended_unimodal_metrics_summary.csv"

text_conditions = [
    "metadata_full_record_realistic",
    "metadata_lower_taxonomy_realistic",
    "metadata_strict_conservative",
]

splits = ["val", "test"]

#necessary helpers
def load_aligned_prediction_and_probs(pred_path, prob_path):
    pred_df = pd.read_csv(pred_path)
    prob_df = pd.read_csv(prob_path)

    prob_cols = [col for col in prob_df.columns if col.startswith("prob_")]

    if not prob_cols:
        raise ValueError(f"No probability columns found in {prob_path}")

    if "gbifID" not in pred_df.columns:
        raise ValueError(f"gbifID missing in prediction file: {pred_path}")

    if "gbifID" not in prob_df.columns:
        raise ValueError(f"gbifID missing in probability file: {prob_path}")

    #aligning by specimen identifier
    merged = pred_df.merge(
        prob_df[["gbifID"] + prob_cols],
        on="gbifID",
        how="inner",
    )

    if len(merged) != len(pred_df):
        print(
            f"warning: merged rows differ from predictions "
            f"({len(merged)} vs {len(pred_df)})"
        )

    #if duplicate gbifIDs, keep first exact match only
    merged = merged.drop_duplicates(subset=["gbifID"]).reset_index(drop=True)

    return merged, prob_cols

#additional evaluation metrics
def compute_extended_metrics(y_true, y_pred, probs, model, condition, split):
    n_classes = probs.shape[1]
    labels = list(range(n_classes))

    return {
        "model": model,
        "condition": condition,
        "split": split,
        "n_samples": len(y_true),
        "n_classes": n_classes,

        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "macro_precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "weighted_precision": precision_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_recall": recall_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),

        #single-label multiclass classification task, top-k accuracy equals recall@k
        "top3_accuracy": top_k_accuracy_score(
            y_true, probs, k=3, labels=labels
        ),
        "top5_accuracy": top_k_accuracy_score(
            y_true, probs, k=5, labels=labels
        ),
        "recall_at_3": top_k_accuracy_score(
            y_true, probs, k=3, labels=labels
        ),
        "recall_at_5": top_k_accuracy_score(
            y_true, probs, k=5, labels=labels
        ),
    }

#image metrics
def recompute_image_metrics():
    rows = []

    for split in splits:
        pred_path = image_dir / f"bioclip2_image_mlp_{split}_predictions.csv"
        prob_path = image_dir / f"bioclip2_image_mlp_{split}_probs.csv"

        merged, prob_cols = load_aligned_prediction_and_probs(pred_path, prob_path)

        y_true = merged["label_id"].astype(int).to_numpy()
        y_pred = merged["image_pred_id"].astype(int).to_numpy()
        probs = merged[prob_cols].to_numpy()

        rows.append(
            compute_extended_metrics(
                y_true=y_true,
                y_pred=y_pred,
                probs=probs,
                model="bioclip2_image_mlp",
                condition="image_only",
                split=split,
            )
        )

    return rows

#text metrics
def recompute_text_metrics():
    rows = []

    for condition in text_conditions:
        condition_dir = text_root / condition
        pred_prefix = f"text_{condition}"

        for split in splits:
            pred_path = condition_dir / f"modernbert_text_mlp_{split}_predictions.csv"
            prob_path = condition_dir / f"modernbert_text_mlp_{split}_probs.csv"

            merged, prob_cols = load_aligned_prediction_and_probs(pred_path, prob_path)

            y_true = merged["label_id"].astype(int).to_numpy()
            y_pred = merged[f"{pred_prefix}_pred_id"].astype(int).to_numpy()
            probs = merged[prob_cols].to_numpy()

            rows.append(
                compute_extended_metrics(
                    y_true=y_true,
                    y_pred=y_pred,
                    probs=probs,
                    model="modernbert_text_mlp",
                    condition=condition,
                    split=split,
                )
            )

    return rows

#sanity checks
def sanity_check_against_existing_metrics():
    existing_files = [
        image_dir / "bioclip2_image_mlp_metrics_summary.csv",
        text_root / "metadata_full_record_realistic/modernbert_text_mlp_metrics_summary.csv",
        text_root / "metadata_lower_taxonomy_realistic/modernbert_text_mlp_metrics_summary.csv",
        text_root / "metadata_strict_conservative/modernbert_text_mlp_metrics_summary.csv",
    ]

    print("\nSanity check source files:")
    print("Recomputed accuracy/F1 should match these original summaries.")

    for path in existing_files:
        if path.exists():
            print(f"  found: {path}")
        else:
            print(f"  missing: {path}")

#main
def main():
    rows = []
    rows.extend(recompute_image_metrics())
    rows.extend(recompute_text_metrics())

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out_path, index=False)

    print("saved:", out_path)

    sanity_check_against_existing_metrics()

    print("\nALL METRICS")
    print(metrics_df.to_string(index=False))

    print("\nTEST METRICS")
    print(metrics_df[metrics_df["split"] == "test"].to_string(index=False))


if __name__ == "__main__":
    main()
