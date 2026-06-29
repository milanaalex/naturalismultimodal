#model performance by order rarity bucket, for clarity used common (>100 records) and underrepresented(<100) categories

#imports
from pathlib import Path
import pandas as pd

#configs
data_path = Path("/Users/milanaalexeeva/Desktop/CMollusca")
results = data_path / "experiments/order_supervised/results"

model_files = {
    "BioCLIP": (
        results / "bioclip2_image_classifier"
        / "bioclip2_image_mlp_test_predictions.csv"
    ),

    "Text_full": (
        results / "modernbert_text_classifier"
        / "metadata_full_record_realistic"
        / "modernbert_text_mlp_test_predictions.csv"
    ),

    "Text_lower": (
        results / "modernbert_text_classifier"
        / "metadata_lower_taxonomy_realistic"
        / "modernbert_text_mlp_test_predictions.csv"
    ),

    "Text_strict": (
        results / "modernbert_text_classifier"
        / "metadata_strict_conservative"
        / "modernbert_text_mlp_test_predictions.csv"
    ),

    "Weighted_full": (
        results / "multimodal_fusion"
        / "metadata_full_record_realistic"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Gated_full": (
        results / "multimodal_fusion"
        / "metadata_full_record_realistic"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),

    "Weighted_lower": (
        results / "multimodal_fusion"
        / "metadata_lower_taxonomy_realistic"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Gated_lower": (
        results / "multimodal_fusion"
        / "metadata_lower_taxonomy_realistic"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),

    "Weighted_strict": (
        results / "multimodal_fusion"
        / "metadata_strict_conservative"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Gated_strict": (
        results / "multimodal_fusion"
        / "metadata_strict_conservative"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),
}

frequency_mapping = {
    "common": "Common",
    "medium": "Medium",
    "rare": "Rare",
    "very_rare": "Very Rare",
    # ultra_rare (<3 records) is intentionally not mapped because ultra-rare is excluded from train val test supervised split
}

group_order = [
    "Common",
    "Medium",
    "Rare",
    "Very Rare",
]

#helpers
def find_correct_column(df, model_name):
    candidates = [c for c in df.columns if c.endswith("_correct")]

    if len(candidates) == 0:
        raise ValueError(f"{model_name}: no *_correct column found")

    if "image_correct" in candidates:
        return "image_correct"

    if len(candidates) > 1:
        print(
            f"Warning: {model_name} has multiple *_correct columns. "
            f"Using first detected column: {candidates[0]}"
        )

    return candidates[0]


def load_model_predictions(model_name, path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file for {model_name}: {path}")

    print(f"Loading {model_name}: {path}")

    df = pd.read_csv(path)

    correct_col = find_correct_column(df, model_name)

    keep_cols = [
        "gbifID",
        correct_col,
    ]

    # BioCLIP file is used as the metadata anchor because it contains order_rarity_global and labels
    if model_name == "BioCLIP":
        keep_cols.extend([
            "order_rarity_global",
            "label",
        ])

    missing_cols = [c for c in keep_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{model_name}: missing columns {missing_cols}")

    out = df[keep_cols].copy()

    out = out.rename(columns={
        correct_col: model_name,
    })

    audit_row = {
        "model": model_name,
        "path": str(path),
        "n_rows": len(df),
        "correct_column": correct_col,
        "top1_accuracy_pct": round(df[correct_col].mean() * 100, 1),
    }

    return out, audit_row


#loading model outputs
master = None
audit_rows = []

for model_name, path in model_files.items():
    df, audit_row = load_model_predictions(model_name, path)
    audit_rows.append(audit_row)

    if master is None:
        master = df
    else:
        master = master.merge(
            df,
            on="gbifID",
            how="inner",
        )

audit_df = pd.DataFrame(audit_rows)

print("\nMODEL FILE SUMMARY")
print("=" * 100)
print(
    audit_df[
        ["model", "n_rows", "correct_column", "top1_accuracy_pct"]
    ].to_string(index=False)
)

print(f"\nShared evaluable test subset used for comparison: n={len(master)}")

#grouping by taxonomic frequency (4 bucket and 2 bucket versions for order rarity)

four_bucket_mapping = {
    "common": "Common",
    "medium": "Medium",
    "rare": "Rare",
    "very_rare": "Very Rare",
}

four_bucket_order = [
    "Common",
    "Medium",
    "Rare",
    "Very Rare",
]

two_bucket_mapping = {
    "common": "Common",
    "medium": "Underrepresented",
    "rare": "Underrepresented",
    "very_rare": "Underrepresented",
}

two_bucket_order = [
    "Common",
    "Underrepresented",
]


def summarise_by_group(master_df, mapping, group_order, group_col_name):
    df = master_df.copy()

    df[group_col_name] = df["order_rarity_global"].map(mapping)

    # drop unmapped categories (ultra_rare)
    df = df[df[group_col_name].notna()].copy()

    summary_rows = []

    for group in group_order:
        subset = df[df[group_col_name] == group].copy()

        if len(subset) == 0:
            continue

        row = {
            group_col_name: group,
            "n": len(subset),
        }

        for model_name in model_files.keys():
            row[model_name] = subset[model_name].mean()

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    metric_cols = [
        c for c in summary.columns
        if c not in [group_col_name, "n"]
    ]

    summary[metric_cols] = (
        summary[metric_cols] * 100
    ).round(1)

    return summary, df


#4 rarity buckets summary
summary_4bucket, master_4bucket = summarise_by_group(
    master,
    four_bucket_mapping,
    four_bucket_order,
    "taxonomic_frequency_group_4bucket",
)

print("\nFOUR-BUCKET PERFORMANCE BY TAXONOMIC FREQUENCY GROUP")
print("=" * 100)
print(summary_4bucket.to_string(index=False))


#2 rarity buckets summary
summary_2bucket, master_2bucket = summarise_by_group(
    master,
    two_bucket_mapping,
    two_bucket_order,
    "taxonomic_frequency_group_2bucket",
)

print("\nTWO-BUCKET PERFORMANCE BY TAXONOMIC FREQUENCY GROUP")
print("=" * 100)
print(summary_2bucket.to_string(index=False))


#thesis summaries for lower-taxonomy (order-cleaned) condition
thesis_summary_4bucket = summary_4bucket[
    [
        "taxonomic_frequency_group_4bucket",
        "n",
        "BioCLIP",
        "Text_lower",
        "Gated_lower",
    ]
].copy()

thesis_summary_4bucket = thesis_summary_4bucket.rename(columns={
    "Text_lower": "Text_lower_realistic",
    "Gated_lower": "Gated_lower_realistic",
})


thesis_summary_2bucket = summary_2bucket[
    [
        "taxonomic_frequency_group_2bucket",
        "n",
        "BioCLIP",
        "Text_lower",
        "Gated_lower",
    ]
].copy()

thesis_summary_2bucket = thesis_summary_2bucket.rename(columns={
    "Text_lower": "Text_lower_realistic",
    "Gated_lower": "Gated_lower_realistic",
})


print("\nFOUR-BUCKET LOWER TAXONOMY INCLUDED SUMMARY")
print("=" * 100)
print(thesis_summary_4bucket.to_string(index=False))

print("\nTWO-BUCKET LOWER TAXONOMY INCLUDED SUMMARY")
print("=" * 100)
print(thesis_summary_2bucket.to_string(index=False))


#saving

out_dir = results / "rarity_analysis"
out_dir.mkdir(parents=True, exist_ok=True)

summary_4bucket_path = out_dir / "taxonomic_frequency_performance_summary_4bucket_all_models.csv"
summary_2bucket_path = out_dir / "taxonomic_frequency_performance_summary_2bucket_all_models.csv"

thesis_4bucket_path = out_dir / "taxonomic_frequency_lower_realistic_summary_4bucket.csv"
thesis_2bucket_path = out_dir / "taxonomic_frequency_lower_realistic_summary_2bucket.csv"

merged_4bucket_path = out_dir / "taxonomic_frequency_merged_records_4bucket.csv"
merged_2bucket_path = out_dir / "taxonomic_frequency_merged_records_2bucket.csv"

audit_path = out_dir / "taxonomic_frequency_model_file_audit.csv"

summary_4bucket.to_csv(summary_4bucket_path, index=False)
summary_2bucket.to_csv(summary_2bucket_path, index=False)

thesis_summary_4bucket.to_csv(thesis_4bucket_path, index=False)
thesis_summary_2bucket.to_csv(thesis_2bucket_path, index=False)

master_4bucket.to_csv(merged_4bucket_path, index=False)
master_2bucket.to_csv(merged_2bucket_path, index=False)

audit_df.to_csv(audit_path, index=False)

print("\nSaved outputs:")
print(summary_4bucket_path)
print(summary_2bucket_path)
print(thesis_4bucket_path)
print(thesis_2bucket_path)
print(merged_4bucket_path)
print(merged_2bucket_path)
print(audit_path)

#post-analysis sanity check for one result(metadata condition with accuracy 0 for most strict metadata condition)
underrepresented = master_2bucket[
    master_2bucket["taxonomic_frequency_group_2bucket"]
    == "Underrepresented"
].copy()

print("\nUNDERREPRESENTED TRUE ORDERS")
print("=" * 100)

print(
    underrepresented["label"]
    .value_counts()
    .sort_values(ascending=False)
)
underrepresented_wrong = underrepresented[
    underrepresented["Text_strict"] == 0
].copy()

print("\nUNDERREPRESENTED ORDERS MISCLASSIFIED BY TEXT_STRICT")
print("=" * 100)

print(
    underrepresented_wrong["label"]
    .value_counts()
    .sort_values(ascending=False)
)
