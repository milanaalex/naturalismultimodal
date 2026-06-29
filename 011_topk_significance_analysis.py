from pathlib import Path
import pandas as pd

#configs

data_path = Path("/Users/milanaalexeeva/Desktop/CMollusca")

rarity_path = (
    data_path
    / "experiments/order_supervised/results/rarity_analysis"
    / "taxonomic_frequency_merged_records_2bucket.csv"
)

text_strict_path = (
    data_path
    / "experiments/order_supervised/results/modernbert_text_classifier"
    / "metadata_strict_conservative"
    / "modernbert_text_mlp_test_predictions.csv"
)

#loading files

if not rarity_path.exists():
    raise FileNotFoundError(f"Missing rarity file: {rarity_path}")

if not text_strict_path.exists():
    raise FileNotFoundError(f"Missing Text_strict file: {text_strict_path}")

rarity_df = pd.read_csv(rarity_path)
strict_df = pd.read_csv(text_strict_path)

print("\nLoaded files")
print("=" * 80)
print(f"Rarity dataframe: {rarity_df.shape}")
print(f"Text_strict dataframe: {strict_df.shape}")

#columns

top3_col = "text_metadata_strict_conservative_true_in_top3"
top5_col = "text_metadata_strict_conservative_true_in_top5"

required_rarity_cols = {
    "gbifID",
    "label",
    "taxonomic_frequency_group_2bucket",
    "Text_strict",
}

required_strict_cols = {
    "gbifID",
    top3_col,
    top5_col,
}

missing_rarity = required_rarity_cols - set(rarity_df.columns)
missing_strict = required_strict_cols - set(strict_df.columns)

if missing_rarity:
    raise ValueError(f"Missing columns in rarity file: {missing_rarity}")

if missing_strict:
    print("\nAvailable Text_strict columns containing 'top':")
    print([c for c in strict_df.columns if "top" in c.lower()])
    raise ValueError(f"Missing columns in Text_strict file: {missing_strict}")

#long-tail subset

underrepresented = rarity_df[
    rarity_df["taxonomic_frequency_group_2bucket"] == "Underrepresented"
].copy()

merged = underrepresented.merge(
    strict_df[
        [
            "gbifID",
            top3_col,
            top5_col,
        ]
    ],
    on="gbifID",
    how="left",
)

if merged[top3_col].isna().any() or merged[top5_col].isna().any():
    missing_ids = merged.loc[
        merged[top3_col].isna() | merged[top5_col].isna(),
        "gbifID",
    ].tolist()

    raise ValueError(
        f"Some underrepresented records are missing Top-3/Top-5 data: {missing_ids}"
    )

#printing summary

n = len(merged)

top1_correct = int(merged["Text_strict"].sum())
top3_correct = int(merged[top3_col].sum())
top5_correct = int(merged[top5_col].sum())

print("\nTEXT_STRICT LONG-TAIL TOP-K CHECK")
print("=" * 80)

print(f"Long-tail / underrepresented specimens: {n}")

print(
    f"Top-1 correct: {top1_correct}/{n} "
    f"({100 * top1_correct / n:.1f}%)"
)

print(
    f"Top-3 correct: {top3_correct}/{n} "
    f"({100 * top3_correct / n:.1f}%)"
)

print(
    f"Top-5 correct: {top5_correct}/{n} "
    f"({100 * top5_correct / n:.1f}%)"
)

#separation by order-level

order_summary = (
    merged
    .groupby("label")
    .agg(
        n=("gbifID", "count"),
        top1_correct=("Text_strict", "sum"),
        top3_correct=(top3_col, "sum"),
        top5_correct=(top5_col, "sum"),
    )
    .reset_index()
)

order_summary["top1_acc_pct"] = (
    order_summary["top1_correct"] / order_summary["n"] * 100
).round(1)

order_summary["top3_acc_pct"] = (
    order_summary["top3_correct"] / order_summary["n"] * 100
).round(1)

order_summary["top5_acc_pct"] = (
    order_summary["top5_correct"] / order_summary["n"] * 100
).round(1)

order_summary = order_summary.sort_values(
    ["n", "label"],
    ascending=[False, True],
)

print("\nORDER-LEVEL TEXT_STRICT TOP-K CHECK")
print("=" * 80)
print(order_summary.to_string(index=False))

#saving results

out_path = (
    data_path
    / "experiments/order_supervised/results/rarity_analysis"
    / "text_strict_long_tail_topk_check.csv"
)

order_summary.to_csv(out_path, index=False)

print(f"\nSaved order-level check to:\n{out_path}")
