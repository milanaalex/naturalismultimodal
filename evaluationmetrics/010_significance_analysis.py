from pathlib import Path
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

#configs

data_path = Path("/Users/milanaalexeeva/Desktop/CMollusca")
results = data_path / "experiments/order_supervised/results"

out_dir = results / "significance_tests"
out_dir.mkdir(parents=True, exist_ok=True)

model_files = {
    "BioCLIP": (
        results / "bioclip2_image_classifier"
        / "bioclip2_image_mlp_test_predictions.csv"
    ),

    "Text_raw": (
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

    "Weighted_raw": (
        results / "multimodal_fusion"
        / "metadata_full_record_realistic"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Weighted_lower": (
        results / "multimodal_fusion"
        / "metadata_lower_taxonomy_realistic"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Weighted_strict": (
        results / "multimodal_fusion"
        / "metadata_strict_conservative"
        / "weighted_probability_fusion"
        / "weighted_probability_fusion_test_predictions.csv"
    ),

    "Gated_raw": (
        results / "multimodal_fusion"
        / "metadata_full_record_realistic"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),

    "Gated_lower": (
        results / "multimodal_fusion"
        / "metadata_lower_taxonomy_realistic"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),

    "Gated_strict": (
        results / "multimodal_fusion"
        / "metadata_strict_conservative"
        / "gated_embedding_fusion"
        / "gated_embedding_fusion_test_predictions.csv"
    ),
}

comparisons = [
    ("Baseline", "BioCLIP", "Majority", "BioCLIP vs Majority"),

    ("RQ1", "Text_raw", "BioCLIP", "Text raw vs BioCLIP"),
    ("RQ1", "Text_lower", "BioCLIP", "Text lower vs BioCLIP"),
    ("RQ1", "Text_strict", "BioCLIP", "Text strict vs BioCLIP"),

    ("RQ2", "Gated_raw", "Text_raw", "Gated raw vs Text raw"),
    ("RQ2", "Gated_lower", "Text_lower", "Gated lower vs Text lower"),
    ("RQ2", "Gated_strict", "Text_strict", "Gated strict vs Text strict"),
    #additional comparison: multimodal vs image-only under taxonomically reduced metadata
    ("RQ2", "Gated_strict", "BioCLIP", "Gated strict vs BioCLIP"),

    ("Fusion", "Gated_raw", "Weighted_raw", "Gated raw vs Weighted raw"),
    ("Fusion", "Gated_lower", "Weighted_lower", "Gated lower vs Weighted lower"),
    ("Fusion", "Gated_strict", "Weighted_strict", "Gated strict vs Weighted strict"),
]

#helpers

def find_correct_column(df):
    candidates = [c for c in df.columns if c.endswith("_correct")]

    if "image_correct" in candidates:
        return "image_correct"

    if len(candidates) != 1:
        raise ValueError(f"Expected one *_correct column, found {candidates}")

    return candidates[0]


def load_majority_baseline():
    bio = pd.read_csv(model_files["BioCLIP"])

    required_cols = {"gbifID", "label"}
    missing = required_cols - set(bio.columns)

    if missing:
        raise ValueError(
            f"BioCLIP file is missing columns required for majority baseline: {missing}"
        )

    majority_label = bio["label"].value_counts().idxmax()

    out = bio[["gbifID", "label"]].copy()
    out["Majority"] = out["label"] == majority_label

    print(f"Majority baseline label on BioCLIP evaluable subset: {majority_label}")

    return out[["gbifID", "Majority"]]


def load_model(model_name):
    if model_name == "Majority":
        return load_majority_baseline()

    path = model_files[model_name]

    if not path.exists():
        raise FileNotFoundError(f"Missing file for {model_name}: {path}")

    df = pd.read_csv(path)
    correct_col = find_correct_column(df)

    out = df[["gbifID", correct_col]].copy()
    out = out.rename(columns={correct_col: model_name})

    return out


def format_p_value(p):
    p = float(p)

    if p < 0.001:
        return "p < 0.001"

    return f"p = {p:.3f}"


def run_mcnemar(challenger, baseline):
    challenger_df = load_model(challenger)
    baseline_df = load_model(baseline)

    paired = challenger_df.merge(
        baseline_df,
        on="gbifID",
        how="inner",
    )

    c = paired[challenger].astype(bool)
    b = paired[baseline].astype(bool)

    both_correct = int((c & b).sum())
    challenger_only = int((c & ~b).sum())
    baseline_only = int((~c & b).sum())
    both_wrong = int((~c & ~b).sum())

    contingency = [
        [both_correct, baseline_only],
        [challenger_only, both_wrong],
    ]

    result = mcnemar(
        contingency,
        exact=False,
        correction=True,
    )

    return {
        "n": len(paired),
        "baseline_accuracy": b.mean() * 100,
        "challenger_accuracy": c.mean() * 100,
        "delta_accuracy": (c.mean() - b.mean()) * 100,
        "both_correct": both_correct,
        "challenger_only_correct": challenger_only,
        "baseline_only_correct": baseline_only,
        "both_wrong": both_wrong,
        "p_value": float(result.pvalue),
    }


#running tests

rows = []

for rq, challenger, baseline, label in comparisons:
    stats = run_mcnemar(
        challenger=challenger,
        baseline=baseline,
    )

    rows.append({
        "RQ": rq,
        "comparison": label,
        "n": stats["n"],
        "baseline_accuracy": round(stats["baseline_accuracy"], 2),
        "challenger_accuracy": round(stats["challenger_accuracy"], 2),
        "delta_accuracy": round(stats["delta_accuracy"], 2),
        "both_correct": stats["both_correct"],
        "challenger_only_correct": stats["challenger_only_correct"],
        "baseline_only_correct": stats["baseline_only_correct"],
        "both_wrong": stats["both_wrong"],
        "p_value_raw": stats["p_value"],
    })

summary = pd.DataFrame(rows)

#applying holm correction

reject, p_holm, _, _ = multipletests(
    summary["p_value_raw"].astype(float).to_numpy(),
    alpha=0.05,
    method="holm",
)

summary["holm_p_raw"] = p_holm
summary["significant"] = reject

summary["p_value"] = summary["p_value_raw"].apply(format_p_value)
summary["holm_p"] = summary["holm_p_raw"].apply(format_p_value)

#saving all outputs

full_out = out_dir / "mcnemar_thesis_summary_with_majority.csv"
compact_out = out_dir / "mcnemar_thesis_compact_with_majority.csv"

summary.to_csv(full_out, index=False)

compact = summary[
    [
        "RQ",
        "comparison",
        "n",
        "baseline_accuracy",
        "challenger_accuracy",
        "delta_accuracy",
        "holm_p",
        "significant",
    ]
].copy()

compact.to_csv(compact_out, index=False)

print("\nMCNEMAR THESIS SUMMARY\n")
print(compact.to_string(index=False))

print("\nSaved:")
print(full_out)
print(compact_out)


#double checking merges

print("\nMERGE AND ALIGNMENT CHECKS")
print("=" * 80)

#row counts and duplicate gbifIDs per model
for model_name in model_files.keys():
    df = pd.read_csv(model_files[model_name])

    n_rows = len(df)
    n_unique = df["gbifID"].nunique()
    n_dupes = n_rows - n_unique

    print(
        f"{model_name:15s} rows={n_rows:5d} "
        f"unique_gbifID={n_unique:5d} "
        f"duplicates={n_dupes:3d}"
    )

#baseline source
majority_df = load_majority_baseline()
print(
    f"{'Majority':15s} rows={len(majority_df):5d} "
    f"unique_gbifID={majority_df['gbifID'].nunique():5d} "
    f"duplicates={len(majority_df) - majority_df['gbifID'].nunique():3d}"
)

#checking merge sizes
merge_rows = []

for rq, challenger, baseline, label in comparisons:
    challenger_df = load_model(challenger)
    baseline_df = load_model(baseline)

    paired = challenger_df.merge(
        baseline_df,
        on="gbifID",
        how="inner",
    )

    challenger_missing = set(baseline_df["gbifID"]) - set(challenger_df["gbifID"])
    baseline_missing = set(challenger_df["gbifID"]) - set(baseline_df["gbifID"])

    merge_rows.append({
        "RQ": rq,
        "comparison": label,
        "challenger": challenger,
        "baseline": baseline,
        "challenger_n": len(challenger_df),
        "baseline_n": len(baseline_df),
        "paired_n": len(paired),
        "challenger_unique": challenger_df["gbifID"].nunique(),
        "baseline_unique": baseline_df["gbifID"].nunique(),
        "paired_unique": paired["gbifID"].nunique(),
        "missing_from_challenger": len(challenger_missing),
        "missing_from_baseline": len(baseline_missing),
    })

merge_check = pd.DataFrame(merge_rows)

print("\nPairwise merge summary:")
print(merge_check.to_string(index=False))

#saving checks
merge_check_out = out_dir / "mcnemar_merge_alignment_check.csv"
merge_check.to_csv(merge_check_out, index=False)

print(f"\nSaved merge check to:\n{merge_check_out}")

#warnings if necessary
problems = merge_check[
    (merge_check["paired_n"] != merge_check["paired_unique"]) |
    (merge_check["challenger_n"] != merge_check["challenger_unique"]) |
    (merge_check["baseline_n"] != merge_check["baseline_unique"])
]

if len(problems) > 0:
    print("\nWARNING: Duplicate gbifIDs detected in one or more comparisons:")
    print(problems.to_string(index=False))
else:
    print("\nNo duplicate gbifID issues detected.")

#listing missing ids
missing_rows = []

for rq, challenger, baseline, label in comparisons:
    challenger_df = load_model(challenger)
    baseline_df = load_model(baseline)

    challenger_ids = set(challenger_df["gbifID"])
    baseline_ids = set(baseline_df["gbifID"])

    for gbif_id in sorted(baseline_ids - challenger_ids):
        missing_rows.append({
            "comparison": label,
            "missing_from": challenger,
            "gbifID": gbif_id,
        })

    for gbif_id in sorted(challenger_ids - baseline_ids):
        missing_rows.append({
            "comparison": label,
            "missing_from": baseline,
            "gbifID": gbif_id,
        })

missing_df = pd.DataFrame(missing_rows)

missing_out = out_dir / "mcnemar_missing_gbifIDs_by_comparison.csv"
missing_df.to_csv(missing_out, index=False)

print(f"Saved missing gbifID details to:\n{missing_out}")

if len(missing_df) == 0:
    print("No missing gbifIDs across pairwise comparisons.")
else:
    print("\nMissing gbifIDs found:")
    print(missing_df.to_string(index=False))

    
