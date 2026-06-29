#imports
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

#header helper
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

#setting paths
data_dir = Path("outputs_eda/datasets/final_model_dataset")

train_csv = data_dir / "order_train.csv"
val_csv = data_dir / "order_val.csv"
test_csv = data_dir / "order_test.csv"

out_dir = Path("experiments/order_supervised/results/majority_baseline")
out_dir.mkdir(parents=True, exist_ok=True)

#loading data
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

print("train:", train_df.shape)
print("val:", val_df.shape)
print("test:", test_df.shape)

#running checks
required_cols = [
    "gbifID",
    "split",
    "label",
    "label_id",
    "test_label_id",
    "is_test_evaluable_order",
    "is_supervised_order",
    "order_count_global",
    "order_rarity_global",
]

for split_name, df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    missing_cols = [col for col in required_cols if col not in df.columns]
    assert not missing_cols, f"{split_name} missing columns: {missing_cols}"

    assert df["gbifID"].is_unique, f"{split_name} has duplicate gbifID values"
    assert (df["split"] == split_name).all(), f"{split_name} has wrong split labels"
    assert df["label"].notna().all(), f"{split_name} has missing labels"
    assert df["label_id"].notna().all(), f"{split_name} has missing label_id"
    assert df["test_label_id"].notna().all(), f"{split_name} has missing test_label_id"

    assert df["is_supervised_order"].all(), f"{split_name} contains non-supervised rows"
    assert df["is_test_evaluable_order"].all(), f"{split_name} contains non-evaluable rows"

#checking data splits 
print_header("CHECKING SPLIT SAFETY")

train_ids = set(train_df["gbifID"])
val_ids = set(val_df["gbifID"])
test_ids = set(test_df["gbifID"])

assert train_ids.isdisjoint(val_ids), "train/val overlap detected"
assert train_ids.isdisjoint(test_ids), "train/test overlap detected"
assert val_ids.isdisjoint(test_ids), "val/test overlap detected"

train_labels = set(train_df["label"])
val_labels = set(val_df["label"])
test_labels = set(test_df["label"])

assert train_labels == val_labels == test_labels, (
    "train, val, and test label spaces do not match"
)

train_label_ids = set(train_df["label_id"].astype(int))
val_label_ids = set(val_df["label_id"].astype(int))
test_label_ids = set(test_df["label_id"].astype(int))

assert train_label_ids == val_label_ids == test_label_ids, (
    "train, val, and test label_id spaces do not match"
)

print("split safety checks passed")
print("shared supervised orders:", len(train_labels))

#for label mapping
label_map_df = (
    train_df[["label_id", "label"]]
    .drop_duplicates()
    .sort_values("label_id")
    .reset_index(drop=True)
)

assert label_map_df["label_id"].is_unique, (
    "label_id values are not unique"
)

assert label_map_df["label"].is_unique, (
    "label names are not unique"
)

n_classes = len(label_map_df)

print("number of supervised classes:", n_classes)


#majority baseline on train
print_header("FITTING MAJORITY BASELINE")

train_counts = (
    train_df["label_id"]
    .astype(int)
    .value_counts()
    .sort_values(ascending=False)
)

majority_label_id = int(train_counts.index[0])

majority_label = train_df.loc[
    train_df["label_id"].astype(int) == majority_label_id,
    "label",
].iloc[0]

majority_train_count = int(train_counts.iloc[0])
majority_train_fraction = majority_train_count / len(train_df)

print("majority label:", majority_label)
print("majority label id:", majority_label_id)
print("majority train count:", majority_train_count)
print("majority train fraction:", majority_train_fraction)


#helper functions for baseline evaluation

#predicting most frequent (majority) for every record
def add_majority_predictions(df):
    pred_df = df.copy()

    pred_df["pred_id"] = majority_label_id
    pred_df["pred_label"] = majority_label
    pred_df["confidence"] = 1.0

    pred_df["correct"] = (
        pred_df["label_id"].astype(int) == majority_label_id
    )

    return pred_df

#main majority baseline result in test set 
def evaluate_split(df, split_name):
    pred_df = add_majority_predictions(df)

    y_true = pred_df["label_id"].astype(int)
    y_pred = pred_df["pred_id"].astype(int)

    metrics = {
        "model": "majority_baseline",
        "split": split_name,
        "n_samples": len(pred_df),
        "n_classes": n_classes,
        "majority_label": majority_label,
        "majority_label_id": majority_label_id,
        "majority_train_count": majority_train_count,
        "majority_train_fraction": majority_train_fraction,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    return metrics, pred_df

#running evaluation
print_header("RUNNING EVALUATION")

metrics_rows = {}
prediction_dfs = {}

for split_name, df in [
    ("train", train_df),
    ("val", val_df),
    ("test", test_df),
]:
    metrics, pred_df = evaluate_split(df, split_name)

    metrics_rows[split_name] = metrics
    prediction_dfs[split_name] = pred_df

    print(f"\n{split_name.upper()} RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

#saving results
print_header("SAVING OUTPUTS")

metrics_df = pd.DataFrame(metrics_rows.values())

test_pred_df = prediction_dfs["test"][[
    "gbifID",
    "split",
    "label",
    "label_id",
    "test_label_id",
    "order_count_global",
    "order_rarity_global",
    "pred_id",
    "pred_label",
    "confidence",
    "correct",
]].copy()

metrics_path = out_dir / "majority_metrics_summary.csv"
test_predictions_path = out_dir / "majority_test_predictions.csv"

metrics_df.to_csv(metrics_path, index=False)
test_pred_df.to_csv(test_predictions_path, index=False)

print("saved:")
print("-", metrics_path)
print("-", test_predictions_path)

#preview
print_header("MAJORITY BASELINE RESULT (FOR TEST SET)")

test_metrics = metrics_df.loc[
    metrics_df["split"] == "test"
].iloc[0]

for key in [
    "model",
    "split",
    "n_samples",
    "n_classes",
    "majority_label",
    "majority_train_fraction",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
]:
    print(f"{key}: {test_metrics[key]}")


print_header("SUMMARY FOR TRAIN, VAL, TEST")

print(metrics_df)


print_header("TEST MAJORITY PREDICTION PREVIEW")

print(test_pred_df.head())
