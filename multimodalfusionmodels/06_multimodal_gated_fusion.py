#imports
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

#configs
seed = 42

image_embedding_dir = Path("experiments/order_supervised/results/bioclip2_embeddings")
image_classifier_dir = Path("experiments/order_supervised/results/bioclip2_image_classifier")

text_embedding_root = Path("experiments/order_supervised/results/modernbert_embeddings")
text_classifier_root = Path("experiments/order_supervised/results/modernbert_text_classifier")

out_root = Path("experiments/order_supervised/results/multimodal_fusion")

conditions = [
    "metadata_full_record_realistic",
    "metadata_lower_taxonomy_realistic",
    "metadata_strict_conservative",
]

weighted_image_weights = [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9]

gated_grid_config = {
    "shared_dims": [256, 512],
    "gate_hidden_dims": [128],
    "classifier_hidden_dims": [(256,), (512, 256)],
    "dropouts": [0.1, 0.3],
    "learning_rates": [1e-3, 3e-4],
    "weight_decays": [1e-4],
    "batch_size": 256,
    "max_epochs": 100,
    "patience": 10,
}

#necessary helpers
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def set_seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def entropy_from_probs(probs):
    eps = 1e-12
    return -(probs * np.log(probs + eps)).sum(axis=1)


def build_gated_grid():
    rows = []
    config_id = 0

    for shared_dim in gated_grid_config["shared_dims"]:
        for gate_hidden_dim in gated_grid_config["gate_hidden_dims"]:
            for classifier_hidden_dims in gated_grid_config["classifier_hidden_dims"]:
                for dropout in gated_grid_config["dropouts"]:
                    for learning_rate in gated_grid_config["learning_rates"]:
                        for weight_decay in gated_grid_config["weight_decays"]:
                            config_id += 1
                            rows.append({
                                "config_id": config_id,
                                "shared_dim": shared_dim,
                                "gate_hidden_dim": gate_hidden_dim,
                                "classifier_hidden_dims": classifier_hidden_dims,
                                "dropout": dropout,
                                "learning_rate": learning_rate,
                                "weight_decay": weight_decay,
                                "batch_size": gated_grid_config["batch_size"],
                                "max_epochs": gated_grid_config["max_epochs"],
                                "patience": gated_grid_config["patience"],
                                "seed": seed,
                            })

    return rows

#helper for reporting supervised metrics
def compute_metrics(y_true, y_pred, probs, split_name, n_classes, model_name):
    labels = list(range(n_classes))

    return {
        "model": model_name,
        "split": split_name,
        "n_samples": len(y_true),
        "n_classes": n_classes,

        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),

        "weighted_precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),

        "top3_accuracy": top_k_accuracy_score(y_true, probs, k=3, labels=labels),
        "top5_accuracy": top_k_accuracy_score(y_true, probs, k=5, labels=labels),
    }

#loading additional helpers
def load_embedding_payload(path):
    return torch.load(path, map_location="cpu")


def payload_to_frame(payload):
    return pd.DataFrame({
        "gbifID": payload["gbifID"],
        "split": payload["split"],
        "label": payload["label"],
        "label_id": payload["label_id"],
        "order_count_global": payload["order_count_global"],
        "order_rarity_global": payload["order_rarity_global"],
        "embedding_index": np.arange(len(payload["gbifID"])),
    })


def load_probability_csv(path):
    df = pd.read_csv(path)
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    return df, prob_cols


def make_label_mapping_from_image():
    label_map = pd.read_csv(image_classifier_dir / "bioclip2_image_label_mapping.csv")
    label_map = label_map.sort_values("label_id").reset_index(drop=True)
    label_id_to_name = dict(zip(label_map["label_id"], label_map["label"]))
    return label_map, label_id_to_name

#saving predictions for every record, for later analysis
def make_fusion_prediction_frame(base_df, y_pred, confidence, probs, label_id_to_name, prefix):
    probs_np = np.asarray(probs)
    pred_np = np.asarray(y_pred)

    sorted_ids = np.argsort(probs_np, axis=1)[:, ::-1] #ranking class prediction probabilities highest to lowest

    top1 = sorted_ids[:, 0]
    top2 = sorted_ids[:, 1]
    top3 = sorted_ids[:, 2]
    top5 = sorted_ids[:, :5]

    top1_prob = probs_np[np.arange(len(probs_np)), top1]
    top2_prob = probs_np[np.arange(len(probs_np)), top2]
    top3_prob = probs_np[np.arange(len(probs_np)), top3]

    #storing all predictions and metadata
    pred_df = pd.DataFrame({
        "gbifID": base_df["gbifID"],
        "split": base_df["split"],
        "label": base_df["label"],
        "label_id": base_df["label_id"],
        "order_count_global": base_df["order_count_global"],
        "order_rarity_global": base_df["order_rarity_global"],

        f"{prefix}_pred_id": pred_np,
        f"{prefix}_pred_label": [label_id_to_name[i] for i in pred_np],
        f"{prefix}_confidence": confidence,

        f"{prefix}_top1_id": top1,
        f"{prefix}_top1_label": [label_id_to_name[i] for i in top1],
        f"{prefix}_top1_prob": top1_prob,

        f"{prefix}_top2_id": top2,
        f"{prefix}_top2_label": [label_id_to_name[i] for i in top2],
        f"{prefix}_top2_prob": top2_prob,

        f"{prefix}_top3_id": top3,
        f"{prefix}_top3_label": [label_id_to_name[i] for i in top3],
        f"{prefix}_top3_prob": top3_prob,

        f"{prefix}_top5_labels": [
            "|".join([label_id_to_name[i] for i in row])
            for row in top5
        ],

        f"{prefix}_margin_top1_top2": top1_prob - top2_prob, #margin between top 1 and 2 as simple confidence
        f"{prefix}_entropy": entropy_from_probs(probs_np),
    })

    pred_df[f"{prefix}_correct"] = (  #binary correctness flag for analysis
        pred_df["label_id"].astype(int) == pred_df[f"{prefix}_pred_id"].astype(int)
    )

    pred_df[f"{prefix}_true_in_top3"] = [
        int(true_id) in list(row)
        for true_id, row in zip(pred_df["label_id"], sorted_ids[:, :3])
    ]

    pred_df[f"{prefix}_true_in_top5"] = [
        int(true_id) in list(row)
        for true_id, row in zip(pred_df["label_id"], sorted_ids[:, :5])
    ]

    return pred_df

#saving probability distributions for later analysis and reproducibility
def make_probability_frame(base_df, probs, label_id_to_name):
    prob_df = pd.DataFrame(
        probs,
        columns=[f"prob_{label_id_to_name[i]}" for i in range(len(label_id_to_name))],
    )
    prob_df.insert(0, "gbifID", base_df["gbifID"].to_numpy())
    return prob_df

#anaysing cross modal consistency before fusion (image-only/text-only predictions)
def make_cross_modal_agreement(condition_name, out_dir):
    image_pred = pd.read_csv(
        image_classifier_dir / "bioclip2_image_mlp_test_predictions.csv"
    )

    text_pred = pd.read_csv(
        text_classifier_root
        / condition_name
        / "modernbert_text_mlp_test_predictions.csv"
    )

    text_prefix = f"text_{condition_name}"

    merged = image_pred.merge( #ensuring aligned images and text per gbifID
        text_pred[
            [
                "gbifID",
                f"{text_prefix}_pred_id",
                f"{text_prefix}_pred_label",
                f"{text_prefix}_confidence",
                f"{text_prefix}_entropy",
                f"{text_prefix}_correct",
            ]
        ],
        on="gbifID",
        how="inner",
    )

    merged["image_text_agree"] = (
        merged["image_pred_id"].astype(int)
        == merged[f"{text_prefix}_pred_id"].astype(int)
    )
    
    #checking cross-modal agreement/correctness
    merged["both_correct"] = merged["image_correct"] & merged[f"{text_prefix}_correct"]
    merged["both_wrong"] = ~merged["image_correct"] & ~merged[f"{text_prefix}_correct"]
    merged["image_only_correct"] = merged["image_correct"] & ~merged[f"{text_prefix}_correct"]
    merged["text_only_correct"] = ~merged["image_correct"] & merged[f"{text_prefix}_correct"]

    #cross-modal consistency check
   
    merged["consistency_type"] = "other"
    merged.loc[merged["image_text_agree"] & merged["both_correct"], "consistency_type"] = "consistent_correct"  #consistent_correct = modalities agree and are correct
    merged.loc[merged["image_text_agree"] & merged["both_wrong"], "consistency_type"] = "consistent_incorrect"  #consistent_incorrect = modalities agree but share the same error
    merged.loc[~merged["image_text_agree"] & (merged["image_only_correct"] | merged["text_only_correct"]), "consistency_type"] = "complementary_disagreement" #complementary_disagreement = modalities disagree and exactly one is correct
    merged.loc[~merged["image_text_agree"] & merged["both_wrong"], "consistency_type"] = "joint_failure_disagreement" #joint_failure_disagreement = modalities disagree and both are wrong
   
    #comparing confidence of modality when there is disagreement
    merged["confidence_gap_image_minus_text"] = ( 
        merged["image_confidence"] - merged[f"{text_prefix}_confidence"]
    )
    merged["entropy_gap_image_minus_text"] = (
        merged["image_entropy"] - merged[f"{text_prefix}_entropy"]
    )

    agree_mask = merged["image_text_agree"]
    disagree_mask = ~merged["image_text_agree"]

    type_counts = (
        merged["consistency_type"]
        .value_counts(normalize=True)
        .rename_axis("consistency_type")
        .reset_index(name="proportion")
    )

    summary = pd.DataFrame([{
        "condition": condition_name,
        "n": len(merged),

        "agreement_rate": agree_mask.mean(),
        "disagreement_rate": disagree_mask.mean(),
        "cohen_kappa_image_text": cohen_kappa_score(
            merged["image_pred_id"].astype(int),
            merged[f"{text_prefix}_pred_id"].astype(int),
        ),
        
        #checking for accuracy when agreement/disagreement of modalities
        "accuracy_when_agree": merged.loc[agree_mask, "image_correct"].mean(),
        "image_accuracy_when_disagree": merged.loc[disagree_mask, "image_correct"].mean(),
        "text_accuracy_when_disagree": merged.loc[
            disagree_mask,
            f"{text_prefix}_correct",
        ].mean(),

        "consistent_correct_rate": (merged["consistency_type"] == "consistent_correct").mean(),
        "consistent_incorrect_rate": (merged["consistency_type"] == "consistent_incorrect").mean(),
        "complementary_disagreement_rate": (merged["consistency_type"] == "complementary_disagreement").mean(),
        "joint_failure_disagreement_rate": (merged["consistency_type"] == "joint_failure_disagreement").mean(),

        "both_correct_rate": merged["both_correct"].mean(),
        "image_only_correct_rate": merged["image_only_correct"].mean(),
        "text_only_correct_rate": merged["text_only_correct"].mean(),
        "both_wrong_rate": merged["both_wrong"].mean(),
    }])

    merged.to_csv(out_dir / "cross_modal_consistency_test_records.csv", index=False)
    summary.to_csv(out_dir / "cross_modal_consistency_test_summary.csv", index=False)
    type_counts.to_csv(out_dir / "cross_modal_consistency_type_counts.csv", index=False)

    print("cross-modal consistency summary:")
    print(summary)

#aligning probabilities for weighted fusion
def align_probability_frames(split_name, condition_name):
    image_pred = pd.read_csv(
        image_classifier_dir / f"bioclip2_image_mlp_{split_name}_predictions.csv"
    )
    image_probs, image_prob_cols = load_probability_csv(
        image_classifier_dir / f"bioclip2_image_mlp_{split_name}_probs.csv"
    )

    text_pred = pd.read_csv(
        text_classifier_root
        / condition_name
        / f"modernbert_text_mlp_{split_name}_predictions.csv"
    )
    text_probs, text_prob_cols = load_probability_csv(
        text_classifier_root
        / condition_name
        / f"modernbert_text_mlp_{split_name}_probs.csv"
    )

    common_prob_cols = sorted(set(image_prob_cols).intersection(set(text_prob_cols))) #keeping only class probability columns for fusion
    assert len(common_prob_cols) > 0

    image_meta = image_pred[
        [
            "gbifID",
            "split",
            "label",
            "label_id",
            "order_count_global",
            "order_rarity_global",
            "image_pred_id",
            "image_pred_label",
            "image_confidence",
            "image_entropy",
            "image_margin_top1_top2",
            "image_correct",
        ]
    ].merge(
        image_probs[["gbifID"] + common_prob_cols],
        on="gbifID",
        how="inner",
    )

    text_prefix = f"text_{condition_name}"

    text_keep = [
        "gbifID",
        f"{text_prefix}_pred_id",
        f"{text_prefix}_pred_label",
        f"{text_prefix}_confidence",
        f"{text_prefix}_entropy",
        f"{text_prefix}_margin_top1_top2",
        f"{text_prefix}_correct",
    ]

    text_meta = text_pred[text_keep].merge(
        text_probs[["gbifID"] + common_prob_cols],
        on="gbifID",
        how="inner",
    )

    merged = image_meta.merge(
        text_meta,
        on="gbifID",
        how="inner",
        suffixes=("_imageprob", "_textprob"),
    )
    
    #converting probability to matrix format for fusion
    image_matrix = merged[[f"{col}_imageprob" for col in common_prob_cols]].to_numpy() 
    text_matrix = merged[[f"{col}_textprob" for col in common_prob_cols]].to_numpy()

    label_names = [col.replace("prob_", "") for col in common_prob_cols]

    return merged, image_matrix, text_matrix, label_names

#weighted probability fusion, fusion baseline chosen by validation macro F1
def evaluate_weighted_probability_fusion(condition_name, out_dir, label_id_to_name):
    print_header(f"WEIGHTED PROBABILITY FUSION: {condition_name}")

    val_merged, val_image_probs, val_text_probs, label_names = align_probability_frames(
        "val",
        condition_name,
    )
    test_merged, test_image_probs, test_text_probs, _ = align_probability_frames(
        "test",
        condition_name,
    )
    train_merged, train_image_probs, train_text_probs, _ = align_probability_frames(
        "train",
        condition_name,
    )

    n_classes = len(label_id_to_name)

    id_to_prob_col_index = {} #checking that vectors follow label_id order
    for idx, label_name in enumerate(label_names):
        matching_ids = [
            label_id
            for label_id, name in label_id_to_name.items()
            if name == label_name
        ]
        assert len(matching_ids) == 1
        id_to_prob_col_index[matching_ids[0]] = idx

    sorted_label_ids = sorted(id_to_prob_col_index.keys())
    prob_order_indices = [id_to_prob_col_index[label_id] for label_id in sorted_label_ids]

    def reorder_probs(probs):
        return probs[:, prob_order_indices]

    rows = []
    
    #grid search
    for w_image in weighted_image_weights:
        w_text = 1.0 - w_image

        val_probs = reorder_probs((w_image * val_image_probs) + (w_text * val_text_probs))
        val_pred = val_probs.argmax(axis=1)
        val_true = val_merged["label_id"].astype(int).to_numpy()

        val_metrics = compute_metrics(
            val_true,
            val_pred,
            val_probs,
            "val",
            n_classes,
            f"weighted_probability_fusion_{condition_name}",
        )

        rows.append({
            "condition": condition_name,
            "w_image": w_image,
            "w_text": w_text,
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in ["model", "split"]},
        })

    grid_df = (
        pd.DataFrame(rows)
        .sort_values("val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    best_w_image = float(grid_df.loc[0, "w_image"])
    best_w_text = 1.0 - best_w_image

    grid_df.to_csv(out_dir / "weighted_probability_fusion_grid_summary.csv", index=False)

    split_results = {
        "train": (train_merged, train_image_probs, train_text_probs),
        "val": (val_merged, val_image_probs, val_text_probs),
        "test": (test_merged, test_image_probs, test_text_probs),
    }
    
    #evaluating best fusion weights across all the splits
    metrics_rows = []

    for split_name, (merged, image_probs, text_probs) in split_results.items():
        fused_probs = reorder_probs((best_w_image * image_probs) + (best_w_text * text_probs))
        y_true = merged["label_id"].astype(int).to_numpy()
        y_pred = fused_probs.argmax(axis=1)
        confidence = fused_probs.max(axis=1)

        metrics_rows.append(
            compute_metrics(
                y_true,
                y_pred,
                fused_probs,
                split_name,
                n_classes,
                f"weighted_probability_fusion_{condition_name}",
            )
        )

        pred_df = make_fusion_prediction_frame(
            merged,
            y_pred,
            confidence,
            fused_probs,
            label_id_to_name,
            prefix=f"weighted_{condition_name}",
        )

        pred_df["weighted_w_image"] = best_w_image
        pred_df["weighted_w_text"] = best_w_text

        pred_df.to_csv(
            out_dir / f"weighted_probability_fusion_{split_name}_predictions.csv",
            index=False,
        )

        make_probability_frame(
            merged,
            fused_probs,
            label_id_to_name,
        ).to_csv(
            out_dir / f"weighted_probability_fusion_{split_name}_probs.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "weighted_probability_fusion_metrics_summary.csv", index=False)

    #taking fusion setting selected for reproducibility
    with open(out_dir / "weighted_probability_fusion_best_config.json", "w") as f:
        json.dump(
            {
                "condition": condition_name,
                "selection_metric": "validation_macro_f1",
                "best_w_image": best_w_image,
                "best_w_text": best_w_text,
                "test_usage": "final_evaluation_only",
            },
            f,
            indent=2,
        )

    print("best weighted fusion:")
    print(grid_df.head(1))
    print("test metrics:")
    print(metrics_df[metrics_df["split"] == "test"])

#gated fusion model
class GatedFusionClassifier(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        shared_dim,
        gate_hidden_dim,
        classifier_hidden_dims,
        n_classes,
        dropout,
    ):
        super().__init__()
        
        #embeddings specific to the modality
        self.image_projection = nn.Linear(image_dim, shared_dim)
        self.text_projection = nn.Linear(text_dim, shared_dim)

        self.gate = nn.Sequential(
            nn.Linear(shared_dim * 2, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, 2),
        )

        layers = []
        previous_dim = shared_dim

        for hidden_dim in classifier_hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, n_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, image_x, text_x):
        image_z = self.image_projection(image_x)
        text_z = self.text_projection(text_x)

        gate_logits = self.gate(torch.cat([image_z, text_z], dim=1)) #normalised image/text gate weights
        gate_weights = torch.softmax(gate_logits, dim=1)
        
        #fused multimodal embedding
        image_weight = gate_weights[:, 0].unsqueeze(1)
        text_weight = gate_weights[:, 1].unsqueeze(1)

        fused_z = (image_weight * image_z) + (text_weight * text_z)
        logits = self.classifier(fused_z)

        return logits, gate_weights

#aligning embeddings for gated fusion
def align_embedding_payloads(split_name, condition_name): #aligning embeddings to same specimen
    image_payload = load_embedding_payload(
        image_embedding_dir / f"bioclip2_{split_name}_embeddings.pt"
    )

    text_payload = load_embedding_payload(
        text_embedding_root
        / condition_name
        / f"modernbert_{split_name}_embeddings.pt"
    )

    image_df = payload_to_frame(image_payload).rename(
        columns={"embedding_index": "image_embedding_index"}
    )

    text_df = payload_to_frame(text_payload).rename(
        columns={"embedding_index": "text_embedding_index"}
    )

    merged = image_df.merge( #matching embeddings using id and label
        text_df[
            [
                "gbifID",
                "label_id",
                "text_embedding_index",
            ]
        ],
        on=["gbifID", "label_id"],
        how="inner",
    )

    image_indices = merged["image_embedding_index"].to_numpy()
    text_indices = merged["text_embedding_index"].to_numpy()

    image_x = image_payload["embeddings"][image_indices].float() #aligned tensors necessary for supervised training
    text_x = text_payload["embeddings"][text_indices].float()
    y = torch.tensor(merged["label_id"].astype(int).to_numpy(), dtype=torch.long)

    return {
        "merged": merged,
        "image_x": image_x,
        "text_x": text_x,
        "y": y,
    }


def make_gated_loaders(split_data, batch_size):
    return {
        split_name: DataLoader(
            TensorDataset(data["image_x"], data["text_x"], data["y"]),
            batch_size=batch_size,
            shuffle=(split_name == "train"),
        )
        for split_name, data in split_data.items()
    }

#helper for batched inference
def predict_gated_loader(model, loader, device):
    model.eval()

    logits_all = []
    labels_all = []
    gates_all = []

    with torch.no_grad():
        for image_x, text_x, y in loader:
            logits, gates = model(image_x.to(device), text_x.to(device))
            logits_all.append(logits.cpu())
            gates_all.append(gates.cpu())
            labels_all.append(y.cpu())

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    gates = torch.cat(gates_all, dim=0)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    confidence = probs.max(dim=1).values

    return labels, preds, confidence, probs, gates

#gated fusion training
def train_one_gated_config(config, split_data, image_dim, text_dim, n_classes, device, condition_name):
    set_seed(config["seed"])

    loaders = make_gated_loaders(split_data, config["batch_size"]) #setting up multimodal data loaders

    model = GatedFusionClassifier(
        image_dim=image_dim,
        text_dim=text_dim,
        shared_dim=config["shared_dim"],
        gate_hidden_dim=config["gate_hidden_dim"],
        classifier_hidden_dims=config["classifier_hidden_dims"],
        n_classes=n_classes,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss() #setting standard multiclass classification objective

    optimizer = torch.optim.AdamW( #AdamW for optimization
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_state = None
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    rows = []

    for epoch in range(1, config["max_epochs"] + 1): #stop based on validation
        model.train()
        losses = []

        for image_x, text_x, y in loaders["train"]:
            image_x = image_x.to(device)
            text_x = text_x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(image_x, text_x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_true, train_pred, _, train_probs, _ = predict_gated_loader(
            model,
            loaders["train"],
            device,
        )

        val_true, val_pred, _, val_probs, val_gates = predict_gated_loader(
            model,
            loaders["val"],
            device,
        )

        train_metrics = compute_metrics(
            train_true.numpy(),
            train_pred.numpy(),
            train_probs.numpy(),
            "train",
            n_classes,
            f"gated_embedding_fusion_{condition_name}",
        )

        val_metrics = compute_metrics(
            val_true.numpy(),
            val_pred.numpy(),
            val_probs.numpy(),
            "val",
            n_classes,
            f"gated_embedding_fusion_{condition_name}",
        )
        
        #tracking train, val performance
        row = {
            "condition": condition_name,
            "config_id": config["config_id"],
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "shared_dim": config["shared_dim"],
            "gate_hidden_dim": config["gate_hidden_dim"],
            "classifier_hidden_dims": str(config["classifier_hidden_dims"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],

            "train_accuracy": train_metrics["accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],

            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],

            "val_mean_image_gate": float(val_gates[:, 0].mean().item()),
            "val_mean_text_gate": float(val_gates[:, 1].mean().item()),
        }

        rows.append(row)

        print(
            f"{condition_name} | gated config {config['config_id']} | "
            f"epoch {epoch:03d} | loss {row['train_loss']:.4f} | "
            f"val macro_f1 {row['val_macro_f1']:.4f} | "
            f"val acc {row['val_accuracy']:.4f} | "
            f"gate image/text {row['val_mean_image_gate']:.3f}/{row['val_mean_text_gate']:.3f}"
        )
        
        #keeping checkpoint with best validation macro-F1  
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(
                f"early stopping gated {condition_name} "
                f"config {config['config_id']} at epoch {epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError("no best gated model state saved")

    model.load_state_dict(best_state)

    return {
        "config": config,
        "model": model,
        "training_log": pd.DataFrame(rows),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "loaders": loaders,
    }

#evaluating gated fusion
def evaluate_gated_fusion(condition_name, out_dir, label_id_to_name, device):
    print_header(f"GATED EMBEDDING FUSION: {condition_name}")

    split_data = {
        "train": align_embedding_payloads("train", condition_name),
        "val": align_embedding_payloads("val", condition_name),
        "test": align_embedding_payloads("test", condition_name),
    }

    image_dim = split_data["train"]["image_x"].shape[1]
    text_dim = split_data["train"]["text_x"].shape[1]
    n_classes = len(label_id_to_name)
    
    grid = build_gated_grid() #hyperparameter search 

    with open(out_dir / "gated_embedding_fusion_grid.json", "w") as f:
        json.dump(
            [
                {
                    **cfg,
                    "classifier_hidden_dims": list(cfg["classifier_hidden_dims"]),
                    "condition": condition_name,
                    "selection_metric": "validation_macro_f1",
                    "test_usage": "final_evaluation_only",
                }
                for cfg in grid
            ],
            f,
            indent=2,
        )

    all_logs = []
    summaries = []
    best_run = None
    
    #training and analysing gated candidates
    for config in grid:
        result = train_one_gated_config(
            config=config,
            split_data=split_data,
            image_dim=image_dim,
            text_dim=text_dim,
            n_classes=n_classes,
            device=device,
            condition_name=condition_name,
        )

        all_logs.append(result["training_log"])

        summaries.append({
            "condition": condition_name,
            "config_id": config["config_id"],
            "shared_dim": config["shared_dim"],
            "gate_hidden_dim": config["gate_hidden_dim"],
            "classifier_hidden_dims": str(config["classifier_hidden_dims"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "best_epoch": result["best_epoch"],
            "best_val_macro_f1": result["best_val_macro_f1"],
        })

        if best_run is None or result["best_val_macro_f1"] > best_run["best_val_macro_f1"]:
            best_run = result

    grid_summary_df = (
        pd.DataFrame(summaries)
        .sort_values("best_val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )

    training_log_df = pd.concat(all_logs, axis=0).reset_index(drop=True)

    grid_summary_df.to_csv(out_dir / "gated_embedding_fusion_grid_summary.csv", index=False)
    training_log_df.to_csv(out_dir / "gated_embedding_fusion_training_log.csv", index=False)
    
    #taking best gated model based on val
    best_model = best_run["model"]
    best_config = best_run["config"]

    with open(out_dir / "gated_embedding_fusion_best_config.json", "w") as f:
        json.dump(
            {
                **best_config,
                "classifier_hidden_dims": list(best_config["classifier_hidden_dims"]),
                "condition": condition_name,
                "selection_metric": "validation_macro_f1",
                "best_epoch": best_run["best_epoch"],
                "best_val_macro_f1": best_run["best_val_macro_f1"],
                "test_usage": "final_evaluation_only",
            },
            f,
            indent=2,
        )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "image_dim": image_dim,
            "text_dim": text_dim,
            "n_classes": n_classes,
            "label_map": label_id_to_name,
            "best_config": {
                **best_config,
                "classifier_hidden_dims": list(best_config["classifier_hidden_dims"]),
            },
            "best_epoch": best_run["best_epoch"],
            "best_val_macro_f1": best_run["best_val_macro_f1"],
            "condition": condition_name,
        },
        out_dir / "gated_embedding_fusion_best.pt",
    )

    metrics_rows = []
    gate_summary_rows = []
    gate_by_rarity_rows = []
    
    #final evaluation with gate analysis
    for split_name, data in split_data.items():
        loader = best_run["loaders"][split_name]

        y_true, y_pred, confidence, probs, gates = predict_gated_loader(
            best_model,
            loader,
            device,
        )

        metrics_rows.append(
            compute_metrics(
                y_true.numpy(),
                y_pred.numpy(),
                probs.numpy(),
                split_name,
                n_classes,
                f"gated_embedding_fusion_{condition_name}",
            )
        )

        merged = data["merged"].reset_index(drop=True)

        pred_df = make_fusion_prediction_frame(
            merged,
            y_pred.numpy(),
            confidence.numpy(),
            probs.numpy(),
            label_id_to_name,
            prefix=f"gated_{condition_name}",
        )

        pred_df["gated_image_weight"] = gates[:, 0].numpy()
        pred_df["gated_text_weight"] = gates[:, 1].numpy()

        pred_df.to_csv(
            out_dir / f"gated_embedding_fusion_{split_name}_predictions.csv",
            index=False,
        )

        make_probability_frame(
            merged,
            probs.numpy(),
            label_id_to_name,
        ).to_csv(
            out_dir / f"gated_embedding_fusion_{split_name}_probs.csv",
            index=False,
        )
        
        #summarising the reliance on modalities
        gate_summary_rows.append({
            "condition": condition_name,
            "split": split_name,
            "mean_image_weight": float(gates[:, 0].mean().item()),
            "mean_text_weight": float(gates[:, 1].mean().item()),
            "median_image_weight": float(gates[:, 0].median().item()),
            "median_text_weight": float(gates[:, 1].median().item()),
        })
        
        #checking results across order rareness
        rarity_summary = (
            pred_df
            .groupby("order_rarity_global")[["gated_image_weight", "gated_text_weight"]]
            .mean()
            .reset_index()
        )
        rarity_summary["condition"] = condition_name
        rarity_summary["split"] = split_name
        gate_by_rarity_rows.append(rarity_summary)

    metrics_df = pd.DataFrame(metrics_rows)
    gate_summary_df = pd.DataFrame(gate_summary_rows)
    gate_by_rarity_df = pd.concat(gate_by_rarity_rows, axis=0).reset_index(drop=True)

    metrics_df.to_csv(out_dir / "gated_embedding_fusion_metrics_summary.csv", index=False)
    gate_summary_df.to_csv(out_dir / "gated_embedding_fusion_gate_summary.csv", index=False)
    gate_by_rarity_df.to_csv(out_dir / "gated_embedding_fusion_gate_by_rarity.csv", index=False)

    print("best gated config:")
    print(grid_summary_df.head(1))
    print("test metrics:")
    print(metrics_df[metrics_df["split"] == "test"])
    print("gate summary:")
    print(gate_summary_df)

#running all conditions, with separate outputs
def run_condition(condition_name, label_id_to_name, device):
    print_header(f"MULTIMODAL FUSION CONDITION: {condition_name}")

    condition_root = out_root / condition_name

    agreement_dir = condition_root / "agreement_analysis"
    weighted_dir = condition_root / "weighted_probability_fusion"
    gated_dir = condition_root / "gated_embedding_fusion"

    agreement_dir.mkdir(parents=True, exist_ok=True)
    weighted_dir.mkdir(parents=True, exist_ok=True)
    gated_dir.mkdir(parents=True, exist_ok=True)

    make_cross_modal_agreement(condition_name, agreement_dir)
    evaluate_weighted_probability_fusion(condition_name, weighted_dir, label_id_to_name)
    evaluate_gated_fusion(condition_name, gated_dir, label_id_to_name, device)

#main
def main():
    print_header("MULTIMODAL FUSION EXPERIMENTS")

    set_seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    label_map_df, label_id_to_name = make_label_mapping_from_image()
    label_map_df.to_csv(out_root / "fusion_label_mapping.csv", index=False)

    for condition_name in conditions:
        run_condition(condition_name, label_id_to_name, device)

    print_header("MULTIMODAL FUSION COMPLETE")


if __name__ == "__main__":
    main()
