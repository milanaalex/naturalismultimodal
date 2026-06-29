#imports
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

#configs
seed = 42

embedding_dir = Path("experiments/order_supervised/results/bioclip2_embeddings")
out_dir = Path("experiments/order_supervised/results/bioclip2_image_classifier")

model_name = "bioclip2_image_mlp"

grid_config = {
    "hidden_configs": [(256,), (512, 256)],
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


def build_grid():
    rows = []
    config_id = 0

    for hidden_dims in grid_config["hidden_configs"]:
        for dropout in grid_config["dropouts"]:
            for learning_rate in grid_config["learning_rates"]:
                for weight_decay in grid_config["weight_decays"]:
                    config_id += 1
                    rows.append({
                        "config_id": config_id,
                        "hidden_dims": hidden_dims,
                        "dropout": dropout,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "batch_size": grid_config["batch_size"],
                        "max_epochs": grid_config["max_epochs"],
                        "patience": grid_config["patience"],
                        "seed": seed,
                    })

    return rows

#defining model
class MLPHead(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout):
        super().__init__()

        layers = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

#loading data
def load_embedding_payload(path):
    payload = torch.load(path, map_location="cpu")

    required = [
        "gbifID",
        "split",
        "label",
        "label_id",
        "order_count_global",
        "order_rarity_global",
        "embeddings",
        "model_name",
    ]

    for key in required:
        assert key in payload, f"missing key: {key}"

    return payload


def make_label_mapping(train_data):
    label_map_df = (
        pd.DataFrame({
            "label_id": train_data["label_id"],
            "label": train_data["label"],
        })
        .drop_duplicates()
        .sort_values("label_id")
        .reset_index(drop=True)
    )

    assert label_map_df["label_id"].is_unique
    assert label_map_df["label"].is_unique
    assert label_map_df["label_id"].tolist() == list(range(len(label_map_df)))

    label_id_to_name = dict(zip(label_map_df["label_id"], label_map_df["label"]))
    return label_map_df, label_id_to_name


def prepare_tensors(train_data, val_data, test_data, n_classes):
    x_train = train_data["embeddings"].float()
    x_val = val_data["embeddings"].float()
    x_test = test_data["embeddings"].float()

    y_train = torch.tensor(train_data["label_id"], dtype=torch.long)
    y_val = torch.tensor(val_data["label_id"], dtype=torch.long)
    y_test = torch.tensor(test_data["label_id"], dtype=torch.long)

    assert x_train.shape[0] == len(y_train)
    assert x_val.shape[0] == len(y_val)
    assert x_test.shape[0] == len(y_test)

    assert y_train.min() >= 0 and y_train.max() < n_classes
    assert y_val.min() >= 0 and y_val.max() < n_classes
    assert y_test.min() >= 0 and y_test.max() < n_classes

    return x_train, y_train, x_val, y_val, x_test, y_test

#model training and evaluation
def make_loaders(x_train, y_train, x_val, y_val, x_test, y_test, batch_size):
    return {
        "train": DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True),
        "val": DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False),
        "test": DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False),
    }

#for gated fusion and evaluation
def predict_loader(model, loader, device):
    model.eval()

    logits_all = []
    labels_all = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch.to(device))
            logits_all.append(logits.cpu())
            labels_all.append(y_batch.cpu())

    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)

    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    confidence = probs.max(dim=1).values

    return labels, preds, confidence, probs


def compute_metrics(y_true, y_pred, probs, split_name, n_classes):
    labels = list(range(n_classes))

    return {
        "model": model_name,
        "split": split_name,
        "n_samples": len(y_true),
        "n_classes": n_classes,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "top3_accuracy": top_k_accuracy_score(y_true, probs, k=3, labels=labels),
        "top5_accuracy": top_k_accuracy_score(y_true, probs, k=5, labels=labels),
    }

#selecting the best checkpoint using the macro-F1 validation
def train_one_config(config, tensors, input_dim, n_classes, device):
    set_seed(config["seed"])

    x_train, y_train, x_val, y_val, x_test, y_test = tensors

    loaders = make_loaders(
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        config["batch_size"],
    )

    model = MLPHead(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        n_classes=n_classes,
        dropout=config["dropout"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_state = None
    best_val_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    rows = []

    #train, val loop
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []

        for x_batch, y_batch in loaders["train"]:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_true, train_pred, _, train_probs = predict_loader(model, loaders["train"], device)
        val_true, val_pred, _, val_probs = predict_loader(model, loaders["val"], device)

        train_metrics = compute_metrics(
            train_true.numpy(),
            train_pred.numpy(),
            train_probs.numpy(),
            "train",
            n_classes,
        )

        val_metrics = compute_metrics(
            val_true.numpy(),
            val_pred.numpy(),
            val_probs.numpy(),
            "val",
            n_classes,
        )

        row = {
            "config_id": config["config_id"],
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "hidden_dims": str(config["hidden_dims"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_top3_accuracy": train_metrics["top3_accuracy"],
            "train_top5_accuracy": train_metrics["top5_accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_top3_accuracy": val_metrics["top3_accuracy"],
            "val_top5_accuracy": val_metrics["top5_accuracy"],
        }

        rows.append(row)

        print(
            f"config {config['config_id']} | epoch {epoch:03d} | "
            f"loss {row['train_loss']:.4f} | "
            f"val macro_f1 {row['val_macro_f1']:.4f} | "
            f"val acc {row['val_accuracy']:.4f}"
        )
        
        #taking best checkpoint for the evaluation
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config["patience"]:
            print(f"early stopping config {config['config_id']} at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("no best model state saved")

    model.load_state_dict(best_state)

    return {
        "config": config,
        "model": model,
        "training_log": pd.DataFrame(rows),
        "best_val_macro_f1": best_val_macro_f1,
        "best_epoch": best_epoch,
        "loaders": loaders,
    }

#helpers for outputs
def make_prediction_frame(split_data, y_pred, confidence, probs, label_id_to_name):
    probs_np = probs.numpy()
    pred_np = y_pred.numpy()
    sorted_ids = np.argsort(probs_np, axis=1)[:, ::-1]

    top1 = sorted_ids[:, 0]
    top2 = sorted_ids[:, 1]
    top3 = sorted_ids[:, 2]
    top5 = sorted_ids[:, :5]

    top1_prob = probs_np[np.arange(len(probs_np)), top1]
    top2_prob = probs_np[np.arange(len(probs_np)), top2]
    top3_prob = probs_np[np.arange(len(probs_np)), top3]
    
    #saving data for later analysis
    pred_df = pd.DataFrame({
        "gbifID": split_data["gbifID"],
        "split": split_data["split"],
        "label": split_data["label"],
        "label_id": split_data["label_id"],
        "order_count_global": split_data["order_count_global"],
        "order_rarity_global": split_data["order_rarity_global"],
        "image_pred_id": pred_np,
        "image_pred_label": [label_id_to_name[i] for i in pred_np],
        "image_confidence": confidence.numpy(),
        "image_top1_id": top1,
        "image_top1_label": [label_id_to_name[i] for i in top1],
        "image_top1_prob": top1_prob,
        "image_top2_id": top2,
        "image_top2_label": [label_id_to_name[i] for i in top2],
        "image_top2_prob": top2_prob,
        "image_top3_id": top3,
        "image_top3_label": [label_id_to_name[i] for i in top3],
        "image_top3_prob": top3_prob,
        "image_top5_labels": ["|".join([label_id_to_name[i] for i in row]) for row in top5],
        "image_margin_top1_top2": top1_prob - top2_prob,
        "image_entropy": entropy_from_probs(probs_np),
    })

    pred_df["image_correct"] = pred_df["label_id"].astype(int) == pred_df["image_pred_id"].astype(int)
    pred_df["image_true_in_top3"] = [int(y) in list(row) for y, row in zip(pred_df["label_id"], sorted_ids[:, :3])]
    pred_df["image_true_in_top5"] = [int(y) in list(row) for y, row in zip(pred_df["label_id"], sorted_ids[:, :5])]

    return pred_df

#probability distributions for the later fusion
def make_probability_frame(split_data, probs, label_id_to_name):
    prob_df = pd.DataFrame(
        probs.numpy(),
        columns=[f"prob_{label_id_to_name[i]}" for i in range(len(label_id_to_name))],
    )
    prob_df.insert(0, "gbifID", split_data["gbifID"])
    return prob_df

#main 
#grid search, selecting model and evaluating
def main():
    print_header("BIOCLIP2 IMAGE-ONLY MLP GRID SEARCH")

    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = load_embedding_payload(embedding_dir / "bioclip2_train_embeddings.pt")
    val_data = load_embedding_payload(embedding_dir / "bioclip2_val_embeddings.pt")
    test_data = load_embedding_payload(embedding_dir / "bioclip2_test_embeddings.pt")

    label_map_df, label_id_to_name = make_label_mapping(train_data)
    n_classes = len(label_map_df)

    tensors = prepare_tensors(train_data, val_data, test_data, n_classes)
    input_dim = tensors[0].shape[1]

    print("device:", device)
    print("train:", tensors[0].shape)
    print("val:", tensors[2].shape)
    print("test:", tensors[4].shape)
    print("classes:", n_classes)

    label_map_df.to_csv(out_dir / "bioclip2_image_label_mapping.csv", index=False)
    
    #ensuring reproducibility with set search space
    grid = build_grid()

    with open(out_dir / "bioclip2_image_mlp_grid.json", "w") as f:
        json.dump(
            [{**cfg, "hidden_dims": list(cfg["hidden_dims"])} for cfg in grid],
            f,
            indent=2,
        )

    all_logs = []
    summaries = []
    best_run = None

    for config in grid:
        result = train_one_config(
            config=config,
            tensors=tensors,
            input_dim=input_dim,
            n_classes=n_classes,
            device=device,
        )

        all_logs.append(result["training_log"])

        summaries.append({
            "config_id": config["config_id"],
            "hidden_dims": str(config["hidden_dims"]),
            "dropout": config["dropout"],
            "learning_rate": config["learning_rate"],
            "weight_decay": config["weight_decay"],
            "best_epoch": result["best_epoch"],
            "best_val_macro_f1": result["best_val_macro_f1"],
        })
        
        #taking the best run
        if best_run is None or result["best_val_macro_f1"] > best_run["best_val_macro_f1"]:
            best_run = result

    grid_summary_df = (
        pd.DataFrame(summaries)
        .sort_values("best_val_macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    training_log_df = pd.concat(all_logs, axis=0).reset_index(drop=True)

    grid_summary_df.to_csv(out_dir / "bioclip2_image_mlp_grid_summary.csv", index=False)
    training_log_df.to_csv(out_dir / "bioclip2_image_mlp_grid_training_log.csv", index=False)

    best_model = best_run["model"]
    best_config = best_run["config"]

    with open(out_dir / "bioclip2_image_mlp_best_config.json", "w") as f:
        json.dump(
            {
                **best_config,
                "hidden_dims": list(best_config["hidden_dims"]),
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
            "input_dim": input_dim,
            "n_classes": n_classes,
            "label_map": label_id_to_name,
            "best_config": {
                **best_config,
                "hidden_dims": list(best_config["hidden_dims"]),
            },
            "best_epoch": best_run["best_epoch"],
            "best_val_macro_f1": best_run["best_val_macro_f1"],
            "bioclip_model_name": train_data["model_name"],
        },
        out_dir / "bioclip2_image_mlp_best.pt",
    )

    metrics_rows = []
    split_payloads = {"train": train_data, "val": val_data, "test": test_data}
    
    #evaluating model
    for split_name, split_data in split_payloads.items():
        loader = best_run["loaders"][split_name]
        y_true, y_pred, confidence, probs = predict_loader(best_model, loader, device)

        metrics_rows.append(
            compute_metrics(
                y_true.numpy(),
                y_pred.numpy(),
                probs.numpy(),
                split_name,
                n_classes,
            )
        )

        make_prediction_frame(split_data, y_pred, confidence, probs, label_id_to_name).to_csv(
            out_dir / f"bioclip2_image_mlp_{split_name}_predictions.csv",
            index=False,
        )

        make_probability_frame(split_data, probs, label_id_to_name).to_csv(
            out_dir / f"bioclip2_image_mlp_{split_name}_probs.csv",
            index=False,
        )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "bioclip2_image_mlp_metrics_summary.csv", index=False)

    print_header("BIOCLIP2 IMAGE-ONLY MLP COMPLETE")
    print("\nbest config:")
    print(grid_summary_df.head(1))
    print("\ntest metrics:")
    print(metrics_df[metrics_df["split"] == "test"])
    print("\nsaved outputs to:", out_dir)


if __name__ == "__main__":
    main()  
