#imports
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

#helper for headings
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

#configs
data_dir = Path("outputs_eda/datasets/final_model_dataset")

train_csv = data_dir / "order_train.csv"
val_csv = data_dir / "order_val.csv"
test_csv = data_dir / "order_test.csv"

image_dir = Path("experiments/order_supervised/images")
out_dir = Path("experiments/order_supervised/results/bioclip2_embeddings")

image_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

model_name = "hf-hub:imageomics/bioclip-2"
batch_size = 32
num_workers = 0
device = "cuda" if torch.cuda.is_available() else "cpu"

#loading data
print_header("LOADING TRAIN, VAL, TEST")

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

split_dfs = {
    "train": train_df,
    "val": val_df,
    "test": test_df,
}

for split_name, df in split_dfs.items():
    assert "gbifID" in df.columns
    assert "split" in df.columns
    assert "label" in df.columns
    assert "label_id" in df.columns
    assert "image" in df.columns
    assert "order_count_global" in df.columns
    assert "order_rarity_global" in df.columns
    assert df["gbifID"].is_unique
    assert (df["split"] == split_name).all()
    assert df["label_id"].notna().all()

    print(split_name, df.shape)

print("device:", device)

#helper for images
def get_cached_image_path(gbif_id):
    return image_dir / f"{gbif_id}.jpg"


def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_image_tensor(image_path, preprocess):
    image = Image.open(image_path).convert("RGB")
    return preprocess(image)

#loading bioclip2
print_header("LOADING BIOCLIP2")

model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
model = model.to(device)
model.eval()

print("model:", model_name)


#bioclip2 embeddings extraction
def extract_split_embeddings(df, split_name):
    print_header(f"EXTRACTING BIOCLIP2 IMAGE EMBEDDINGS: {split_name.upper()}")

    manifest_rows = []
    image_tensors = []
    kept_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        gbif_id = row["gbifID"]
        image_path = get_cached_image_path(gbif_id)

        if not image_path.exists():
            manifest_rows.append({
                "gbifID": gbif_id,
                "split": split_name,
                "label": row["label"],
                "label_id": row["label_id"],
                "image_url": row["image"],
                "image_path": str(image_path),
                "image_success": False,
                "image_status": "missing_cached_image",
                "image_error": "",
            })
            continue

        if not verify_image(image_path):
            manifest_rows.append({
                "gbifID": gbif_id,
                "split": split_name,
                "label": row["label"],
                "label_id": row["label_id"],
                "image_url": row["image"],
                "image_path": str(image_path),
                "image_success": False,
                "image_status": "invalid_cached_image",
                "image_error": "",
            })
            continue

        try:
            image_tensor = load_image_tensor(image_path, preprocess_val)

            image_tensors.append(image_tensor)
            kept_rows.append(row)

            manifest_rows.append({
                "gbifID": gbif_id,
                "split": split_name,
                "label": row["label"],
                "label_id": row["label_id"],
                "image_url": row["image"],
                "image_path": str(image_path),
                "image_success": True,
                "image_status": "cached",
                "image_error": "",
            })

        except Exception as e:
            manifest_rows.append({
                "gbifID": gbif_id,
                "split": split_name,
                "label": row["label"],
                "label_id": row["label_id"],
                "image_url": row["image"],
                "image_path": str(image_path),
                "image_success": False,
                "image_status": "preprocess_failed",
                "image_error": str(e),
            })

    if len(image_tensors) == 0:
        raise ValueError(f"no valid cached images found for {split_name}")

    embeddings = []

    with torch.no_grad():
        for start in tqdm(range(0, len(image_tensors), batch_size)):
            batch = torch.stack(
                image_tensors[start:start + batch_size]
            ).to(device)

            image_features = model.encode_image(batch)
            image_features = torch.nn.functional.normalize(
                image_features,
                dim=-1,
            )

            embeddings.append(image_features.cpu())

    embeddings = torch.cat(embeddings, dim=0)

    kept_df = pd.DataFrame(kept_rows).reset_index(drop=True)

    assert len(kept_df) == embeddings.shape[0]

    output = {
        "gbifID": kept_df["gbifID"].tolist(),
        "split": kept_df["split"].tolist(),
        "label": kept_df["label"].tolist(),
        "label_id": kept_df["label_id"].astype(int).tolist(),
        "order_count_global": kept_df["order_count_global"].tolist(),
        "order_rarity_global": kept_df["order_rarity_global"].tolist(),
        "embeddings": embeddings,
        "model_name": model_name,
        "embedding_dim": embeddings.shape[1],
    }

    embedding_path = out_dir / f"bioclip2_{split_name}_embeddings.pt"
    manifest_path = out_dir / f"bioclip2_{split_name}_image_manifest.csv"

    torch.save(output, embedding_path)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)

    print("saved embeddings:", embedding_path)
    print("saved manifest:", manifest_path)
    print("valid images:", embeddings.shape[0], "/", len(df))
    print("embedding shape:", embeddings.shape)

    return manifest_df

#embeddings for all splits
print_header("RUNNING BIOCLIP2 IMAGE EMBEDDING EXTRACTION")

all_manifests = []

for split_name, df in split_dfs.items():
    manifest_df = extract_split_embeddings(df, split_name)
    all_manifests.append(manifest_df)

all_manifest_df = pd.concat(all_manifests, axis=0).reset_index(drop=True)

all_manifest_path = out_dir / "bioclip2_image_manifest_all.csv"
all_manifest_df.to_csv(all_manifest_path, index=False)


#saving and summary
print_header("BIOCLIP2 IMAGE EMBEDDING EXTRACTION COMPLETE")

print("saved combined manifest:", all_manifest_path)

print("\nimage success rate:")
print(all_manifest_df["image_success"].mean())

print("\nimage status counts:")
print(all_manifest_df["image_status"].value_counts())

print("\nfailed examples:")
print(
    all_manifest_df.loc[
        ~all_manifest_df["image_success"],
        [
            "gbifID",
            "split",
            "label",
            "image_url",
            "image_status",
            "image_error",
        ],
    ].head(20)
)