#imports
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import time
from PIL import Image
from tqdm import tqdm

#helper for headers
def print_header(title):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)

#data paths
data_dir = Path("outputs_eda/datasets/final_model_dataset")
image_dir = Path("experiments/order_supervised/images")
out_dir = Path("experiments/order_supervised/results/image_cache")

image_dir.mkdir(parents=True, exist_ok=True)
out_dir.mkdir(parents=True, exist_ok=True)

#configs
max_workers = 2
request_timeout = 30
retry_sleep_seconds = 10
max_retries = 3

#loading splits
print_header("LOADING DATA")

train_df = pd.read_csv(data_dir / "order_train.csv")
val_df = pd.read_csv(data_dir / "order_val.csv")
test_df = pd.read_csv(data_dir / "order_test.csv")

df_all = pd.concat(
    [train_df, val_df, test_df],
    axis=0,
).reset_index(drop=True)

assert df_all["gbifID"].is_unique
assert df_all["image"].notna().all()

print("total records:", len(df_all))

#helper for images
def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False

#all image helpers
def verify_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def make_result(
    row,
    image_path,
    success,
    status,
    error="",
):
    return {
        "gbifID": row["gbifID"],
        "split": row["split"],
        "label": row["label"],
        "label_id": row["label_id"],
        "image_url": row["image"],
        "image_path": str(image_path),
        "image_success": success,
        "image_status": status,
        "image_error": error,
    }


def download_with_retries(image_url):
    last_status = None
    last_error = ""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                image_url,
                timeout=request_timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            last_status = response.status_code

            if response.status_code == 200:
                return response, "ok", ""

            if response.status_code == 429:
                time.sleep(retry_sleep_seconds * attempt)
                continue

            return None, f"http_status_{response.status_code}", ""

        except Exception as e:
            last_error = str(e)
            time.sleep(retry_sleep_seconds * attempt)

    if last_status is not None:
        return None, f"http_status_{last_status}", last_error

    return None, "failed", last_error


def cache_one_image(row):
    gbif_id = row["gbifID"]
    image_url = row["image"]
    image_path = image_dir / f"{gbif_id}.jpg"

    if image_path.exists() and verify_image(image_path):
        return make_result(
            row=row,
            image_path=image_path,
            success=True,
            status="already_exists",
        )

    if image_path.exists():
        image_path.unlink()

    response, status, error = download_with_retries(image_url)

    if response is None:
        return make_result(
            row=row,
            image_path=image_path,
            success=False,
            status=status,
            error=error,
        )

    try:
        image_path.write_bytes(response.content)

        if not verify_image(image_path):
            if image_path.exists():
                image_path.unlink()

            return make_result(
                row=row,
                image_path=image_path,
                success=False,
                status="invalid_image",
            )

        return make_result(
            row=row,
            image_path=image_path,
            success=True,
            status="downloaded",
        )

    except Exception as e:
        if image_path.exists():
            image_path.unlink()

        return make_result(
            row=row,
            image_path=image_path,
            success=False,
            status="failed",
            error=str(e),
        )

#caching all images
print_header("CACHING IMAGES")

records = df_all.to_dict("records")
rows = []

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [
        executor.submit(cache_one_image, row)
        for row in records
    ]

    for future in tqdm(as_completed(futures), total=len(futures)):
        rows.append(future.result())

#for reproducibility, manifest
manifest_df = pd.DataFrame(rows)

manifest_path = out_dir / "image_cache_manifest.csv"
manifest_df.to_csv(manifest_path, index=False)

#print summary
print_header("IMAGE CACHE COMPLETE")

print("saved:", manifest_path)
print("success rate:", manifest_df["image_success"].mean())

print("\nstatus counts:")
print(manifest_df["image_status"].value_counts())

print("\nfailed examples:")
print(
    manifest_df.loc[
        ~manifest_df["image_success"],
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