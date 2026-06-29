#exploratory analysis, Phi4MM on ultra-rare orders

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import time
import json
import re

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

#configs

model_name = "microsoft/Phi-4-multimodal-instruct"

project_root = Path(__file__).resolve().parents[2]
dataset_dir = project_root / "outputs_eda/datasets/final_model_dataset"

exp_dir = project_root / "experiments/phi4_exploratory"
out_dir = exp_dir / "results"
image_dir = exp_dir / "cached_ultra_rare_images"

out_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)

phi_subset_path = dataset_dir / "phi_core_eval_subset.csv"

metadata_col = "metadata_lower_taxonomy_realistic"
metadata_condition_name = "order_cleaned_lower_taxonomy_realistic"

full_out = out_dir / "phi4_all_ultra_rare_order_cleaned_outputs.csv"
selected_out = out_dir / "phi4_selected_ultra_rare_thesis_case.csv"
run_config_out = out_dir / "phi4_ultra_rare_run_config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("project_root:", project_root, flush=True)
print("phi_subset_path:", phi_subset_path, flush=True)
print("out_dir:", out_dir, flush=True)
print("device:", device, flush=True)

assert phi_subset_path.exists(), f"Missing Phi subset file: {phi_subset_path}"

#loading data

phi_core_eval_df = pd.read_csv(phi_subset_path)

required_cols = {
    "gbifID",
    "label",
    "image",
    "phi_subset_type",
    "order_count_global",
    "order_rarity_global",
    metadata_col,
}

missing_cols = required_cols - set(phi_core_eval_df.columns)
assert not missing_cols, f"Missing required columns: {missing_cols}"

phi_ultra_df = phi_core_eval_df[
    phi_core_eval_df["phi_subset_type"] == "ultra_rare_tail"
].copy()

assert len(phi_ultra_df) > 0, "No ultra-rare records found."

phi_ultra_df = (
    phi_ultra_df
    .sort_values(["order_count_global", "label", "gbifID"])
    .reset_index(drop=True)
)

print("\nUltra-rare records:", len(phi_ultra_df), flush=True)
print(
    phi_ultra_df[
        ["gbifID", "label", "order_count_global", "order_rarity_global", "image"]
    ].to_string(index=False),
    flush=True,
)

#saving configs

run_config = {
    "model_name": model_name,
    "metadata_col": metadata_col,
    "metadata_condition_name": metadata_condition_name,
    "n_ultra_rare_records": int(len(phi_ultra_df)),
    "device": device,
    "dtype": str(dtype),
    "phi_subset_path": str(phi_subset_path),
    "out_dir": str(out_dir),
    "image_dir": str(image_dir),
}

with open(run_config_out, "w") as f:
    json.dump(run_config, f, indent=2)

#caching images

def download_image(url, out_path, max_retries=5, sleep_seconds=10):
    if out_path.exists() and out_path.stat().st_size > 0:
        return "cached"

    headers = {"User-Agent": "Mozilla/5.0 academic-research-script"}
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            req = Request(url, headers=headers)
            with urlopen(req, timeout=60) as response:
                img = Image.open(response).convert("RGB")
                img.save(out_path)
            return "downloaded"

        except HTTPError as e:
            last_error = f"HTTPError {e.code}: {e.reason}"
            print(f"Download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds * attempt if e.code == 429 else sleep_seconds)

        except URLError as e:
            last_error = f"URLError: {e.reason}"
            print(f"Download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds)

        except Exception as e:
            last_error = repr(e)
            print(f"Download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"Could not download image: {last_error}")


def local_image_path(row):
    return image_dir / f"{row['gbifID']}.jpg"


print("\nCaching images...", flush=True)

cached_paths = []
download_statuses = []

for _, row in phi_ultra_df.iterrows():
    path = local_image_path(row)

    try:
        status = download_image(row["image"], path)
    except Exception as e:
        status = f"failed: {e}"

    cached_paths.append(str(path))
    download_statuses.append(status)

    print(row["gbifID"], status, path, flush=True)

phi_ultra_df["local_image_path"] = cached_paths
phi_ultra_df["image_cache_status"] = download_statuses

#loading model

torch.manual_seed(0)

print("\nLoading processor...", flush=True)
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

print("Loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    trust_remote_code=True,
    _attn_implementation="eager",
)

if torch.cuda.is_available():
    model = model.to("cuda")

model.eval()

print("Model loaded.", flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)

#helpers

def build_phi_prompt(row):
    metadata = str(row[metadata_col])

    return f"""
<|im_start|>system<|im_sep|>
You are an expert malacologist specialized in mollusc taxonomy. Use the specimen image and metadata to infer the missing molluscan taxonomic order. Be careful not to confuse taxonomic ranks.
<|im_end|>
<|im_start|>user<|im_sep|>
<|image_1|>

Your task is to infer the missing molluscan ORDER from the specimen image and metadata.

The metadata is order-cleaned: direct order information has been removed, but class, family, genus, species, locality, and specimen context may remain.

Taxonomic hierarchy reminder:
Phylum > Class > Order > Family > Genus > Species

Important rules:
- Predict the ORDER only.
- Do not output a class, subclass, superfamily, family, genus, or species as the order.
- If metadata contains family, genus, or species, use it as evidence to infer the parent order.
- The predicted order must be taxonomically compatible with the class and family evidence.
- If uncertain, still provide the most plausible order and mark confidence as low.

Metadata:
{metadata}

Return exactly this format, with no extra sections:

Visual evidence: <one concise sentence>
Metadata evidence: <one concise sentence>
Rank check: <confirm that the final prediction is an order, not a family/genus/species>
Taxonomic assessment: <one concise sentence linking class/family evidence to the predicted order>
Phylum: Mollusca
Class: <predicted class>
Family evidence: <family/genus/species evidence used, if available>
Order: <one taxonomic order only>
Confidence: high / medium / low
Justification: <two concise sentences explaining why this order fits the image and metadata>
<|im_end|>
<|im_start|>assistant<|im_sep|>
""".strip()


def clean_phi_output(text):
    text = str(text)

    if "<|im_start|>assistant<|im_sep|>" in text:
        text = text.split("<|im_start|>assistant<|im_sep|>")[-1]

    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    return text.strip()


def extract_field(output, field_name):
    if output is None:
        return None

    pattern = rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"

    for line in str(output).splitlines():
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def simple_match(predicted_order, true_order):
    if predicted_order is None or str(predicted_order).strip() == "":
        return "unparsed"

    pred = str(predicted_order).strip().lower()
    true = str(true_order).strip().lower()

    if pred == true:
        return "exact_match"

    if true in pred or pred in true:
        return "partial_string_match"

    return "no_match"


def run_phi_case(row, max_new_tokens=240):
    img_path = Path(row["local_image_path"])

    if not img_path.exists() or img_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing cached image: {img_path}")

    image = Image.open(img_path).convert("RGB")
    prompt = build_phi_prompt(row)

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]

    return prompt, clean_phi_output(output)

#running phi4MM

phi_outputs = []

for i, row in phi_ultra_df.iterrows():
    print("\n" + "=" * 80, flush=True)
    print(f"Running case {i + 1}/{len(phi_ultra_df)}", flush=True)
    print("gbifID:", row["gbifID"], flush=True)
    print("True order:", row["label"], flush=True)
    print("Image:", row["image_cache_status"], flush=True)

    try:
        if str(row["image_cache_status"]).startswith("failed"):
            raise RuntimeError(row["image_cache_status"])

        prompt, output = run_phi_case(row)

        predicted_class = extract_field(output, "Class")
        family_evidence = extract_field(output, "Family evidence")
        predicted_order = extract_field(output, "Order")
        confidence = extract_field(output, "Confidence")
        rank_check = extract_field(output, "Rank check")
        match_type = simple_match(predicted_order, row["label"])

        row_out = {
            "gbifID": row["gbifID"],
            "true_order": row["label"],
            "order_count_global": row["order_count_global"],
            "order_rarity_global": row["order_rarity_global"],
            "metadata_condition": metadata_condition_name,
            "image_url": row["image"],
            "local_image_path": row["local_image_path"],
            "image_cache_status": row["image_cache_status"],
            "prompt": prompt,
            "phi_output": output,
            "phi_predicted_class_parsed": predicted_class,
            "phi_family_evidence_parsed": family_evidence,
            "phi_predicted_order_parsed": predicted_order,
            "phi_confidence_parsed": confidence,
            "phi_rank_check_parsed": rank_check,
            "string_match_to_true_order": match_type,
            "error": None,
        }

        print(output, flush=True)
        print("Parsed order:", predicted_order, flush=True)
        print("Match:", match_type, flush=True)

    except Exception as e:
        row_out = {
            "gbifID": row["gbifID"],
            "true_order": row["label"],
            "order_count_global": row["order_count_global"],
            "order_rarity_global": row["order_rarity_global"],
            "metadata_condition": metadata_condition_name,
            "image_url": row["image"],
            "local_image_path": row["local_image_path"],
            "image_cache_status": row["image_cache_status"],
            "prompt": None,
            "phi_output": None,
            "phi_predicted_class_parsed": None,
            "phi_family_evidence_parsed": None,
            "phi_predicted_order_parsed": None,
            "phi_confidence_parsed": None,
            "phi_rank_check_parsed": None,
            "string_match_to_true_order": None,
            "error": str(e),
        }

        print("ERROR:", e, flush=True)

    phi_outputs.append(row_out)

    #saving (intermediate step)
    pd.DataFrame(phi_outputs).to_csv(full_out, index=False)
    print("Saved partial:", full_out, flush=True)

phi_ultra_results_df = pd.DataFrame(phi_outputs)
phi_ultra_results_df.to_csv(full_out, index=False)

print("\nSaved full results:", full_out, flush=True)

print(
    phi_ultra_results_df[
        [
            "gbifID",
            "true_order",
            "order_count_global",
            "phi_predicted_order_parsed",
            "phi_confidence_parsed",
            "string_match_to_true_order",
            "error",
        ]
    ].to_string(index=False),
    flush=True,
)

#selecting sample case

successful = phi_ultra_results_df[
    phi_ultra_results_df["error"].isna()
].copy()

if len(successful) == 0:
    print("\nNo successful outputs. Selected-case file not created.", flush=True)

else:
    selection_priority = {
        "exact_match": 0,
        "partial_string_match": 1,
        "no_match": 2,
        "unparsed": 3,
    }

    successful["selection_priority"] = (
        successful["string_match_to_true_order"]
        .map(selection_priority)
        .fillna(9)
    )

    selected_phi_case = (
        successful
        .sort_values(["selection_priority", "order_count_global", "true_order", "gbifID"])
        .iloc[0]
    )

    pd.DataFrame([selected_phi_case]).to_csv(selected_out, index=False)

    print("\nSelected thesis case:", flush=True)
    print("gbifID:", selected_phi_case["gbifID"], flush=True)
    print("True order:", selected_phi_case["true_order"], flush=True)
    print("Phi order:", selected_phi_case["phi_predicted_order_parsed"], flush=True)
    print("Confidence:", selected_phi_case["phi_confidence_parsed"], flush=True)
    print("Match:", selected_phi_case["string_match_to_true_order"], flush=True)
    print("\nOutput:", flush=True)
    print(selected_phi_case["phi_output"], flush=True)

    print("\nSaved selected case:", selected_out, flush=True)

print("\nDone.", flush=True)
