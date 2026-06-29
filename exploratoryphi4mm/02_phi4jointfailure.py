#exploratory, phi4MM shared failure case analysis

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import json
import re
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

#configs

model_name = "microsoft/Phi-4-multimodal-instruct"

project_root = Path(__file__).resolve().parents[2]
dataset_dir = project_root / "outputs_eda/datasets/final_model_dataset"

exp_dir = project_root / "experiments/phi4_exploratory"
out_dir = exp_dir / "results_joint_failure"
image_dir = exp_dir / "cached_joint_failure_images"

out_dir.mkdir(parents=True, exist_ok=True)
image_dir.mkdir(parents=True, exist_ok=True)

phi_subset_path = dataset_dir / "phi_core_eval_subset.csv"

metadata_col = "metadata_lower_taxonomy_realistic"
metadata_condition_name = "order_cleaned_lower_taxonomy_realistic"
subset_type = "joint_failure"
expected_n = 10

full_out = out_dir / "phi4_joint_failure_open_set_outputs.csv"
selected_out = out_dir / "phi4_selected_joint_failure_case.csv"
config_out = out_dir / "phi4_joint_failure_config.json"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("project_root:", project_root, flush=True)
print("phi_subset_path:", phi_subset_path, flush=True)
print("out_dir:", out_dir, flush=True)
print("device:", device, flush=True)

assert phi_subset_path.exists(), f"missing file: {phi_subset_path}"

#loading data

df = pd.read_csv(phi_subset_path)

required_cols = {
    "gbifID",
    "label",
    "image",
    "phi_subset_type",
    "order_count_global",
    "order_rarity_global",
    metadata_col,
}

missing = required_cols - set(df.columns)
assert not missing, f"missing columns: {missing}"

phi_df = df[df["phi_subset_type"] == subset_type].copy()
assert len(phi_df) > 0, "no joint-failure records found."
assert len(phi_df) == expected_n, f"expected {expected_n} joint-failure records, found {len(phi_df)}."

phi_df = (
    phi_df
    .sort_values(["order_count_global", "label", "gbifID"])
    .reset_index(drop=True)
)

print("\njoint-failure records:", len(phi_df), flush=True)
print(
    phi_df[
        ["gbifID", "label", "order_count_global", "order_rarity_global", "image"]
    ].to_string(index=False),
    flush=True,
)

with open(config_out, "w") as f:
    json.dump(
        {
            "model_name": model_name,
            "metadata_col": metadata_col,
            "metadata_condition": metadata_condition_name,
            "subset_type": subset_type,
            "n_records": int(len(phi_df)),
            "expected_n": expected_n,
            "device": device,
            "dtype": str(dtype),
            "task_type": "open_set_zero_shot_order_prediction",
            "out_dir": str(out_dir),
            "image_dir": str(image_dir),
        },
        f,
        indent=2,
    )

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
            last_error = f"httperror {e.code}: {e.reason}"
            print(f"download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds * attempt if e.code == 429 else sleep_seconds)

        except URLError as e:
            last_error = f"urlerror: {e.reason}"
            print(f"download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds)

        except Exception as e:
            last_error = repr(e)
            print(f"download failed ({attempt}/{max_retries}): {last_error}", flush=True)
            time.sleep(sleep_seconds)

    raise RuntimeError(f"could not download image: {last_error}")


def image_path(row):
    return image_dir / f"{row['gbifID']}.jpg"


paths = []
statuses = []

print("\ncaching images...", flush=True)

for _, row in phi_df.iterrows():
    path = image_path(row)

    try:
        status = download_image(row["image"], path)
    except Exception as e:
        status = f"failed: {e}"

    paths.append(str(path))
    statuses.append(status)

    print(row["gbifID"], status, path, flush=True)

phi_df["local_image_path"] = paths
phi_df["image_cache_status"] = statuses

#loading model

torch.manual_seed(0)

print("\nloading processor...", flush=True)
processor = AutoProcessor.from_pretrained(
    model_name,
    trust_remote_code=True,
)

print("loading model...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    trust_remote_code=True,
    _attn_implementation="eager",
)

if torch.cuda.is_available():
    model = model.to("cuda")

model.eval()

print("model loaded.", flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)

#helpers and prompt

def build_prompt(row):
    metadata = str(row[metadata_col])

    return f"""
<|im_start|>system<|im_sep|>
you are an expert malacologist specialized in mollusc taxonomy. use the specimen image and metadata to infer the missing molluscan taxonomic order. be careful not to confuse taxonomic ranks.
<|im_end|>
<|im_start|>user<|im_sep|>
<|image_1|>

your task is to infer the missing molluscan order from the specimen image and metadata.

the metadata is order-cleaned: direct order information has been removed, but class, family, genus, species, locality, and specimen context may remain.

taxonomic hierarchy reminder:
phylum > class > order > family > genus > species

important rules:
- predict the order only.
- do not output a class, subclass, superfamily, family, genus, or species as the order.
- if metadata contains family, genus, or species, use it as evidence to infer the parent order.
- the predicted order must be taxonomically compatible with the class and family evidence.
- if uncertain, still provide the most plausible order and mark confidence as low.
- do not invent a higher rank from a family name by simply changing the suffix.

metadata:
{metadata}

return exactly this format, with no extra sections:

visual evidence: <one concise sentence>
metadata evidence: <one concise sentence>
rank check: <confirm that the final prediction is an order, not a family/genus/species>
taxonomic assessment: <one concise sentence linking class/family evidence to the predicted order>
phylum: mollusca
class: <predicted class>
family evidence: <family/genus/species evidence used, if available>
order: <one taxonomic order only>
confidence: high / medium / low
justification: <two concise sentences explaining why this order fits the image and metadata>
<|im_end|>
<|im_start|>assistant<|im_sep|>
""".strip()

def clean_output(text):
    text = str(text)

    if "<|im_start|>assistant<|im_sep|>" in text:
        text = text.split("<|im_start|>assistant<|im_sep|>")[-1]

    if "<|assistant|>" in text:
        text = text.split("<|assistant|>")[-1]

    return text.strip()


def extract_field(output, field_name):
    pattern = rf"^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$"

    for line in str(output).splitlines():
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def match_type(predicted_order, true_order):
    if predicted_order is None or str(predicted_order).strip() == "":
        return "unparsed"

    pred = str(predicted_order).strip().lower()
    true = str(true_order).strip().lower()

    if pred == true:
        return "exact_match"

    if true in pred or pred in true:
        return "partial_string_match"

    return "no_match"


def flag_possible_rank_confusion(predicted_order):
    if predicted_order is None:
        return None

    pred = str(predicted_order).strip().lower()

    if pred.endswith(("idae", "inae", "oidea")):
        return True

    return False


def run_phi_case(row, max_new_tokens=240):
    img_path = Path(row["local_image_path"])

    if not img_path.exists() or img_path.stat().st_size == 0:
        raise FileNotFoundError(f"missing cached image: {img_path}")

    image = Image.open(img_path).convert("RGB")
    prompt = build_prompt(row)

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    new_tokens = generated_ids[:, input_len:]

    output = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
    )[0]

    return prompt, clean_output(output)

#running

outputs = []

for i, row in phi_df.iterrows():
    print("\n" + "=" * 80, flush=True)
    print(f"running case {i + 1}/{len(phi_df)}", flush=True)
    print("gbifid:", row["gbifID"], flush=True)
    print("true order:", row["label"], flush=True)

    try:
        if str(row["image_cache_status"]).startswith("failed"):
            raise RuntimeError(row["image_cache_status"])

        prompt, output = run_phi_case(row)

        pred_class = extract_field(output, "class")
        family_evidence = extract_field(output, "family evidence")
        pred_order = extract_field(output, "order")
        confidence = extract_field(output, "confidence")
        rank_check = extract_field(output, "rank check")
        taxonomic_assessment = extract_field(output, "taxonomic assessment")
        justification = extract_field(output, "justification")
        mtype = match_type(pred_order, row["label"])
        possible_rank_confusion = flag_possible_rank_confusion(pred_order)

        row_out = {
            "gbifID": row["gbifID"],
            "true_order": row["label"],
            "order_count_global": row["order_count_global"],
            "order_rarity_global": row["order_rarity_global"],
            "metadata_condition": metadata_condition_name,
            "subset_type": subset_type,
            "image_url": row["image"],
            "local_image_path": row["local_image_path"],
            "image_cache_status": row["image_cache_status"],
            "prompt": prompt,
            "phi_output": output,
            "phi_predicted_class_parsed": pred_class,
            "phi_family_evidence_parsed": family_evidence,
            "phi_predicted_order_parsed": pred_order,
            "phi_confidence_parsed": confidence,
            "phi_rank_check_parsed": rank_check,
            "phi_taxonomic_assessment_parsed": taxonomic_assessment,
            "phi_justification_parsed": justification,
            "possible_rank_confusion": possible_rank_confusion,
            "string_match_to_true_order": mtype,
            "error": None,
        }

        print(output, flush=True)
        print("parsed order:", pred_order, flush=True)
        print("match:", mtype, flush=True)
        print("possible rank confusion:", possible_rank_confusion, flush=True)

    except Exception as e:
        row_out = {
            "gbifID": row["gbifID"],
            "true_order": row["label"],
            "order_count_global": row["order_count_global"],
            "order_rarity_global": row["order_rarity_global"],
            "metadata_condition": metadata_condition_name,
            "subset_type": subset_type,
            "image_url": row["image"],
            "local_image_path": row.get("local_image_path", None),
            "image_cache_status": row.get("image_cache_status", None),
            "prompt": None,
            "phi_output": None,
            "phi_predicted_class_parsed": None,
            "phi_family_evidence_parsed": None,
            "phi_predicted_order_parsed": None,
            "phi_confidence_parsed": None,
            "phi_rank_check_parsed": None,
            "phi_taxonomic_assessment_parsed": None,
            "phi_justification_parsed": None,
            "possible_rank_confusion": None,
            "string_match_to_true_order": None,
            "error": str(e),
        }

        print("error:", e, flush=True)

    outputs.append(row_out)

    pd.DataFrame(outputs).to_csv(full_out, index=False)
    print("saved partial:", full_out, flush=True)

results_df = pd.DataFrame(outputs)
results_df.to_csv(full_out, index=False)

summary_cols = [
    "gbifID",
    "true_order",
    "phi_predicted_order_parsed",
    "phi_confidence_parsed",
    "possible_rank_confusion",
    "string_match_to_true_order",
    "error",
]

print("\nsaved full results:", full_out, flush=True)
print(results_df[summary_cols].to_string(index=False), flush=True)

assert len(results_df) == expected_n, f"expected {expected_n} saved results, found {len(results_df)}."
assert full_out.exists(), f"missing full results file: {full_out}"
print(f"\nsaved {len(results_df)} joint-failure results to {full_out}", flush=True)

#selecting example case

successful = results_df[results_df["error"].isna()].copy()

if len(successful) > 0:
    priority = {
        "exact_match": 0,
        "partial_string_match": 1,
        "no_match": 2,
        "unparsed": 3,
    }

    successful["selection_priority"] = (
        successful["string_match_to_true_order"]
        .map(priority)
        .fillna(9)
    )

    selected_case = (
        successful
        .sort_values(["selection_priority", "possible_rank_confusion", "true_order", "gbifID"])
        .iloc[0]
    )

    pd.DataFrame([selected_case]).to_csv(selected_out, index=False)

    print("\nselected case:", flush=True)
    print("gbifid:", selected_case["gbifID"], flush=True)
    print("true order:", selected_case["true_order"], flush=True)
    print("phi order:", selected_case["phi_predicted_order_parsed"], flush=True)
    print("match:", selected_case["string_match_to_true_order"], flush=True)
    print("rank confusion:", selected_case["possible_rank_confusion"], flush=True)
    print("\noutput:", flush=True)
    print(selected_case["phi_output"], flush=True)
    print("\nsaved selected case:", selected_out, flush=True)
else:
    print("\nno successful outputs. selected-case file not created.", flush=True)

print("\ndone.", flush=True)
