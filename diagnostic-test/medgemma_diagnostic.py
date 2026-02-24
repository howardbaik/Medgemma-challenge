# classify 13 categorical(7+6) and 2 regression tasks on A4C echo images using MedGemma.
import os
import argparse
import json
import re
import time
from collections import Counter
from glob import glob

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


IMAGE_EXTS = (".jpg")
VIDEO_EXTS = (".mp4")

# ---- Hardcoded paths (set USE_HARDCODED=True to ignore CLI inputs) ----
USE_HARDCODED = True
HARDCODED_IMAGE = None
HARDCODED_STUDY_UID = None
HARDCODED_IMAGES_ROOT = None
HARDCODED_GLOB_PATTERN = "*"
RUN_ALL_STUDIES = True
RUN_ALL_ROOT = "data/all_A4C_test_896" # A4C images path, use size 896
RUN_ALL_OUT = "medgemma_predictions_a4c_test_896_2plus1.jsonl"

TASKS_JSON_PATH = "assets/tasks.json"

CATEGORICAL_TASKS = [
    "rv_systolic_function",
    "lv_size",
    "la_size",
    "rv_size",
    "ra_size",
    "mv_regurgitation",
    "av_regurgitation",
    "tv_regurgitation",
    "pv_regurgitation",
    "mv_stenosis",
    "av_stenosis",
    "pv_stenosis",
    "tv_stenosis",
]

REGRESSION_TASKS = ["lv_lvef", "pa_pressure_numerical"]

TASK_GROUPS = [
    ["rv_systolic_function", "lv_size", "la_size", "rv_size", "ra_size", "mv_regurgitation"],
    ["av_regurgitation", "tv_regurgitation", "pv_regurgitation", "mv_stenosis", "av_stenosis", "pv_stenosis", "tv_stenosis"],
]

PROMPT_PREFIX_CAT = (
    "<start_of_image>\n"
    "You are given an echocardiography image. Predict the following categorical findings.\n"
    "For each task, output the class index (integer) according to the class list provided.\n"
    "Output ONE LINE of JSON only. Keys must be exactly as listed and in the same order.\n"
    "Do NOT output null. Always output an integer class for every key.\n"
    "Do not output any extra text or multiple candidates.\n\n"
)

PROMPT_PREFIX_REG = (
    "<start_of_image>\n"
    "Estimate the following continuous measurements from the echo image.\n"
    "Output ONE LINE of JSON only. Keys must be exactly as listed and in the same order.\n"
    "For each key, output a value within its specified range (up to 2 decimals).\n"
    "Do NOT output null. Always output a number for every key.\n"
    "Do not output any extra text.\n\n"
)

OUTPUT_JSON = True
DEFAULT_AGGREGATE = "vote"
UNKNOWN_TOKEN = "unknown"
PRINT_TIMING = False
PRINT_RAW_OUTPUT = False
SCALE_LVEF_LT1 = True
DEFAULT_REG_RANGE = (0.0, 200.0)
STRICT_SUFFIX = "\nSTRICT: Output values ONLY. No null. No explanations. One JSON line.\n"


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _list_media(images_root: str, study_uid: str, pattern: str | None) -> list[str]:
    if not images_root:
        raise SystemExit("--images_root is required when using --study_uid")
    if pattern:
        if "{study_uid}" in pattern:
            path_pattern = os.path.join(images_root, pattern.format(study_uid=study_uid))
        else:
            path_pattern = os.path.join(images_root, study_uid, pattern)
        return sorted(glob(path_pattern))
    base_dir = os.path.join(images_root, study_uid)
    matches = []
    for ext in IMAGE_EXTS + VIDEO_EXTS:
        matches.extend(glob(os.path.join(base_dir, f"*{ext}")))
    return sorted(matches)


def _iter_study_dirs(root_dir: str):
    if not os.path.isdir(root_dir):
        raise SystemExit(f"Root dir not found: {root_dir}")
    # If root_dir is a flat folder of images, yield each file as its own "study"
    flat_files = [
        os.path.join(root_dir, f)
        for f in sorted(os.listdir(root_dir))
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS
    ]
    if flat_files:
        for path in flat_files:
            name = os.path.basename(path)
            study_uid = name.split("__", 1)[0]
            yield root_dir, study_uid, path
        return
    # Otherwise, expect date/study hierarchy
    for day_name in sorted(os.listdir(root_dir)):
        day_dir = os.path.join(root_dir, day_name)
        if not os.path.isdir(day_dir):
            continue
        for study_uid in sorted(os.listdir(day_dir)):
            study_dir = os.path.join(day_dir, study_uid)
            if os.path.isdir(study_dir):
                yield day_dir, study_uid, None


def _load_processed_paths(jsonl_path: str) -> set[str]:
    processed = set()
    if not os.path.exists(jsonl_path):
        return processed
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = obj.get("media_path")
            if isinstance(path, str):
                processed.add(path)
    return processed


def _load_task_defs(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_prompt_cat(task_keys: list[str], task_defs: dict) -> str:
    lines = [PROMPT_PREFIX_CAT, "Class definitions:\n"]
    for key in task_keys:
        values = task_defs.get(key, {}).get("values", [])
        lines.append(f"{key}:\n")
        for idx, phrases in enumerate(values):
            label = phrases[0] if isinstance(phrases, list) and phrases else phrases
            lines.append(f"{idx} = {label}\n")
        lines.append("\n")
    lines.append("Output JSON keys (in order):\n")
    lines.append("[")
    lines.append(", ".join([f'\"{k}\"' for k in task_keys]))
    lines.append("]\n")
    return "".join(lines)


def _build_prompt_reg(task_keys: list[str], task_defs: dict) -> str:
    lines = [PROMPT_PREFIX_REG, "Ranges:\n"]
    for key in task_keys:
        r = task_defs.get(key, {}).get("range")
        if isinstance(r, list) and len(r) == 2:
            lines.append(f"{key}: range {r[0]} to {r[1]}\n")
        else:
            lines.append(f"{key}: range {DEFAULT_REG_RANGE[0]} to {DEFAULT_REG_RANGE[1]}\n")
    lines.append("\nkeys = [")
    lines.append(", ".join([f'\"{k}\"' for k in task_keys]))
    lines.append("]")
    return "".join(lines)


def _strip_code_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _extract_json_block(text: str) -> str | None:
    def _extract_balanced(s: str, open_ch: str, close_ch: str) -> str | None:
        starts = [i for i, ch in enumerate(s) if ch == open_ch]
        if not starts:
            return None
        last_block = None
        for start in starts:
            depth = 0
            for i in range(start, len(s)):
                if s[i] == open_ch:
                    depth += 1
                elif s[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        last_block = s[start : i + 1]
                        break
        return last_block

    obj = _extract_balanced(text, "{", "}")
    arr = _extract_balanced(text, "[", "]")
    if obj and arr:
        return arr if text.rfind("[") > text.rfind("{") else obj
    return obj or arr


def _repair_json_text(text: str) -> str | None:
    text = _strip_code_fences(text)
    block = _extract_json_block(text)
    if not block:
        return None
    s = block.strip()
    s = s.replace("'", "\"")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _parse_json_any(text: str):
    s = _repair_json_text(text)
    if not s:
        return None
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


def _normalize_categorical_value(v, num_classes: int):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = int(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+", v)
        if not m:
            return None
        x = int(m.group(0))
    else:
        return None
    if x < 0 or x >= num_classes:
        return None
    return x


def _normalize_reg_value(v, key: str, task_defs: dict):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        x = float(v)
    elif isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        if not m:
            return None
        x = float(m.group(0))
    else:
        return None
    if key == "lv_lvef" and SCALE_LVEF_LT1 and x < 1:
        x = x * 100.0
    r = task_defs.get(key, {}).get("range")
    if isinstance(r, list) and len(r) == 2:
        min_v, max_v = float(r[0]), float(r[1])
    else:
        min_v, max_v = DEFAULT_REG_RANGE
    if x < min_v or x > max_v:
        return None
    return round(x, 2)


def _normalize_json(data: dict, task_keys: list[str], task_defs: dict, mode: str) -> dict:
    out = {}
    if mode == "categorical":
        for k in task_keys:
            values = task_defs.get(k, {}).get("values", [])
            out[k] = _normalize_categorical_value(data.get(k), len(values))
    else:
        for k in task_keys:
            out[k] = _normalize_reg_value(data.get(k), k, task_defs)
    return out


def _format_json_line(data: dict, task_keys: list[str]) -> str:
    ordered = {k: data.get(k) for k in task_keys}
    return json.dumps(ordered, ensure_ascii=True, separators=(",", ":"))


def _postprocess_output(raw_text: str, task_keys: list[str], task_defs: dict, mode: str):
    if OUTPUT_JSON:
        parsed = _parse_json_any(raw_text)
        if isinstance(parsed, list):
            if len(parsed) != len(task_keys):
                return UNKNOWN_TOKEN, None
            mapped = {k: parsed[i] for i, k in enumerate(task_keys)}
            norm = _normalize_json(mapped, task_keys, task_defs, mode)
            return _format_json_line(norm, task_keys), norm
        if isinstance(parsed, dict):
            norm = _normalize_json(parsed, task_keys, task_defs, mode)
            return _format_json_line(norm, task_keys), norm
        return UNKNOWN_TOKEN, None
    return raw_text, None


def _aggregate_json_vote(results: list[dict], task_keys: list[str]) -> dict | None:
    counts = {k: {} for k in task_keys}
    for r in results:
        d = r.get("json")
        if not isinstance(d, dict):
            continue
        for k in task_keys:
            v = d.get(k)
            if isinstance(v, (int, float)):
                counts[k][v] = counts[k].get(v, 0) + 1
    out = {}
    for k in task_keys:
        if not counts[k]:
            out[k] = None
        else:
            out[k] = max(counts[k], key=counts[k].get)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Test MedGemma for 13 categorical + 2 regression tasks.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--image", help="Path to a single input image or video.")
    group.add_argument("--study_uid", help="Study UID to load multiple images.")
    parser.add_argument("--images_root", default=None, help="Root dir that contains study subfolders.")
    parser.add_argument("--glob_pattern", default=None)
    parser.add_argument("--model_id", default="google/medgemma-1.5-4b-it")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hf_token", default=None)
    parser.add_argument("--print_per_view", action="store_true")
    args = parser.parse_args()

    if USE_HARDCODED and not args.image and not args.study_uid:
        args.image = HARDCODED_IMAGE
        args.study_uid = HARDCODED_STUDY_UID
        if args.images_root is None:
            args.images_root = HARDCODED_IMAGES_ROOT
        if args.glob_pattern is None:
            args.glob_pattern = HARDCODED_GLOB_PATTERN

    if RUN_ALL_STUDIES:
        args.image = None
        args.study_uid = None

    if args.hf_token:
        from huggingface_hub import login
        login(args.hf_token)

    if PRINT_TIMING:
        print("Loading model/processor...")
    t0 = time.time()

    dtype = _resolve_dtype(args.dtype)
    if not torch.cuda.is_available() and dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, torch_dtype=dtype, device_map=args.device_map
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    model.eval()
    if PRINT_TIMING:
        p = next(model.parameters())
        print(f"Model device: {p.device}, dtype: {p.dtype}")
        print(f"Model/processor loaded in {time.time() - t0:.1f}s")

    task_defs = _load_task_defs(TASKS_JSON_PATH)
    missing = [k for k in CATEGORICAL_TASKS if k not in task_defs]
    if missing:
        raise SystemExit(f"Missing tasks in tasks.json: {missing}")
    group_prompts = [(group, _build_prompt_cat(group, task_defs)) for group in TASK_GROUPS]
    reg_prompt = _build_prompt_reg(REGRESSION_TASKS, task_defs)

    generate_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    if args.temperature and args.temperature > 0:
        generate_kwargs.update(dict(do_sample=True, temperature=args.temperature))
    else:
        generate_kwargs.update(dict(do_sample=False))

    def generate_one(image: Image.Image, prompt: str) -> str:
        boi_token = getattr(processor, "boi_token", None) or getattr(
            processor.tokenizer, "boi_token", None
        ) or "<start_of_image>"
        if boi_token not in prompt:
            prompt = f"{boi_token} {prompt}".strip()
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            outputs = model.generate(**inputs, **generate_kwargs)
        return processor.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_with_retry(
        image: Image.Image,
        prompt: str,
        task_keys: list[str],
        task_defs: dict,
        mode: str,
        retries: int = 7,
        forbid_null: bool = False,
    ) -> str:
        raw = generate_one(image, prompt)
        if OUTPUT_JSON:
            parsed = _parse_json_any(raw)
            if parsed is None or ("{" not in raw and "[" not in raw):
                if retries > 0:
                    return generate_with_retry(
                        image, prompt, task_keys, task_defs, mode, retries=retries - 1, forbid_null=forbid_null
                    )
                return raw
            if forbid_null:
                # If any null is present (raw or normalized), retry
                if "null" in raw.lower():
                    if retries > 0:
                        return generate_with_retry(
                            image,
                            prompt + STRICT_SUFFIX,
                            task_keys,
                            task_defs,
                            mode,
                            retries=retries - 1,
                            forbid_null=forbid_null,
                        )
                if isinstance(parsed, list):
                    if len(parsed) == len(task_keys):
                        mapped = {k: parsed[i] for i, k in enumerate(task_keys)}
                    else:
                        mapped = {}
                elif isinstance(parsed, dict):
                    mapped = parsed
                else:
                    mapped = {}
                norm = _normalize_json(mapped, task_keys, task_defs, mode)
                if any(v is None for v in norm.values()):
                    if retries > 0:
                        return generate_with_retry(
                            image,
                            prompt + STRICT_SUFFIX,
                            task_keys,
                            task_defs,
                            mode,
                            retries=retries - 1,
                            forbid_null=forbid_null,
                        )
        return raw

    def load_as_image(path: str) -> Image.Image:
        if _is_video(path):
            try:
                import cv2
            except Exception:
                cv2 = None
            if cv2 is not None:
                cap = cv2.VideoCapture(path)
                ok, frame = cap.read()
                cap.release()
                if not ok:
                    raise SystemExit(f"Failed to read first frame: {path}")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return Image.fromarray(frame)
            import imageio.v3 as iio
            frame = iio.imread(path, index=0)
            if frame.ndim == 2:
                frame = frame[:, :, None].repeat(3, axis=2)
            return Image.fromarray(frame)
        return Image.open(path).convert("RGB")

    def process_media(path: str) -> dict:
        image = load_as_image(path)
        merged = {k: None for k in CATEGORICAL_TASKS + REGRESSION_TASKS}
        raw_outputs = []

        for group, prompt in group_prompts:
            raw_text = generate_with_retry(
                image, prompt, group, task_defs, "categorical", retries=3, forbid_null=True
            )
            if PRINT_RAW_OUTPUT:
                print(raw_text)
            raw_outputs.append(raw_text)
            _, json_obj = _postprocess_output(raw_text, group, task_defs, "categorical")
            if isinstance(json_obj, dict):
                for k in group:
                    merged[k] = json_obj.get(k)

        raw_text = generate_with_retry(
            image, reg_prompt, REGRESSION_TASKS, task_defs, "regression", retries=7, forbid_null=True
        )
        if PRINT_RAW_OUTPUT:
            print(raw_text)
        raw_outputs.append(raw_text)
        _, json_obj = _postprocess_output(raw_text, REGRESSION_TASKS, task_defs, "regression")
        if isinstance(json_obj, dict):
            for k in REGRESSION_TASKS:
                merged[k] = json_obj.get(k)

        return {
            "text": _format_json_line(merged, CATEGORICAL_TASKS + REGRESSION_TASKS),
            "json": merged,
            "raw": raw_outputs,
        }

    def process_study(images_root: str, study_uid: str):
        image_paths = _list_media(images_root, study_uid, args.glob_pattern)
        if not image_paths:
            return None, None, None
        image_paths = [p for p in image_paths if os.path.splitext(p)[1].lower() in (IMAGE_EXTS + VIDEO_EXTS)]

        results = []
        for idx, path in enumerate(image_paths, start=1):
            if PRINT_TIMING:
                print(f"[{study_uid}] [{idx}/{len(image_paths)}] Loading: {path}")
            if PRINT_TIMING:
                print(f"[{study_uid}] [{idx}/{len(image_paths)}] Generating...")
            out = process_media(path)
            results.append(
                {
                    "image_path": path,
                    "text": out["text"],
                    "json": out["json"],
                    "raw": out["raw"],
                    "view_index": idx,
                }
            )
            if args.print_per_view:
                print(f"[VIEW {idx}] {path}")
                print(out["text"])
                print("-" * 40)

        if DEFAULT_AGGREGATE == "vote":
            agg_json = _aggregate_json_vote(results, CATEGORICAL_TASKS + REGRESSION_TASKS)
            aggregated = _format_json_line(agg_json, CATEGORICAL_TASKS + REGRESSION_TASKS) if agg_json else UNKNOWN_TOKEN
        else:
            aggregated = UNKNOWN_TOKEN

        return results, aggregated, image_paths

    if RUN_ALL_STUDIES:
        out_path = RUN_ALL_OUT
        processed_paths = _load_processed_paths(out_path)
        start = time.time()
        total = len([p for p in os.listdir(RUN_ALL_ROOT) if os.path.splitext(p)[1].lower() in IMAGE_EXTS]) if os.path.isdir(RUN_ALL_ROOT) else 0
        done = 0
        with open(out_path, "a", encoding="utf-8", buffering=1) as f:
            for day_dir, study_uid, single_path in _iter_study_dirs(RUN_ALL_ROOT):
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"Progress: {done}/{total} | {rate:.2f}/s | ETA {eta/60:.1f} min")
                if single_path:
                    if single_path in processed_paths:
                        continue
                    out = process_media(single_path)
                    record = {
                        "study_uid": study_uid,
                        "day_dir": day_dir,
                        "media_path": single_path,
                        "media_name": os.path.basename(single_path),
                        "model_output_All15": out.get("text"),
                    }
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    f.flush()
                    processed_paths.add(single_path)
                    continue
                results, aggregated, image_paths = process_study(day_dir, study_uid)
                if results is None:
                    continue
                for r in results:
                    media_path = r.get("image_path")
                    if not media_path or media_path in processed_paths:
                        continue
                    record = {
                        "study_uid": study_uid,
                        "day_dir": day_dir,
                        "media_path": media_path,
                        "media_name": os.path.basename(media_path) if media_path else None,
                        "model_output_All15": r.get("text"),
                    }
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    f.flush()
                    processed_paths.add(media_path)
        print(f"Saved predictions to {out_path}")
        return

    if not args.image and not args.study_uid:
        raise SystemExit("Provide --image or --study_uid, or set USE_HARDCODED with paths.")

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"Image not found: {args.image}")
        if PRINT_TIMING:
            print(f"Loading image: {args.image}")
        t1 = time.time()
        image = load_as_image(args.image)
        if PRINT_TIMING:
            print(f"Image loaded in {time.time() - t1:.1f}s")
            print("Generating...")
        t2 = time.time()
        out = process_media(args.image)
        if PRINT_TIMING:
            print(f"Generation finished in {time.time() - t2:.1f}s")
        print(out["text"])
        return

    results, aggregated, image_paths = process_study(args.images_root, args.study_uid)
    if results is None:
        raise SystemExit(f"No images found for study_uid={args.study_uid}")
    print(aggregated)


if __name__ == "__main__":
    main()
