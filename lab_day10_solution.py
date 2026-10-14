"""Runnable helpers for the Day 10 Qwen3-VL road-damage lab.

This file owns the learner implementation. Instructor data preparation, scoring,
configuration, tests, and instructions remain unchanged.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.labcommon import CLASSES, read_jsonl, write_jsonl
from scripts.score import iou_xyxy, parse_output


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
WORK_DIR = ROOT / "work"
EVIDENCE_DIR = ROOT / "report" / "evidence"
FIGURES_DIR = ROOT / "report" / "figures"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROMPT = (
    "Locate road damage in this image. Detect only objects of classes: pothole, "
    "crack, manhole. Output a JSON list of objects with label and bbox_2d "
    "coordinates normalized on a 0-1000 scale: "
    '[{"label": "...", "bbox_2d": [x1, y1, x2, y2]}]. '
    "If no damage is found, output []."
)
EXPECTED_COUNTS = {
    "train": {"images": 1422, "boxes": {"pothole": 1074, "crack": 1750, "manhole": 691}},
    "val": {"images": 276, "boxes": {"pothole": 66, "crack": 470, "manhole": 103}},
    "test": {"images": 311, "boxes": {"pothole": 121, "crack": 299, "manhole": 163}},
}


def yolo_to_box(label: str, yolo: list[float]) -> dict:
    """Convert normalized YOLO center-width-height to integer 0–1000 XYXY."""
    xc, yc, width, height = (float(v) for v in yolo)
    x1 = int(round(max(0.0, min(1000.0, (xc - width / 2) * 1000))))
    y1 = int(round(max(0.0, min(1000.0, (yc - height / 2) * 1000))))
    x2 = int(round(max(0.0, min(1000.0, (xc + width / 2) * 1000))))
    y2 = int(round(max(0.0, min(1000.0, (yc + height / 2) * 1000))))
    if x2 <= x1:
        x2 = min(1000, x1 + 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 1)
    return {"label": label, "bbox_2d": [x1, y1, x2, y2]}


def objects_to_answer(objects: list[dict]) -> str:
    boxes = [yolo_to_box(obj["label"], obj["yolo"]) for obj in objects]
    boxes.sort(key=lambda box: (box["bbox_2d"][1], box["bbox_2d"][0]))
    return json.dumps(boxes, separators=(",", ":"), ensure_ascii=False)


def verify_dataset(*, strict_counts: bool = True) -> dict:
    """Check expected split sizes/classes, dataset signature, and session isolation."""
    meta_path = SPLITS_DIR / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Chưa có data/splits/meta.json; hãy chạy prepare_data.py trước.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    version = meta.get("dataset_version", "")
    if not version.startswith("d10-"):
        raise ValueError(f"dataset_version phải bắt đầu bằng d10-, nhận được {version!r}")

    split_rows = {name: read_jsonl(SPLITS_DIR / f"{name}.jsonl") for name in EXPECTED_COUNTS}
    session_owner: dict[str, str] = {}
    for split, rows in split_rows.items():
        counts = {"images": len(rows), "boxes": {name: 0 for name in CLASSES}}
        for row in rows:
            session = row.get("session")
            if not session:
                raise ValueError(f"{split}/{row.get('image_id')}: thiếu session")
            previous = session_owner.setdefault(session, split)
            if previous != split:
                raise ValueError(f"Session {session!r} xuất hiện ở {previous} và {split}")
            for obj in row.get("objects", []):
                if obj.get("label") not in CLASSES:
                    raise ValueError(f"Nhãn không hợp lệ: {obj.get('label')!r}")
                counts["boxes"][obj["label"]] += 1
        if strict_counts and counts != EXPECTED_COUNTS[split]:
            raise ValueError(
                f"Số liệu {split} không khớp yêu cầu. Thực tế={counts}; "
                f"kỳ vọng={EXPECTED_COUNTS[split]}"
            )
        if counts != meta.get("counts", {}).get(split):
            raise ValueError(f"Số liệu file {split}.jsonl không khớp meta.json: {counts}")
        print(f"{split}: {counts['images']} ảnh, boxes={counts['boxes']}")
    print(f"dataset_version={version}; {len(session_owner)} sessions không giao nhau giữa split")
    return {"meta": meta, "splits": split_rows}


def _image_path(entry: dict) -> Path:
    path = DATA_DIR / entry["image_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _ground_truth_pixel_box(obj: dict, width: int, height: int) -> tuple[int, int, int, int]:
    xc, yc, box_w, box_h = obj["yolo"]
    return (
        round((xc - box_w / 2) * width), round((yc - box_h / 2) * height),
        round((xc + box_w / 2) * width), round((yc + box_h / 2) * height),
    )


def _label_font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        return ImageFont.load_default()


def visualize_ground_truth(records: list[dict] | None = None, count: int = 6) -> Path:
    """Save a six-image contact sheet of dataset labels for visual inspection."""
    if records is None:
        records = read_jsonl(SPLITS_DIR / "train.jsonl")
    chosen = records[:count]
    if len(chosen) < count:
        raise ValueError(f"Cần ít nhất {count} ảnh, hiện tại có {len(chosen)}")
    thumb_w, thumb_h, caption_h = 640, 360, 26
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + caption_h) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = _label_font()
    for index, row in enumerate(chosen):
        with Image.open(_image_path(row)) as source:
            image = source.convert("RGB").resize((thumb_w, thumb_h))
        image_draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = _ground_truth_pixel_box(obj, row["width"], row["height"])
            image_draw.rectangle(box, outline=(35, 210, 90), width=3)
            image_draw.text((box[0] + 2, max(0, box[1] - 15)), obj["label"],
                            fill=(20, 130, 50), font=font, stroke_width=1, stroke_fill="white")
        x, y = (index % 2) * thumb_w, (index // 2) * (thumb_h + caption_h)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + thumb_h + 4), f"GT | {row['image_id']} | {len(row['objects'])} boxes",
                  fill="black", font=font)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / "ground_truth_samples.png"
    sheet.save(out)
    print(f"Đã lưu ảnh kiểm tra nhãn: {out}")
    return out


def _torch_and_transformers():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    return torch, AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration


def load_processor():
    _, AutoProcessor, _, _ = _torch_and_transformers()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.tokenizer.padding_side = "right"
    return processor


def load_base_model():
    torch, _, BitsAndBytesConfig, Qwen3VLForConditionalGeneration = _torch_and_transformers()
    if not torch.cuda.is_available():
        raise RuntimeError("Bài lab cần Google Colab GPU T4; torch.cuda.is_available() = False")
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["visual", "lm_head"],
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quantization, dtype=torch.float16, device_map="auto"
    )
    model.config.use_cache = False
    return model


def _device_of(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    raise RuntimeError("Không xác định được device của model")


def encode_example(image: Image.Image, answer: str, processor) -> dict:
    user_messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}
    ]}]
    complete_messages = user_messages + [{"role": "assistant", "content": answer}]
    prompt_enc = processor.apply_chat_template(
        user_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )
    full_enc = processor.apply_chat_template(
        complete_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    prompt_len = prompt_enc["input_ids"].shape[1]
    if not torch_equal(full_enc["input_ids"][0, :prompt_len], prompt_enc["input_ids"][0]):
        raise AssertionError("Chat template đầy đủ không có prompt làm tiền tố; mask loss sẽ sai.")
    labels = full_enc["input_ids"].clone()
    labels[:, :prompt_len] = -100
    sample = {
        "input_ids": full_enc["input_ids"][0],
        "attention_mask": full_enc["attention_mask"][0],
        "labels": labels[0],
        "pixel_values": full_enc["pixel_values"],
        "image_grid_thw": full_enc["image_grid_thw"],
    }
    if "mm_token_type_ids" in full_enc:
        sample["mm_token_type_ids"] = full_enc["mm_token_type_ids"][0]
    return sample


def torch_equal(left, right) -> bool:
    return bool((left == right).all().item())


def collate_examples(batch: list[dict], pad_token_id: int) -> dict:
    import torch
    from torch.nn.utils.rnn import pad_sequence

    result = {
        "input_ids": pad_sequence([x["input_ids"] for x in batch], batch_first=True,
                                  padding_value=pad_token_id),
        "attention_mask": pad_sequence([x["attention_mask"] for x in batch], batch_first=True,
                                       padding_value=0),
        "labels": pad_sequence([x["labels"] for x in batch], batch_first=True,
                               padding_value=-100),
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0),
        "image_grid_thw": torch.cat([x["image_grid_thw"] for x in batch], dim=0),
    }
    if all("mm_token_type_ids" in x for x in batch):
        result["mm_token_type_ids"] = pad_sequence(
            [x["mm_token_type_ids"] for x in batch], batch_first=True, padding_value=0
        )
    return result


class RoadDamageDataset:
    def __init__(self, records: list[dict], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with Image.open(_image_path(row)) as source:
            image = source.convert("RGB")
        return encode_example(image, objects_to_answer(row["objects"]), self.processor)


def inspect_masking_and_collation():
    import torch
    rows = read_jsonl(SPLITS_DIR / "train.jsonl")
    processor = load_processor()
    with Image.open(_image_path(rows[0])) as source:
        sample = encode_example(source.convert("RGB"), objects_to_answer(rows[0]["objects"]), processor)
    target_ids = sample["input_ids"][sample["labels"] != -100]
    decoded = processor.decode(target_ids, skip_special_tokens=False)
    if not decoded.startswith("[") or "<|im_end|>" not in decoded:
        raise AssertionError(f"Token đích không phải JSON assistant kết thúc im_end: {decoded!r}")
    batch = collate_examples([sample], processor.tokenizer.pad_token_id or 151643)
    grid = sample["image_grid_thw"].long()
    merge_size = getattr(processor.image_processor, "merge_size", 2)
    visual_tokens = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item() // (merge_size ** 2))
    print(f"Target sau mask: {decoded}")
    print(f"Grid ảnh: {grid.tolist()}, spatial merge={merge_size}, token thị giác thực tế={visual_tokens}")
    print("220 token là kỳ vọng của ảnh 640x360; dùng số thực tế của processor hiện tại.")
    print(f"Batch keys={list(batch)}, pixel_values shape={tuple(batch['pixel_values'].shape)}")
    assert bool((sample["labels"][:len(sample["labels"]) - len(target_ids)] == -100).all().item())
    return processor


def _move_batch(batch: dict, device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _chat_inputs(processor, image: Image.Image):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": PROMPT}
    ]}]
    return processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    )


def predict_records(model, processor, records: list[dict], *, max_new_tokens: int = 256) -> list[dict]:
    import torch
    model.eval()
    device = _device_of(model)
    output = []
    for n, row in enumerate(records, 1):
        with Image.open(_image_path(row)) as source:
            image = source.convert("RGB")
        inputs = _chat_inputs(processor, image)
        inputs = _move_batch(inputs, device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                       do_sample=False, use_cache=True)
        new_tokens = generated[0, prompt_len:]
        raw = processor.decode(new_tokens, skip_special_tokens=True).strip()
        output.append({"image_id": row["image_id"], "raw": raw})
        if n % 50 == 0 or n == len(records):
            print(f"Inference: {n}/{len(records)}")
    return output


def _save_predictions(name: str, predictions: list[dict]) -> Path:
    path = WORK_DIR / f"{name.lower()}.jsonl"
    write_jsonl(path, predictions)
    print(f"Đã lưu {len(predictions)} dự đoán: {path}")
    return path


def run_zero_shot() -> Path:
    rows = read_jsonl(SPLITS_DIR / "test.jsonl")
    processor = load_processor()
    model = load_base_model()
    try:
        return _save_predictions("e0", predict_records(model, processor, rows))
    finally:
        del model, processor
        gc.collect()
        import torch
        torch.cuda.empty_cache()


def _lora_targets(setup_name: str) -> list[str]:
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if setup_name == "E3":
        targets += ["qkv", "linear_fc1", "linear_fc2"]
    return targets


def apply_setup(model, setup_name: str, *, rank: int = 16):
    from peft import LoraConfig, get_peft_model
    if setup_name not in {"E1", "E2", "E3"}:
        raise ValueError(f"Setup không hợp lệ: {setup_name}")
    if rank < 1:
        raise ValueError("LoRA rank phải >= 1")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.enable_input_require_grads()
    peft_model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules=_lora_targets(setup_name), bias="none", task_type="CAUSAL_LM",
    ))
    if setup_name in {"E2", "E3"}:
        opened = 0
        for name, parameter in peft_model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                parameter.data = parameter.data.to(dtype=__import__("torch").float32)
                parameter.requires_grad_(True)
                opened += parameter.numel()
        if opened == 0:
            raise RuntimeError("Không tìm thấy visual.merger/deepstack_merger_list để mở train.")
    adapter_layers = sum(1 for name, _ in peft_model.named_parameters() if ".lora_A." in name)
    expected_layers = 252 if setup_name in {"E1", "E2"} else 348
    if adapter_layers != expected_layers:
        raise RuntimeError(
            f"LoRA gắn vào {adapter_layers} module, kỳ vọng {expected_layers} theo lab "
            f"({setup_name}, transformers={getattr(__import__('transformers'), '__version__', '?')})."
        )
    return peft_model


def _projector_state(model) -> dict:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if "visual.merger" in name or "visual.deepstack_merger_list" in name
    }


def _restore_projector(model, path: Path) -> None:
    import torch
    if not path.exists():
        raise FileNotFoundError(f"Thiếu trọng số projector đã train: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    current = dict(model.named_parameters())
    missing = sorted(set(saved) - set(current))
    if missing:
        raise KeyError(f"Projector checkpoint không khớp model; thiếu {missing[:3]}")
    with torch.no_grad():
        for name, value in saved.items():
            current[name].copy_(value.to(device=current[name].device, dtype=current[name].dtype))


def _save_model(model, processor, adapter_dir: Path, setup_name: str) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)
    projector_file = adapter_dir / "projector.pt"
    if setup_name in {"E2", "E3"}:
        state = _projector_state(model)
        if not state:
            raise RuntimeError("Không có tham số projector để lưu.")
        import torch
        torch.save(state, projector_file)


def _manifest_path() -> Path:
    path = WORK_DIR / "runs" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _update_manifest(key: str, run_info: dict) -> None:
    path = _manifest_path()
    current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    current[key] = run_info
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")


def _score(pred_path: Path, *, name: str, split: str, out_path: Path, meta: dict | None = None) -> dict:
    command = [sys.executable, "scripts/score.py", "--pred", str(pred_path), "--split", split,
               "--name", name, "--out", str(out_path)]
    for key, value in (meta or {}).items():
        command += ["--meta", f"{key}={value}"]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _make_run_dir(name: str, rank: int) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = WORK_DIR / "runs" / f"{name.lower()}-r{rank}-{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _backup_to_drive(checkpoint_dir: Path, run_dir: Path) -> None:
    drive_root = Path("/content/drive/MyDrive/day10_checkpoints")
    if drive_root.exists():
        target = drive_root / run_dir.name / checkpoint_dir.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(checkpoint_dir, target, dirs_exist_ok=True)
        print(f"Drive backup: {target}")
    else:
        print("Drive chưa mount; checkpoint vẫn được lưu trong work/.")


def _train_one(setup_name: str, *, rank: int = 16, steps: int = 300,
               records: list[dict] | None = None) -> dict:
    import torch
    from torch.utils.data import DataLoader

    if records is None:
        records = read_jsonl(SPLITS_DIR / "train.jsonl")
    random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = load_processor()
    model = apply_setup(load_base_model(), setup_name, rank=rank)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in trainable)
    if not trainable_params:
        raise RuntimeError("Không có trainable parameters.")
    run_dir = _make_run_dir(setup_name, rank)
    data = RoadDamageDataset(records, processor)
    loader = DataLoader(
        data, batch_size=2, shuffle=True, drop_last=True, num_workers=0,
        collate_fn=lambda items: collate_examples(items, processor.tokenizer.pad_token_id or 151643),
    )
    if not len(loader):
        raise ValueError("Dataset phải có ít nhất 2 ảnh để huấn luyện.")
    optimizer = torch.optim.AdamW(trainable, lr=2e-4, weight_decay=0.0)
    warmup_steps = max(1, int(steps * 0.05))

    def lr_factor(step):
        if step < warmup_steps:
            return max(1e-8, (step + 1) / warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    device = _device_of(model)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for step in range(1, steps + 1):
        step_loss = 0.0
        for _ in range(4):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            batch = _move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = model(**batch).loss
                scaled_loss = loss / 4
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Loss không hữu hạn tại {setup_name} step {step}: {loss}")
            scaler.scale(scaled_loss).backward()
            step_loss += float(loss.detach().float().item()) / 4
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(step_loss)
        if step == 1 or step % 10 == 0:
            print(f"{setup_name} r={rank} step={step}/{steps} loss={step_loss:.5f} "
                  f"lr={scheduler.get_last_lr()[0]:.3g}")
        if step % 100 == 0:
            checkpoint = run_dir / f"checkpoint-{step:04d}"
            _save_model(model, processor, checkpoint / "adapter", setup_name)
            (checkpoint / "progress.json").write_text(json.dumps({
                "step": step, "loss": step_loss, "trainable_params": trainable_params,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            }, indent=2), encoding="utf-8")
            _backup_to_drive(checkpoint, run_dir)

    train_minutes = (time.monotonic() - started) / 60
    peak_vram_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    adapter_dir = run_dir / "adapter"
    _save_model(model, processor, adapter_dir, setup_name)
    run_info = {
        "setup": setup_name, "rank": rank, "alpha": rank * 2, "steps": steps,
        "trainable_params": trainable_params, "peak_vram_gb": round(peak_vram_gb, 4),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 4),
        "train_minutes": round(train_minutes, 4), "final_loss": losses[-1],
        "adapter_dir": str(adapter_dir), "run_dir": str(run_dir),
        "projector_saved": setup_name in {"E2", "E3"},
        "dataset_version": json.loads((SPLITS_DIR / "meta.json").read_text(encoding="utf-8"))["dataset_version"],
    }
    (run_dir / "run.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
    _update_manifest(f"{setup_name}_r{rank}", run_info)
    del optimizer, scheduler, model, processor, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return run_info


def _load_experiment(run_info: dict, *, trainable: bool = False):
    import torch
    from peft import PeftModel
    processor = load_processor()
    base = load_base_model()
    model = PeftModel.from_pretrained(base, run_info["adapter_dir"], is_trainable=trainable)
    setup_name = run_info["setup"]
    projector_path = Path(run_info["adapter_dir"]) / "projector.pt"
    if setup_name in {"E2", "E3"}:
        for name, parameter in model.named_parameters():
            if "visual.merger" in name or "visual.deepstack_merger_list" in name:
                parameter.data = parameter.data.to(dtype=torch.float32 if trainable else torch.float16)
                parameter.requires_grad_(trainable)
        _restore_projector(model, projector_path)
    model.config.use_cache = not trainable
    if trainable:
        model.enable_input_require_grads()
    return model, processor


def _run_supervised_sweep() -> dict:
    validation = read_jsonl(SPLITS_DIR / "val.jsonl")
    result = {}
    for setup_name in ("E1", "E2", "E3"):
        run_info = _train_one(setup_name, rank=16, steps=300)
        model, processor = _load_experiment(run_info)
        try:
            pred_path = _save_predictions(f"{setup_name}_val", predict_records(model, processor, validation))
            score = _score(pred_path, name=setup_name, split="val",
                           out_path=Path(run_info["run_dir"]) / "val_evidence.json",
                           meta={k: run_info[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")})
        finally:
            del model, processor
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        run_info["val_f1"] = score["micro"]["f1"]
        run_info["val_ci95"] = score["f1_ci95"]
        run_info["val_evidence"] = str(Path(run_info["run_dir"]) / "val_evidence.json")
        result[setup_name] = run_info
        _update_manifest(f"{setup_name}_r16", run_info)
    best = max(result, key=lambda name: result[name]["val_f1"])
    print(f"Chọn setup theo validation (không dùng test): {best}, F1 val={result[best]['val_f1']:.4f}")
    state = json.loads(_manifest_path().read_text(encoding="utf-8"))
    state["best_setup"] = best
    _manifest_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def train_supervised_sweep():
    checks = verify_dataset()
    print(f"Huấn luyện trên {len(checks['splits']['train'])} train ảnh; validation dùng chọn setup.")
    return _run_supervised_sweep()


def _manifest() -> dict:
    path = _manifest_path()
    if not path.exists():
        raise FileNotFoundError("Chưa có training manifest; chạy train_supervised_sweep() trước.")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_supervised() -> dict:
    test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")
    manifest = _manifest()
    scores = {}
    for setup_name in ("E1", "E2", "E3"):
        run_info = manifest.get(f"{setup_name}_r16")
        if not run_info:
            raise FileNotFoundError(f"Thiếu run của {setup_name}; chạy train_supervised_sweep() trước.")
        model, processor = _load_experiment(run_info)
        try:
            pred_path = _save_predictions(setup_name.lower(), predict_records(model, processor, test_rows))
        finally:
            del model, processor
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        meta = {k: run_info[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
        evidence = _score(pred_path, name=setup_name, split="test",
                          out_path=EVIDENCE_DIR / f"{setup_name.lower()}.json", meta=meta)
        scores[setup_name] = evidence

    best = manifest.get("best_setup")
    if best not in ("E1", "E2", "E3"):
        raise ValueError(f"Không có best_setup trong manifest: {best}")
    best_predictions = read_jsonl(WORK_DIR / f"{best.lower()}.jsonl")
    make_prediction_contact_sheet(test_rows, best_predictions, count=6)
    e0_path = EVIDENCE_DIR / "e0.json"
    if e0_path.exists():
        e0_score = json.loads(e0_path.read_text(encoding="utf-8"))["micro"]["f1"]
        e1 = scores["E1"]
        gain = e1["micro"]["f1"] - e0_score
        print(f"Acceptance E1: parse_valid_rate={e1['parse_valid_rate']:.4f} "
              f"({'PASS' if e1['parse_valid_rate'] >= 0.95 else 'FAIL'}; cần ≥0.95); "
              f"F1 gain={gain:.4f} ({'PASS' if gain >= 0.10 else 'FAIL'}; cần ≥0.10)")
    return scores


def make_prediction_contact_sheet(records: list[dict], predictions: list[dict], count: int = 6) -> Path:
    """Draw six actual test examples with green GT boxes and red model boxes."""
    raw_by_id = {row["image_id"]: row["raw"] for row in predictions}
    thumb_w, thumb_h, caption_h = 640, 360, 36
    sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + caption_h) * 3), "white")
    font = _label_font()
    scored_rows = []
    for row in records:
        raw = raw_by_id.get(row["image_id"], "")
        parsed = parse_output(raw) or []
        gt_boxes = [{"label": obj["label"], "bbox": tuple(yolo_to_box(obj["label"], obj["yolo"])["bbox_2d"])}
                    for obj in row["objects"]]
        tp = 0
        for label in set(CLASSES) | {obj["label"] for obj in parsed}:
            gt = [obj["bbox"] for obj in gt_boxes if obj["label"] == label]
            pred = [obj["bbox"] for obj in parsed if obj["label"] == label]
            pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(gt) for j, b in enumerate(pred)),
                           reverse=True)
            used_gt, used_pred = set(), set()
            for value, i, j in pairs:
                if value < 0.5:
                    break
                if i in used_gt or j in used_pred:
                    continue
                used_gt.add(i)
                used_pred.add(j)
                tp += 1
        fp, fn = len(parsed) - tp, len(gt_boxes) - tp
        scored_rows.append((row, raw, tp, fp, fn))

    buckets = [
        [item for item in scored_rows if item[2] > 0 and item[3] == 0 and item[4] == 0],
        [item for item in scored_rows if item[2] > 0 and (item[3] > 0 or item[4] > 0)],
        [item for item in scored_rows if item[4] > 0],
        [item for item in scored_rows if item[3] > 0],
        [item for item in scored_rows if item[2] > 0],
        scored_rows,
    ]
    chosen = []
    seen = set()
    for bucket in buckets:
        for item in bucket:
            image_id = item[0]["image_id"]
            if image_id not in seen:
                chosen.append(item)
                seen.add(image_id)
                break
        if len(chosen) >= count:
            break
    for item in scored_rows:
        if len(chosen) >= count:
            break
        if item[0]["image_id"] not in seen:
            chosen.append(item)
            seen.add(item[0]["image_id"])

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for index, (row, raw, tp, fp, fn) in enumerate(chosen[:count]):
        image_path = _image_path(row)
        with Image.open(image_path) as source:
            image = source.convert("RGB").resize((thumb_w, thumb_h))
        draw = ImageDraw.Draw(image)
        for obj in row["objects"]:
            box = _ground_truth_pixel_box(obj, row["width"], row["height"])
            draw.rectangle(box, outline=(30, 210, 80), width=3)
            draw.text((box[0] + 2, max(0, box[1] - 15)), f"GT {obj['label']}",
                      fill=(10, 120, 40), font=font, stroke_width=1, stroke_fill="white")
        parsed = parse_output(raw) or []
        for obj in parsed:
            x1, y1, x2, y2 = obj["bbox"]
            box = (round(x1 / 1000 * row["width"]), round(y1 / 1000 * row["height"]),
                   round(x2 / 1000 * row["width"]), round(y2 / 1000 * row["height"]))
            draw.rectangle(box, outline=(240, 40, 40), width=3)
            draw.text((box[0] + 2, min(thumb_h - 15, box[1] + 2)), f"Pred {obj['label']}",
                      fill=(180, 20, 20), font=font, stroke_width=1, stroke_fill="white")
        x, y = (index % 2) * thumb_w, (index // 2) * (thumb_h + caption_h)
        sheet.paste(image, (x, y))
        draw_sheet = ImageDraw.Draw(sheet)
        caption = f"{row['image_id']} | TP={tp} FP={fp} FN={fn} | GT green / Pred red"
        draw_sheet.text((x + 6, y + thumb_h + 5), caption,
                        fill="black", font=font)
        safe_id = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in row["image_id"])
        image.save(FIGURES_DIR / f"prediction_example_{index + 1:02d}_{safe_id}.png")
    out = FIGURES_DIR / "best_validation_setup_test_examples.png"
    sheet.save(out)
    print(f"Đã lưu 6 ảnh GT/Pred trên test: {out}")
    return out


def run_rank_ablation() -> dict:
    """Run E4 rank 4/64 with every other chosen setup parameter unchanged."""
    manifest = _manifest()
    base_setup = manifest.get("best_setup")
    if base_setup not in {"E1", "E2", "E3"}:
        raise ValueError("Cần best_setup được chọn trên validation trước E4.")
    test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")
    results = {}
    for rank in (4, 64):
        run_info = _train_one(base_setup, rank=rank, steps=300)
        run_info["ablation_base_setup"] = base_setup
        run_info["only_changed_variable"] = "lora_rank"
        _update_manifest(f"E4_rank_{rank}", run_info)
        model, processor = _load_experiment(run_info)
        try:
            preds = _save_predictions(f"e4_rank_{rank}", predict_records(model, processor, test_rows))
        finally:
            del model, processor
            gc.collect()
            import torch
            torch.cuda.empty_cache()
        meta = {k: run_info[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}
        evidence = _score(preds, name=f"E4_rank_{rank}", split="test",
                          out_path=EVIDENCE_DIR / f"e4_rank_{rank}.json", meta=meta)
        evidence["ablation"] = {"base_setup": base_setup, "only_changed_variable": "lora_rank",
                                 "rank": rank, "alpha": rank * 2}
        (EVIDENCE_DIR / f"e4_rank_{rank}.json").write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        results[rank] = evidence
    ci_a, ci_b = results[4]["f1_ci95"], results[64]["f1_ci95"]
    overlap = not (ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0])
    print(f"E4 CI95 rank 4={ci_a}; rank 64={ci_b}; overlap={overlap}")
    return results


def _gt_norm_boxes(objects: list[dict]) -> list[dict]:
    output = []
    for obj in objects:
        xc, yc, width, height = obj["yolo"]
        output.append({"label": obj["label"], "bbox": [
            (xc - width / 2) * 1000, (yc - height / 2) * 1000,
            (xc + width / 2) * 1000, (yc + height / 2) * 1000,
        ]})
    return output


def compute_iou_reward(completion: str, gt_boxes: list[dict]) -> float:
    """Verifiable IoU reward described in the lab; invalid JSON receives zero."""
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total_iou = 0.0
    labels = {obj["label"] for obj in gt_boxes} | {obj["label"] for obj in pred}
    for label in labels:
        gt = [obj["bbox"] for obj in gt_boxes if obj["label"] == label]
        predicted = [obj["bbox"] for obj in pred if obj["label"] == label]
        pairs = sorted(
            ((iou_xyxy(a, b), i, j) for i, a in enumerate(gt) for j, b in enumerate(predicted)),
            reverse=True,
        )
        used_gt, used_pred = set(), set()
        for value, i, j in pairs:
            if value < 0.5:
                break
            if i in used_gt or j in used_pred:
                continue
            used_gt.add(i)
            used_pred.add(j)
            total_iou += value
    denom = max(len(gt_boxes), len(pred), 1)
    return 0.2 + 0.8 * (total_iou / denom)


def check_iou_reward_cases() -> dict:
    gt = [{"label": "pothole", "bbox": [100, 100, 300, 300]}]
    perfect = '[{"label":"pothole","bbox_2d":[100,100,300,300]}]'
    wrong_label = '[{"label":"crack","bbox_2d":[100,100,300,300]}]'
    extra = '[{"label":"pothole","bbox_2d":[100,100,300,300]},' \
            '{"label":"crack","bbox_2d":[500,500,700,700]}]'
    got = {
        "perfect": compute_iou_reward(perfect, gt),
        "invalid_json": compute_iou_reward("not json", gt),
        "wrong_label": compute_iou_reward(wrong_label, gt),
        "extra_box": compute_iou_reward(extra, gt),
        "both_empty": compute_iou_reward("[]", []),
    }
    expected = {"perfect": 1.0, "invalid_json": 0.0, "wrong_label": 0.2,
                "extra_box": 0.6, "both_empty": 1.0}
    for key, value in expected.items():
        if not math.isclose(got[key], value, abs_tol=1e-8):
            raise AssertionError(f"Reward {key}: nhận {got[key]}, kỳ vọng {value}")
    print(f"Verifiable IoU reward self-check: 5/5 — {got}")
    return got


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return _completion_text(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    parts.extend(str(piece.get("text", "")) if isinstance(piece, dict) else str(piece)
                                 for piece in content)
                else:
                    parts.append(str(content))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def grpo_iou_reward(completions, gt_boxes_json=None, **kwargs):
    if gt_boxes_json is None:
        gt_boxes_json = kwargs.get("gt_boxes_json", [])
    rewards = []
    for completion, serialized_gt in zip(completions, gt_boxes_json):
        gt = json.loads(serialized_gt) if isinstance(serialized_gt, str) else serialized_gt
        rewards.append(compute_iou_reward(_completion_text(completion), gt))
    return rewards


def run_grpo(*, steps: int = 60) -> dict:
    """Optional E5. TRL 1.9.2 docs do not certify Qwen3-VL compatibility."""
    check_iou_reward_cases()
    try:
        from datasets import Dataset
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError("E5 yêu cầu datasets và trl==1.9.2 trong Colab.") from error
    manifest = _manifest()
    best_setup = manifest.get("best_setup")
    base_info = manifest.get(f"{best_setup}_r16")
    if not base_info:
        raise ValueError("Chưa có E1–E3; hãy huấn luyện SFT trước E5.")
    train_rows = read_jsonl(SPLITS_DIR / "train.jsonl")
    processor = load_processor()
    dataset_rows = []
    for row in train_rows:
        with Image.open(_image_path(row)) as source:
            image = source.convert("RGB")
        dataset_rows.append({
            "prompt": [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": PROMPT}
            ]}],
            "image": image,
            "gt_boxes_json": json.dumps(_gt_norm_boxes(row["objects"])),
        })
    model, processor = _load_experiment(base_info, trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out_dir = WORK_DIR / "E5_GRPO"
    output_dir = out_dir / "trainer"
    args = GRPOConfig(
        output_dir=str(output_dir), learning_rate=5e-6, max_steps=steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, max_completion_length=256, beta=0.0,
        use_vllm=False, logging_steps=1, save_steps=30, report_to="none",
        remove_unused_columns=False, gradient_checkpointing=True, fp16=True,
    )
    try:
        trainer = GRPOTrainer(
            model=model, args=args, train_dataset=Dataset.from_list(dataset_rows),
            reward_funcs=grpo_iou_reward, processing_class=processor,
        )
        started = time.monotonic()
        trainer.train()
        train_minutes = (time.monotonic() - started) / 60
        adapter_dir = out_dir / "adapter"
        trainer.save_model(str(adapter_dir))
        processor.save_pretrained(adapter_dir)
        if best_setup in {"E2", "E3"}:
            import torch
            torch.save(_projector_state(trainer.model), adapter_dir / "projector.pt")
        (out_dir / "training_log.json").write_text(json.dumps({
            "best_sft_setup": best_setup, "steps": steps, "train_minutes": train_minutes,
            "log_history": trainer.state.log_history,
            "reward_hacking_checks": [
                "Empty prediction on non-empty GT earns only the 0.2 floor; both-empty earns 1.0 by rubric.",
                "A giant full-frame box must pass same-class IoU >= 0.5 and is penalized as an extra/poor match.",
            ],
            "trl_version": __import__("trl").__version__,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        test_rows = read_jsonl(SPLITS_DIR / "test.jsonl")
        predictions = _save_predictions("e5", predict_records(trainer.model, processor, test_rows))
        evidence = _score(predictions, name="E5", split="test", out_path=EVIDENCE_DIR / "e5.json",
                          meta={"train_minutes": round(train_minutes, 4), "base_sft_setup": best_setup,
                                "grpo_steps": steps})
        sft_path = EVIDENCE_DIR / f"{best_setup.lower()}.json"
        if sft_path.exists():
            sft = json.loads(sft_path.read_text(encoding="utf-8"))
            sft_ci, e5_ci = sft["f1_ci95"], evidence["f1_ci95"]
            comparison = {
                "sft_setup": best_setup, "sft_f1": sft["micro"]["f1"],
                "e5_f1": evidence["micro"]["f1"],
                "f1_delta": evidence["micro"]["f1"] - sft["micro"]["f1"],
                "sft_ci95": sft_ci, "e5_ci95": e5_ci,
                "ci95_overlap": not (sft_ci[1] < e5_ci[0] or e5_ci[1] < sft_ci[0]),
            }
            evidence["sft_comparison"] = comparison
            (EVIDENCE_DIR / "e5.json").write_text(
                json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"E5 vs {best_setup}: ΔF1={comparison['f1_delta']:.4f}; "
                  f"CI95 overlap={comparison['ci95_overlap']}")
        return evidence
    except Exception as error:
        if "Qwen3" in str(error) or "model_type" in str(error).lower() or "unsupported" in str(error).lower():
            raise RuntimeError(
                "TRL GRPO không tương thích với model/runtime hiện tại. Reward 5/5 đã qua; "
                "chưa có E5 evidence. Giữ nguyên lỗi gốc để chẩn đoán phiên bản."
            ) from error
        raise
