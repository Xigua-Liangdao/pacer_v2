import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def get_active_labels(label_mode: str) -> List[str]:
    if label_mode == "binary":
        return ["negative", "positive"]
    if label_mode == "ternary":
        return ["negative", "neutral", "positive"]
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def log(message: str) -> None:
    import time

    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def weighted_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    if not y_true:
        return 0.0
    support = Counter(y_true)
    total = len(y_true)
    weighted = 0.0
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        weighted += (support.get(label, 0) / total) * f1
    return weighted


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    mat = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        if t in mat and p in mat[t]:
            mat[t][p] += 1
    return mat


def evaluate_split(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict:
    return {
        "accuracy": round(accuracy(y_true, y_pred), 6),
        "weighted_f1": round(weighted_f1(y_true, y_pred, labels), 6),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
    }


def build_prompt_templates(prompt_set: str) -> List[str]:
    if prompt_set == "single":
        return ["The sentiment is <LABEL>."]
    if prompt_set == "vision_sentiment_7":
        return [
            "The person appears <LABEL>.",
            "The facial expression looks <LABEL>.",
            "The speaker's visual emotion is <LABEL>.",
            "The video shows a <LABEL> emotional state.",
            "This face conveys <LABEL> sentiment.",
            "The visible reaction is <LABEL>.",
            "The visual mood is <LABEL>.",
        ]
    if prompt_set == "sentiment_5":
        return [
            "The sentiment is <LABEL>.",
            "This utterance is <LABEL>.",
            "Overall sentiment: <LABEL>.",
            "The speaker expresses <LABEL> sentiment.",
            "This opinion sounds <LABEL>.",
        ]
    if prompt_set == "sentiment_7":
        return [
            "The sentiment is <LABEL>.",
            "This utterance is <LABEL>.",
            "Overall sentiment: <LABEL>.",
            "The speaker expresses <LABEL> sentiment.",
            "This opinion sounds <LABEL>.",
            "The emotional polarity is <LABEL>.",
            "Sentiment label: <LABEL>.",
        ]
    custom = [x.strip() for x in prompt_set.split("||") if x.strip()]
    return custom if custom else ["The sentiment is <LABEL>."]


def build_class_prompts(prompt_set: str, labels: List[str]) -> List[List[str]]:
    templates = build_prompt_templates(prompt_set)
    return [[tpl.replace("<LABEL>", label) for tpl in templates] for label in labels]


def build_prompt_group_indices(num_prompts: int, group_size: int) -> List[List[int]]:
    if group_size <= 0 or group_size >= num_prompts:
        return [list(range(num_prompts))]
    groups = []
    for start in range(0, num_prompts, group_size):
        groups.append(list(range(start, min(start + group_size, num_prompts))))
    return groups


def sentiment_to_class(value: float, label_mode: str):
    if label_mode == "binary":
        if value == 0:
            return None, None
        return ("positive", 1) if value > 0 else ("negative", 0)
    if label_mode == "ternary":
        if value > 0:
            return "positive", 2
        if value < 0:
            return "negative", 0
        return "neutral", 1
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def load_processed_dataset(input_path: str, label_mode: str):
    with Path(input_path).open("rb") as f:
        data = pickle.load(f)

    result = {}
    for split_name in ["train", "valid", "test"]:
        payload = data[split_name]
        vision = np.asarray(payload["vision"], dtype=np.float32)
        sample_ids = [str(x) for x in payload["id"]]
        vision_mask = (np.abs(vision).sum(axis=-1) > 0).astype(np.float32)
        raw_labels = np.asarray(payload["regression_labels"], dtype=np.float32).reshape(-1)

        keep_indices = []
        label_ids = []
        label_names = []
        for idx, value in enumerate(raw_labels):
            name, class_id = sentiment_to_class(float(value), label_mode)
            if name is None:
                continue
            keep_indices.append(idx)
            label_ids.append(class_id)
            label_names.append(name)

        keep = np.asarray(keep_indices, dtype=np.int64)
        result[split_name] = {
            "vision_seq": vision[keep],
            "vision_mask": vision_mask[keep],
            "y": np.asarray(label_ids, dtype=np.int64),
            "label_names": label_names,
            "id": [sample_ids[i] for i in keep.tolist()],
            "vision_dim": int(vision.shape[-1]),
            "seq_len": int(vision.shape[1]),
        }
    return result


class MsaClipAdapter:
    def __init__(
        self,
        vision_dim: int,
        clip_dim: int,
        hidden_dim: int,
        dropout: float,
        device: str,
        num_classes: int,
        num_prompts: int,
    ):
        import torch
        import torch.nn as nn

        self.device = device
        self.max_seq_len = 50
        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_dim),
            nn.GELU(),
        ).to(device)
        self.position_embed = nn.Parameter(torch.zeros(1, self.max_seq_len, clip_dim, device=device))
        nn.init.normal_(self.position_embed, std=0.01)

        self.logit_scale = nn.Parameter(torch.tensor(1.0, device=device))
        self.prompt_weight_logits = nn.Parameter(torch.zeros(num_classes, num_prompts, device=device))
        self.class_logit_scale = nn.Parameter(torch.zeros(num_classes, device=device))
        self.class_bias = nn.Parameter(torch.zeros(num_classes, device=device))

    def parameters(self):
        return (
            list(self.vision_proj.parameters())
            + [self.position_embed, self.logit_scale, self.prompt_weight_logits, self.class_logit_scale, self.class_bias]
        )

    def state_dict(self):
        return {
            "vision_proj": self.vision_proj.state_dict(),
            "position_embed": self.position_embed.detach().cpu().clone(),
            "logit_scale": self.logit_scale.detach().cpu().clone(),
            "prompt_weight_logits": self.prompt_weight_logits.detach().cpu().clone(),
            "class_logit_scale": self.class_logit_scale.detach().cpu().clone(),
            "class_bias": self.class_bias.detach().cpu().clone(),
        }

    def load_state_dict(self, state):
        self.vision_proj.load_state_dict(state["vision_proj"])
        self.position_embed.data.copy_(state["position_embed"].to(self.device))
        self.logit_scale.data.copy_(state["logit_scale"].to(self.device))
        self.prompt_weight_logits.data.copy_(state["prompt_weight_logits"].to(self.device))
        self.class_logit_scale.data.copy_(state["class_logit_scale"].to(self.device))
        self.class_bias.data.copy_(state["class_bias"].to(self.device))

    def train(self):
        self.vision_proj.train()

    def eval(self):
        self.vision_proj.eval()

    def _encode_vision_tokens(self, vision_seq, vision_mask):
        import torch

        seq_len = vision_seq.shape[1]
        pos = self.position_embed[:, :seq_len, :]
        tokens = self.vision_proj(vision_seq) + pos
        tokens = tokens / tokens.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        mask = vision_mask.unsqueeze(-1)
        return tokens * mask

    def logits(self, vision_seq, vision_mask, prompt_features, prompt_mask):
        import torch.nn.functional as F
        import torch

        vision_tokens = self._encode_vision_tokens(vision_seq, vision_mask)
        txt = prompt_features / prompt_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bvd,cptd->bcpvt", vision_tokens, txt)
        text_mask = prompt_mask.unsqueeze(0).unsqueeze(3)
        sim = sim.masked_fill(text_mask == 0, -1e4)
        sim = sim.max(dim=-1).values
        time_mask = vision_mask.unsqueeze(1).unsqueeze(1)
        sim = (sim * time_mask).sum(dim=-1) / time_mask.sum(dim=-1).clamp(min=1.0)
        prompt_w = F.softmax(self.prompt_weight_logits, dim=-1).unsqueeze(0)
        class_sim = (sim * prompt_w).sum(dim=-1)
        global_scale = self.logit_scale.exp().clamp(max=100.0)
        class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).unsqueeze(0)
        return global_scale * class_sim * class_scale + self.class_bias.unsqueeze(0)

    def grouped_logits(self, vision_seq, vision_mask, prompt_features, prompt_mask, group_indices: List[List[int]]):
        import torch

        vision_tokens = self._encode_vision_tokens(vision_seq, vision_mask)
        txt = prompt_features / prompt_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        sim = torch.einsum("bvd,cptd->bcpvt", vision_tokens, txt)
        text_mask = prompt_mask.unsqueeze(0).unsqueeze(3)
        sim = sim.masked_fill(text_mask == 0, -1e4)
        sim = sim.max(dim=-1).values
        time_mask = vision_mask.unsqueeze(1).unsqueeze(1)
        sim = (sim * time_mask).sum(dim=-1) / time_mask.sum(dim=-1).clamp(min=1.0)
        group_scores = []
        for gidx in group_indices:
            group_scores.append(sim[:, :, gidx].mean(dim=-1))
        scores = torch.stack(group_scores, dim=-1)
        global_scale = self.logit_scale.exp().clamp(max=100.0)
        class_scale = self.class_logit_scale.exp().clamp(min=0.5, max=2.5).view(1, -1, 1)
        return global_scale * scores * class_scale + self.class_bias.view(1, -1, 1)


def load_clip_processor_and_model(model_id: str, device: str, clip_mode: str):
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    if clip_mode == "auto":
        try:
            processor = CLIPTokenizer.from_pretrained(model_id)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False)
        except Exception:
            processor = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
            model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)
    else:
        processor = CLIPTokenizer.from_pretrained(model_id, local_files_only=True)
        model = CLIPModel.from_pretrained(model_id, use_safetensors=False, local_files_only=True)

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    return processor, model


def extract_prompt_features(prompt_groups: List[List[str]], processor, model, device: str):
    import torch

    features = []
    masks = []
    for prompts in prompt_groups:
        inputs = processor(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_outputs = model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            token_features = model.text_projection(text_outputs.last_hidden_state)
            token_features = token_features / token_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        token_mask = inputs["attention_mask"].clone()
        special_mask = (inputs["input_ids"] == processor.bos_token_id) | (inputs["input_ids"] == processor.eos_token_id)
        token_mask = token_mask * (~special_mask).long()
        features.append(token_features.float().detach())
        masks.append(token_mask.float().detach())
    return torch.stack(features, dim=0), torch.stack(masks, dim=0)


def predict_labels(split_x, prompt_features, prompt_mask, adapter, batch_size, use_test_ensemble, ensemble_group_size, labels: List[str]):
    import torch

    group_indices = build_prompt_group_indices(int(prompt_features.shape[1]), ensemble_group_size)
    preds = []
    for start in range(0, split_x["vision_seq"].shape[0], batch_size):
        vx = split_x["vision_seq"][start:start + batch_size].to(adapter.device)
        vm = split_x["vision_mask"][start:start + batch_size].to(adapter.device)
        with torch.no_grad():
            if use_test_ensemble and len(group_indices) > 1:
                g_logits = adapter.grouped_logits(vx, vm, prompt_features.to(adapter.device), prompt_mask.to(adapter.device), group_indices)
                group_pred = g_logits.argmax(dim=1)
                total_scores = g_logits.sum(dim=-1)
                idxs = []
                for i in range(group_pred.shape[0]):
                    votes = torch.bincount(group_pred[i], minlength=len(labels))
                    top = votes.max()
                    cands = (votes == top).nonzero(as_tuple=False).view(-1)
                    if cands.numel() == 1:
                        idxs.append(int(cands[0].item()))
                    else:
                        cs = total_scores[i, cands]
                        idxs.append(int(cands[cs.argmax()].item()))
            else:
                logits = adapter.logits(vx, vm, prompt_features.to(adapter.device), prompt_mask.to(adapter.device))
                idxs = logits.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([labels[i] for i in idxs])
    return preds
def train_model(train_split, val_split, prompt_features, prompt_mask, adapter, epochs, batch_size, lr, weight_decay, max_grad_norm, use_class_weight, label_smoothing, select_metric, use_test_ensemble, ensemble_group_size, labels: List[str]):
    import torch
    import torch.nn as nn

    class_weights = None
    if use_class_weight:
        class_counts = torch.bincount(train_split["y"], minlength=len(labels)).float()
        class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(adapter.device)
        class_weights = class_weights / class_weights.mean().clamp(min=1e-12)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_metric = -1.0

    for epoch in range(epochs):
        adapter.train()
        perm = torch.randperm(train_split["vision_seq"].shape[0])
        train_vision_seq = train_split["vision_seq"][perm]
        train_vision_mask = train_split["vision_mask"][perm]
        train_y = train_split["y"][perm]

        losses = []
        for start in range(0, train_vision_seq.shape[0], batch_size):
            vx = train_vision_seq[start:start + batch_size].to(adapter.device)
            vm = train_vision_mask[start:start + batch_size].to(adapter.device)
            by = train_y[start:start + batch_size].to(adapter.device)
            logits = adapter.logits(vx, vm, prompt_features.to(adapter.device), prompt_mask.to(adapter.device))
            loss = criterion(logits, by)
            optimizer.zero_grad()
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))

        adapter.eval()
        val_pred = predict_labels(val_split, prompt_features, prompt_mask, adapter, batch_size, use_test_ensemble, ensemble_group_size, labels)
        val_true = [labels[int(x)] for x in val_split["y"].cpu().tolist()]
        val_acc = accuracy(val_true, val_pred)
        val_wf1 = weighted_f1(val_true, val_pred, labels)
        metric = val_wf1 if select_metric == "weighted_f1" else val_acc
        log(f"epoch={epoch + 1} loss={sum(losses)/max(len(losses),1):.4f} val_acc={val_acc:.4f} val_wf1={val_wf1:.4f}")

        if metric > best_metric:
            best_metric = metric
            best_state = adapter.state_dict()

    if best_state is not None:
        adapter.load_state_dict(best_state)
    adapter.eval()
    return adapter


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP-style MOSI/MOSEI vision-sequence to text-prompt sentiment training")
    parser.add_argument("--dataset", required=True, choices=["mosi", "mosei"])
    parser.add_argument("--input", required=True)
    parser.add_argument("--label_mode", default="binary", choices=["binary", "ternary"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_mode", choices=["offline_only", "auto"], default="auto")
    parser.add_argument("--model_id", default="openai/clip-vit-base-patch32")
    parser.add_argument("--prompt_set", default="vision_sentiment_7")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adapter_hidden_dim", type=int, default=512)
    parser.add_argument("--adapter_dropout", type=float, default=0.2)
    parser.add_argument("--use_class_weight", action="store_true")
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--select_metric", choices=["accuracy", "weighted_f1"], default="weighted_f1")
    parser.add_argument("--use_test_ensemble", action="store_true")
    parser.add_argument("--ensemble_group_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    import torch

    torch.manual_seed(args.seed)

    active_labels = get_active_labels(args.label_mode)

    dataset = load_processed_dataset(args.input, args.label_mode)
    prompt_groups = build_class_prompts(args.prompt_set, active_labels)
    processor, model = load_clip_processor_and_model(args.model_id, args.device, args.clip_mode)
    prompt_features, prompt_mask = extract_prompt_features(prompt_groups, processor, model, args.device)
    clip_dim = int(prompt_features.shape[-1])

    train_split = {
        "vision_seq": torch.tensor(dataset["train"]["vision_seq"], dtype=torch.float32),
        "vision_mask": torch.tensor(dataset["train"]["vision_mask"], dtype=torch.float32),
        "y": torch.tensor(dataset["train"]["y"], dtype=torch.long),
    }
    val_split = {
        "vision_seq": torch.tensor(dataset["valid"]["vision_seq"], dtype=torch.float32),
        "vision_mask": torch.tensor(dataset["valid"]["vision_mask"], dtype=torch.float32),
        "y": torch.tensor(dataset["valid"]["y"], dtype=torch.long),
    }
    test_split = {
        "vision_seq": torch.tensor(dataset["test"]["vision_seq"], dtype=torch.float32),
        "vision_mask": torch.tensor(dataset["test"]["vision_mask"], dtype=torch.float32),
        "y": torch.tensor(dataset["test"]["y"], dtype=torch.long),
    }

    adapter = MsaClipAdapter(
        vision_dim=dataset["train"]["vision_dim"],
        clip_dim=clip_dim,
        hidden_dim=args.adapter_hidden_dim,
        dropout=args.adapter_dropout,
        device=args.device,
        num_classes=len(active_labels),
        num_prompts=int(prompt_features.shape[1]),
    )

    adapter = train_model(
        train_split=train_split,
        val_split=val_split,
        prompt_features=prompt_features,
        prompt_mask=prompt_mask,
        adapter=adapter,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        use_class_weight=args.use_class_weight,
        label_smoothing=args.label_smoothing,
        select_metric=args.select_metric,
        use_test_ensemble=args.use_test_ensemble,
        ensemble_group_size=args.ensemble_group_size,
        labels=active_labels,
    )

    val_pred = predict_labels(val_split, prompt_features, prompt_mask, adapter, args.batch_size, args.use_test_ensemble, args.ensemble_group_size, active_labels)
    test_pred = predict_labels(test_split, prompt_features, prompt_mask, adapter, args.batch_size, args.use_test_ensemble, args.ensemble_group_size, active_labels)
    val_true = [active_labels[int(x)] for x in val_split["y"].cpu().tolist()]
    test_true = [active_labels[int(x)] for x in test_split["y"].cpu().tolist()]

    result = {
        "config": {
            "dataset": args.dataset,
            "input": str(Path(args.input).resolve()),
            "label_mode": args.label_mode,
            "model_id": args.model_id,
            "prompt_set": args.prompt_set,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "adapter_hidden_dim": args.adapter_hidden_dim,
            "adapter_dropout": args.adapter_dropout,
            "use_class_weight": args.use_class_weight,
            "label_smoothing": args.label_smoothing,
            "select_metric": args.select_metric,
            "use_test_ensemble": args.use_test_ensemble,
            "ensemble_group_size": args.ensemble_group_size,
            "seed": args.seed,
            "active_labels": active_labels,
            "feature_dims": {
                "vision": dataset["train"]["vision_dim"],
                "clip": clip_dim,
            },
            "seq_len": dataset["train"]["seq_len"],
        },
        "dataset": {
            "train": int(train_split["y"].shape[0]),
            "valid": int(val_split["y"].shape[0]),
            "test": int(test_split["y"].shape[0]),
            "label_distribution_train": dict(Counter([active_labels[int(x)] for x in train_split["y"].tolist()])),
        },
        "val": evaluate_split(val_true, val_pred, active_labels),
        "test": evaluate_split(test_true, test_pred, active_labels),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"[DONE] wrote {output_path}")


if __name__ == "__main__":
    main()