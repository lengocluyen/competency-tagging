"""Supervised neural baseline (BERT-style multi-label classifier).

This runner trains a transformer-based text classifier on fragments
and gold competencies, then evaluates on held-out UVs.

It uses HuggingFace Transformers and PyTorch, and the same metrics
module as the main pipeline.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import copy

import data_io
import metrics as metrics_module

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)


@dataclass
class FragmentExample:
    fragment_id: str
    text: str
    labels: np.ndarray  # multi-hot vector


class FragmentDataset(Dataset):
    def __init__(self, examples: List[FragmentExample], tokenizer, max_length: int = 256):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        enc = self.tokenizer(
            ex.text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex.labels, dtype=torch.float32)
        item["fragment_id"] = ex.fragment_id
        return item


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_cv_splits(uv_list: List[str], n_folds: int = 5) -> List[Tuple[List[str], List[str]]]:
    """Same CV logic as main runner, kept separate to avoid import cycles."""
    uv_array = np.array(uv_list)
    np.random.shuffle(uv_array)

    splits: List[Tuple[List[str], List[str]]] = []
    fold_size = len(uv_array) // n_folds

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else len(uv_array)
        test_uvs = uv_array[test_start:test_end].tolist()
        train_uvs = [uv for uv in uv_list if uv not in test_uvs]
        splits.append((train_uvs, test_uvs))

    return splits


def build_label_mapping(competencies: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, int], List[str]]:
    comp_ids = sorted(competencies.keys())
    id2idx = {cid: i for i, cid in enumerate(comp_ids)}
    return id2idx, comp_ids


def build_examples_for_uvs(
    uvs: List[str],
    uv_to_frag: Dict[str, Path],
    uv_to_gold: Dict[str, Path],
    id2idx: Dict[str, int],
) -> Tuple[List[FragmentExample], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """Load fragments & gold for a set of UVs and build training examples."""
    all_examples: List[FragmentExample] = []
    all_fragments: Dict[str, Dict[str, Any]] = {}
    all_gold: Dict[str, Dict[str, Any]] = {}

    for uv in uvs:
        frags = data_io.load_fragments(uv_to_frag[uv])
        gold = data_io.load_gold_fragments(uv_to_gold[uv])

        all_fragments.update(frags)
        all_gold.update(gold)

        for frag_id, frag in frags.items():
            text = frag.get("text", "")
            label_vec = np.zeros(len(id2idx), dtype=np.float32)
            if frag_id in gold:
                for g in gold[frag_id].get("gold", []):
                    cid = g["competency_id"]
                    if cid in id2idx:
                        label_vec[id2idx[cid]] = 1.0
            all_examples.append(FragmentExample(fragment_id=frag_id, text=text, labels=label_vec))

    return all_examples, all_fragments, all_gold


def train_supervised_model(
    train_examples: List[FragmentExample],
    dev_examples: List[FragmentExample],
    dev_gold: Dict[str, Dict[str, Any]],
    model_name: str,
    num_labels: int,
    device: torch.device,
    batch_size: int = 8,
    epochs: int = 1,
    lr: float = 2e-5,
    max_length: int = 256,
    pos_weight: Optional[torch.Tensor] = None,
    early_stopping_patience: int = 2,
    pred_top_k: Optional[int] = 5,
) -> Tuple[AutoModelForSequenceClassification, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )
    model.to(device)
    model.train()

    dataset = FragmentDataset(train_examples, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    if pos_weight is not None:
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    def eval_on_dev() -> float:
        model.eval()
        dev_fragments = {ex.fragment_id: {"text": ex.text} for ex in dev_examples}
        raw = predict_supervised(
            model=model,
            tokenizer=tokenizer,
            device=device,
            fragments=dev_fragments,
            threshold=0.5,
            top_k=pred_top_k,
            max_length=max_length,
        )
        mapped = map_idx_to_comp_ids(raw, comp_ids=[str(i) for i in range(model.num_labels)])
        # Temporarily remap comp_ids indices back to label indices as strings; caller will provide
        # gold with true competency ids at the UV level, so here we only use macro-F1 proxy.
        # We compute macro-F1 over label indices for early stopping signal.
        m = metrics_module.compute_fragment_metrics(mapped, dev_gold)
        return float(m.get("macro_f1", 0.0))

    best_state = None
    best_score = -1.0
    bad_epochs = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(loader, desc=f"Training epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss / max(1, len(loader)):.4f}")

        # Early stopping on dev macro-F1 (proxy over label indices)
        if dev_examples:
            score = eval_on_dev()
            print(f"Dev macro-F1 (proxy): {score:.4f}")
            if score > best_score + 1e-6:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= max(1, early_stopping_patience):
                    print("Early stopping triggered.")
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, tokenizer


def predict_supervised(
    model: AutoModelForSequenceClassification,
    tokenizer,
    device: torch.device,
    fragments: Dict[str, Dict[str, Any]],
    threshold: float = 0.5,
    top_k: Optional[int] = None,
    max_length: int = 256,
) -> List[Dict[str, Any]]:
    model.eval()

    examples: List[FragmentExample] = []
    for frag_id, frag in fragments.items():
        text = frag.get("text", "")
        # Dummy labels, not used at inference
        examples.append(
            FragmentExample(fragment_id=frag_id, text=text, labels=np.zeros(model.num_labels))
        )

    dataset = FragmentDataset(examples, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    all_preds: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Supervised inference"):
            frag_ids = batch.pop("fragment_id")
            labels_dummy = batch.pop("labels")  # not used
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits).cpu().numpy()

            for fid, p in zip(frag_ids, probs):
                preds = []
                if top_k is not None and top_k > 0:
                    # Always emit at least top_k predictions to avoid all-zero outputs
                    # when the label space is large.
                    top_idx = np.argsort(-p)[:top_k]
                    preds = [
                        {"label_idx": int(idx), "confidence": float(p[idx])}
                        for idx in top_idx
                    ]
                else:
                    for idx, score in enumerate(p):
                        if score >= threshold:
                            preds.append({"label_idx": int(idx), "confidence": float(score)})
                all_preds.append({"fragment_id": fid, "predictions": preds})

    return all_preds


def compute_mrr_from_probs(
    probs_by_fragment: Dict[str, np.ndarray],
    gold: Dict[str, Dict[str, Any]],
    comp_ids: List[str],
) -> float:
    rrs: List[float] = []
    for fid, probs in probs_by_fragment.items():
        g = gold.get(fid) or {}
        gold_ids = {str(x.get("competency_id")) for x in (g.get("gold") or []) if x.get("competency_id")}
        if not gold_ids:
            continue

        idx = np.argsort(-probs)
        rr = 0.0
        for rank0, j in enumerate(idx):
            if comp_ids[int(j)] in gold_ids:
                rr = 1.0 / float(rank0 + 1)
                break
        rrs.append(rr)

    return float(np.mean(rrs) if rrs else 0.0)


def map_idx_to_comp_ids(
    fragment_predictions: List[Dict[str, Any]], comp_ids: List[str]
) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for item in fragment_predictions:
        fid = item["fragment_id"]
        preds = [
            {
                "competency_id": comp_ids[p["label_idx"]],
                "confidence": p["confidence"],
            }
            for p in item["predictions"]
        ]
        mapped.append({"fragment_id": fid, "predictions": preds})
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised BERT baseline")
    parser.add_argument("--data_dir", type=Path, required=True, help="Fragment dir")
    parser.add_argument("--gold_dir", type=Path, required=True, help="Gold dir")
    parser.add_argument(
        "--competencies", type=Path, required=True, help="Competencies JSON/JSONL"
    )
    parser.add_argument("--config", type=Path, required=True, help="Config YAML")
    parser.add_argument(
        "--output_dir", type=Path, default=Path("output_supervised"), help="Output dir"
    )

    args = parser.parse_args()
    config = load_config(args.config)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load competencies
    print("Loading competencies...")
    competencies = data_io.load_competencies(args.competencies)
    print(f"Loaded {len(competencies)} competencies")

    id2idx, comp_ids = build_label_mapping(competencies)

    # Discover UVs
    fragment_files = data_io.find_fragment_files(args.data_dir)
    gold_files = data_io.find_gold_files(args.gold_dir)
    uv_to_frag = {data_io.get_uv_from_path(p): p for p in fragment_files}
    uv_to_gold = {data_io.get_uv_from_path(p): p for p in gold_files}

    uvs = sorted(set(uv_to_frag.keys()) & set(uv_to_gold.keys()))
    print(f"Found {len(uvs)} UVs: {uvs}")

    n_folds = config.get("n_folds", 5)
    cv_splits = create_cv_splits(uvs, n_folds)

    model_name = config.get("supervised_model_name", "distilbert-base-uncased")
    batch_size = config.get("supervised_batch_size", 8)
    epochs = int(config.get("supervised_epochs", 1))
    lr = config.get("supervised_lr", 2e-5)
    max_length = config.get("supervised_max_length", 256)
    # In multi-label settings with a large label space, a fixed 0.5 threshold often
    # yields empty predictions. Prefer top-k unless the user explicitly sets it.
    threshold = float(config.get("supervised_threshold", 0.5))
    top_k = config.get("supervised_top_k", 5)
    top_k = int(top_k) if top_k is not None else None
    agg_threshold = float(config.get("supervised_aggregation_threshold", 0.0))
    patience = int(config.get("supervised_early_stopping_patience", 2))
    dev_frac = float(config.get("supervised_dev_fraction", 0.1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_rows: List[Dict[str, Any]] = []

    for fold_idx, (train_uvs, test_uvs) in enumerate(cv_splits):
        print(f"\n=== Fold {fold_idx+1}/{n_folds} ===")
        print("Train UVs:", train_uvs)
        print("Test UVs:", test_uvs)

        # Build training data
        train_examples, _, _ = build_examples_for_uvs(
            train_uvs, uv_to_frag, uv_to_gold, id2idx
        )

        # Split off a small dev set for early stopping
        rng = np.random.RandomState(7 + fold_idx)
        idxs = np.arange(len(train_examples))
        rng.shuffle(idxs)
        dev_n = int(len(idxs) * dev_frac)
        dev_idx = set(idxs[:dev_n].tolist())
        train_ex = [ex for i, ex in enumerate(train_examples) if i not in dev_idx]
        dev_ex = [ex for i, ex in enumerate(train_examples) if i in dev_idx]

        # Build dev gold over label-index space for early stopping signal
        # (we reuse the same examples and treat label indices as strings)
        dev_gold = {}
        for ex in dev_ex:
            gold_list = []
            for j, v in enumerate(ex.labels):
                if float(v) >= 0.5:
                    gold_list.append({"competency_id": str(j)})
            dev_gold[ex.fragment_id] = {"gold": gold_list}

        # Compute pos_weight for BCEWithLogitsLoss
        if train_ex:
            Y = np.stack([ex.labels for ex in train_ex], axis=0)
            pos = Y.sum(axis=0)
            neg = Y.shape[0] - pos
            pos_weight = (neg / np.maximum(pos, 1.0)).astype(np.float32)
            pos_weight_t = torch.tensor(pos_weight, dtype=torch.float32, device=device)
        else:
            pos_weight_t = None

        model, tokenizer = train_supervised_model(
            train_ex,
            dev_ex,
            dev_gold,
            model_name=model_name,
            num_labels=len(comp_ids),
            device=device,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            max_length=max_length,
            pos_weight=pos_weight_t,
            early_stopping_patience=patience,
            pred_top_k=top_k,
        )

        # Evaluate on each test UV
        for uv in test_uvs:
            print(f"\nEvaluating on UV: {uv}")
            fragments = data_io.load_fragments(uv_to_frag[uv])
            gold = data_io.load_gold_fragments(uv_to_gold[uv])
            print(f"Loaded {len(fragments)} fragments, {len(gold)} gold annotations")

            # Precompute probabilities for MRR (full ranking over all labels).
            # We do this separately from predict_supervised() to avoid changing
            # the prediction interface used elsewhere.
            probs_by_frag: Dict[str, np.ndarray] = {}
            model.eval()
            frag_ids = list(fragments.keys())
            texts = [fragments[fid].get("text", "") for fid in frag_ids]
            if texts:
                ds = FragmentDataset(
                    [
                        FragmentExample(fragment_id=fid, text=fragments[fid].get("text", ""), labels=np.zeros(len(comp_ids)))
                        for fid in frag_ids
                    ],
                    tokenizer,
                    max_length=max_length,
                )
                loader = DataLoader(ds, batch_size=16, shuffle=False)
                with torch.no_grad():
                    for batch in tqdm(loader, desc="Supervised probs (MRR)"):
                        batch_frag_ids = batch.pop("fragment_id")
                        _ = batch.pop("labels")
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        logits = outputs.logits
                        probs = torch.sigmoid(logits).cpu().numpy()
                        for fid, p in zip(batch_frag_ids, probs):
                            probs_by_frag[str(fid)] = p
            mrr = compute_mrr_from_probs(probs_by_frag, gold, comp_ids)

            raw_preds = predict_supervised(
                model=model,
                tokenizer=tokenizer,
                device=device,
                fragments=fragments,
                threshold=threshold,
                top_k=top_k,
                max_length=max_length,
            )

            fragment_predictions = map_idx_to_comp_ids(raw_preds, comp_ids)

            # Supervised baseline has no graph reconciliation
            reconcile_stats = [
                {
                    "parent_child_redundancy": 0,
                    "prereq_violations": 0,
                    "capped_by_max_labels": 0,
                }
                for _ in fragment_predictions
            ]

            # Add fragment ids to predictions in expected format
            frag_preds_with_ids: List[Dict[str, Any]] = []
            for item in fragment_predictions:
                frag_preds_with_ids.append(
                    {
                        "fragment_id": item["fragment_id"],
                        "predictions": item["predictions"],
                    }
                )

            from aggregate import ResourceAggregator

            aggregator = ResourceAggregator(
                aggregation_method=config.get("aggregation_method", "max"),
                fragment_type_weights=config.get("fragment_type_weights", {}),
                top_k_per_resource=config.get("top_k_per_resource", 10),
                threshold=agg_threshold,
            )

            resource_predictions = aggregator.aggregate(frag_preds_with_ids, fragments)

            metrics = metrics_module.evaluate_pipeline(
                fragment_predictions=frag_preds_with_ids,
                resource_predictions=resource_predictions,
                gold=gold,
                fragments=fragments,
                reconcile_stats=reconcile_stats,
            )

            row = {"fold": fold_idx, "uv": uv, "mrr": mrr, **metrics}
            all_rows.append(row)
            print(f"Metrics for UV {uv}: {metrics}")

    df = pd.DataFrame(all_rows)
    df.to_csv(args.output_dir / "supervised_results.csv", index=False)
    print("\nSaved supervised baseline results to:", args.output_dir / "supervised_results.csv")


if __name__ == "__main__":
    main()
