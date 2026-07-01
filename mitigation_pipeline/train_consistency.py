"""Wk3 paired-view consistency trainer -- the actual method.

Loss:
    L = L_task(x) + lambda_c * L_consistency(x, x')

    L_task = NLL of GT emotion at the decision token, on the intact view x.
        (Same as bite_tune.py's Wk1 loss; preserves task supervision so the
        model can't collapse to a constant prediction.)
    L_consistency = KL(f_intact || f_view) on the admissible-set decision-token
        distribution -- our label-space instance of a paired-view invariance
        objective. Intact view is the reference (stop-grad); gradient flows
        through the redacted-view side so it moves to match the intact view.

This is where Wk1's diagnosis ("task-only NLL has no signal that face > bg")
gets its fix: consistency loss makes any hair-driven prediction *change under
redaction* a penalty, so the model is forced to lean on cues that survive
x -> x' (face geometry for emotion).

Placement locked to ``llm_only`` (Wk1 gate: reach is sufficient; direction was
the issue). ``--loss-mode`` is stubbed for the Sec 5.4 SSL ablations (SimSiam-
style cosine on penultimate LLM embeddings; not implemented in this pass).

Usage (Wk3 first run, on the 1395-image annotated train subset):
    python train_consistency.py \\
        --data-dir /path/RAF_DB/DATASET/train \\
        --annotations /path/train_subset/anno/annotations.json \\
        --hair-dir   /path/train_subset/anno/hair_masks \\
        --out adapters/wk3_kl_lambda1 \\
        --steps 1000 --lambda-c 1.0
"""

from __future__ import annotations

import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from label_decoding import (
    _append_prefix,
    build_label_prompt,
    consistency_kl,
    decision_logits,
    label_decision_set,
    label_distribution,
)
from lora_describer import DEVICE, build_lora_describer, save_adapter
from paired_dataset import RAFPairedDataset


def _upcast_trainable(model) -> None:
    """LoRA params -> fp32 for a stable optimiser step (base stays fp16)."""
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def _forward_dist(processor, model, image, prompt, prefix_ids, label_ids):
    """One forward pass -> log-probs over the admissible label set. Keeps graph."""
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    logits = decision_logits(model, inputs, prefix_ids)
    return label_distribution(logits, label_ids)


def _forward_h_logits(processor, model, image, prompt, prefix_ids):
    """Forward -> (hidden state at decision-token position, logits at same position).

    Used by the ``simsiam`` embedding-space loss: we need both the LLM's
    penultimate representation (for the consistency term) and the vocab logits
    (for the task NLL, which stays a label-space objective).
    ``output_hidden_states=True`` returns the LLM's per-layer stack; we take
    the last layer's activation at the decision position.
    """
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(DEVICE)
    inputs = _append_prefix(inputs, prefix_ids)
    out = model(**inputs, output_hidden_states=True)
    h = out.hidden_states[-1][:, -1, :]
    logits = out.logits[:, -1, :]
    return h, logits


class HiddenPredictor(nn.Module):
    """SimSiam-style predictor MLP.

    Two Linear layers with LayerNorm + GELU between them. LayerNorm (not the
    original SimSiam BatchNorm) because we run micro-batch of 1 per step, so
    BN statistics would be degenerate.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _resolve_hidden_size(model) -> int:
    """LLM hidden-state width for LlavaNext. Robust to peft wrapper + config layout."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    cfg = base.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    return cfg.hidden_size


def train(
    data_dir: str,
    annotations: str,
    hair_dir: str,
    out: str,
    placement: str,
    steps: int,
    lr: float,
    subset: int,
    lambda_c: float,
    seed: int,
    log_every: int,
    redact_mode: str,
    loss_mode: str,
) -> None:
    processor, model = build_lora_describer(placement)
    _upcast_trainable(model)
    model.train()

    prefix_ids, label_ids = label_decision_set(processor)
    prompt = build_label_prompt(processor)
    label_ids_t = torch.as_tensor(label_ids, device=DEVICE)

    # simsiam mode adds a trainable predictor MLP; kept in fp32 for stable
    # cosine gradients even when the backbone runs in fp16 autocast.
    predictor = None
    if loss_mode == "simsiam":
        hidden_size = _resolve_hidden_size(model)
        predictor = HiddenPredictor(hidden_size).to(DEVICE).float()
        predictor.train()
        print(f"predictor (simsiam) hidden_size={hidden_size}, "
              f"params={sum(p.numel() for p in predictor.parameters()):,}")

    ds = RAFPairedDataset(data_dir, annotations, hair_dir, mode=redact_mode)
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    if subset:
        idxs = idxs[:subset]
    print(
        f"{len(idxs)} paired samples, placement={placement}, steps={steps}, "
        f"lambda={lambda_c}, redact={redact_mode}, loss={loss_mode}, device={DEVICE}"
    )

    params = [p for p in model.parameters() if p.requires_grad]
    if predictor is not None:
        params += list(predictor.parameters())
    optim = torch.optim.AdamW(params, lr=lr)

    step = 0
    running_task = 0.0
    running_cons = 0.0
    running_loss = 0.0
    while step < steps:
        rng.shuffle(idxs)
        for i in idxs:
            if step >= steps:
                break
            item = ds[i]
            with torch.autocast(device_type=DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
                if loss_mode == "kl":
                    logp_x = _forward_dist(processor, model, item["x"], prompt, prefix_ids, label_ids)
                    logp_xp = _forward_dist(processor, model, item["x_prime"], prompt, prefix_ids, label_ids)
                    target = torch.tensor([item["gt_index"]], device=logp_x.device)
                    loss_task = F.nll_loss(logp_x, target)
                    # Intact side stop-grad; gradient flows through x' only.
                    loss_cons = consistency_kl(logp_x.detach(), logp_xp)
                elif loss_mode == "simsiam":
                    h_x, logits_x = _forward_h_logits(processor, model, item["x"], prompt, prefix_ids)
                    h_xp, _ = _forward_h_logits(processor, model, item["x_prime"], prompt, prefix_ids)
                    # Task NLL from vocab logits, restricted to admissible label tokens.
                    logp_x = F.log_softmax(logits_x.index_select(-1, label_ids_t), dim=-1)
                    target = torch.tensor([item["gt_index"]], device=logp_x.device)
                    loss_task = F.nll_loss(logp_x, target)
                    # Embedding-space consistency: predict intact hidden from x'
                    # side; stop-grad on the intact target so gradient flows
                    # through predictor and through the x'-side model only.
                    p_xp = predictor(h_xp.float())
                    z_x = h_x.float().detach()
                    loss_cons = -F.cosine_similarity(p_xp, z_x, dim=-1).mean()
                else:
                    raise NotImplementedError(
                        f"loss_mode={loss_mode!r} is a Sec 5.4 ablation stub; "
                        "not implemented yet."
                    )

                loss = loss_task + lambda_c * loss_cons

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            running_task += loss_task.item()
            running_cons += loss_cons.item()
            running_loss += loss.item()
            step += 1
            if step % log_every == 0:
                n = float(log_every)
                print(
                    f"step {step}/{steps}  "
                    f"loss={running_loss / n:.4f}  "
                    f"task={running_task / n:.4f}  "
                    f"cons={running_cons / n:.4f}"
                )
                running_task = running_cons = running_loss = 0.0

    save_adapter(model, out)
    print(f"adapter saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="RAF split root ({1..7}/*.jpg)")
    ap.add_argument("--annotations", required=True, help="annotations.json from make_annotations.py")
    ap.add_argument("--hair-dir", required=True, help="dir of hair-mask .npy files")
    ap.add_argument("--out", required=True, help="adapter output dir")
    ap.add_argument(
        "--placement",
        default="llm_only",
        choices=["llm_only", "llm_projector", "llm_projector_vision_late"],
        help="LoRA target scope (Wk1 gate locked llm_only)",
    )
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--subset", type=int, default=0, help="0 = use all annotated samples")
    ap.add_argument("--lambda-c", type=float, default=1.0, help="consistency weight")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument(
        "--redact-mode",
        default="hair",
        choices=["hair", "hair+skin", "hair+bg"],
        help="which x' to produce for paired training (Roadmap Sec 3)",
    )
    ap.add_argument(
        "--loss-mode",
        default="kl",
        choices=["kl", "simsiam", "byol"],
        help="Wk3=kl (roadmap Sec 4.2). simsiam/byol are Sec 5.4 ablation stubs.",
    )
    args = ap.parse_args()
    train(**vars(args))
