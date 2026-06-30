"""Admissible-set decision-token readout for the describer.

The parent repo (``utils/description.py`` / ``eval/raf_img_eval.py``) does
free-form ``generate()`` + regex parsing — there is **no** logit-level decision
mechanism. Roadmap Sec 4.2 needs one: a constrained read of the distribution
over an admissible label set at the *decision token*. This module builds it and
is shared by:

  - ``bite_tune.py``  -- task loss = NLL toward the GT label token (intact view).
  - ``train_consistency.py`` (Wk3) -- consistency loss = KL between the
    admissible-label distributions of the intact view ``x`` and redacted ``x'``.

Tokenizer reality (Mistral/SentencePiece, verified on Colab): a label encoded
with a leading space tokenizes as ``[28705, <content>, ...]`` where ``28705`` is
the standalone space (``▁``) piece — shared by *every* label, so the first
sub-token can't separate classes, and the model emits that space token *first*
at the decision position. So the discriminative token is the first one **after**
the shared prefix.

:func:`label_decision_set` returns ``(prefix_ids, label_ids)``: ``prefix_ids`` is
the shared non-discriminative prefix we feed to the model (simulating it having
emitted the leading space); ``label_ids`` is the first content token per label,
read at the next position. ``prefix_ids`` is empty for tokenizers that don't
split the leading space — the readout then happens at ``logits[:, -1, :]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Mirror of ``utils/__init__.py``::RAF_DB_EMOTIONS (1-indexed there). Duplicated
# because the parent package dir name has a space and is not importable.
RAF_DB_EMOTIONS = [
    "Surprise",
    "Fear",
    "Disgust",
    "Happiness",
    "Sadness",
    "Anger",
    "Neutral",
]
RAF_EMOTION_LABELS = [e.lower() for e in RAF_DB_EMOTIONS]

# Default instruction: restrict the describer to a single admissible label. Kept
# close to the Grad-CAM notebook's emotion prompt so token behaviour matches.
_INSTRUCTION = (
    "Focusing only on the expression and emotion of the person shown, answer "
    "with exactly one word, the single best emotion from this list: [{labels}]."
)


def build_label_prompt(processor, labels=RAF_EMOTION_LABELS, instruction=_INSTRUCTION) -> str:
    """Templated chat prompt that constrains the answer to ``labels``."""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction.format(labels=", ".join(labels))},
                {"type": "image"},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def _longest_common_prefix(seqs: list[list[int]]) -> list[int]:
    prefix = []
    for toks in zip(*seqs):
        if len(set(toks)) == 1:
            prefix.append(toks[0])
        else:
            break
    return prefix


def label_decision_set(processor, labels=RAF_EMOTION_LABELS):
    """Return ``(prefix_ids, label_ids)`` for constrained decision-token readout.

    Labels are encoded with a leading space (matching how an answer token is
    emitted mid-sequence). The shared prefix (e.g. the standalone space piece) is
    stripped off; the next token per label is the discriminative class id.

    :raises ValueError: if the first content tokens still collide (then labels
        aren't first-token separable — switch to full-string scoring).
    """
    tok = processor.tokenizer
    encs = [tok.encode(f" {l}", add_special_tokens=False) for l in labels]
    if any(len(e) == 0 for e in encs):
        raise ValueError(f"a label produced no tokens: {list(zip(labels, encs))}")

    prefix = _longest_common_prefix(encs)
    if any(len(e) <= len(prefix) for e in encs):
        raise ValueError(f"a label is fully consumed by the shared prefix: {encs}")

    label_ids = [e[len(prefix)] for e in encs]
    if len(set(label_ids)) != len(label_ids):
        raise ValueError(
            f"non-unique content tokens for labels {labels}: {label_ids}. "
            "Labels aren't first-token separable — score full strings instead."
        )
    return prefix, label_ids


def _append_prefix(inputs, prefix_ids):
    """Append ``prefix_ids`` to ``input_ids`` (+ attention_mask) so the next-token
    logits at position -1 predict the discriminative content token.

    Returns a shallow copy; the original ``inputs`` is untouched. ``pixel_values``
    / ``image_sizes`` pass through — the appended tokens sit after the image.
    """
    if not prefix_ids:
        return inputs
    inputs = dict(inputs)
    ii = inputs["input_ids"]
    pre = torch.tensor(prefix_ids, device=ii.device, dtype=ii.dtype)
    pre = pre.unsqueeze(0).expand(ii.shape[0], -1)
    inputs["input_ids"] = torch.cat([ii, pre], dim=1)
    if "attention_mask" in inputs and inputs["attention_mask"] is not None:
        am = inputs["attention_mask"]
        ones = torch.ones((am.shape[0], len(prefix_ids)), device=am.device, dtype=am.dtype)
        inputs["attention_mask"] = torch.cat([am, ones], dim=1)
    return inputs


def decision_logits(model, inputs, prefix_ids) -> torch.Tensor:
    """Forward pass; next-token logits at the decision position [B, vocab].

    Feeds the shared prefix first, then reads ``logits[:, -1, :]``. Keeps the
    graph (no ``torch.no_grad``) so callers can backprop.
    """
    out = model(**_append_prefix(inputs, prefix_ids))
    return out.logits[:, -1, :]


def label_distribution(logits: torch.Tensor, label_ids: list[int]) -> torch.Tensor:
    """Renormalised log-probs over the admissible label set only [B, n_labels].

    Softmax is restricted to ``label_ids`` (constrained decoding), so mass on
    out-of-set tokens is discarded rather than treated as a competing class.
    """
    idx = torch.as_tensor(label_ids, device=logits.device)
    return F.log_softmax(logits.index_select(-1, idx), dim=-1)


def task_nll(model, inputs, prefix_ids, label_ids: list[int], gt_index: int) -> torch.Tensor:
    """NLL of the ground-truth label at the decision token (scalar, batch-mean)."""
    log_probs = label_distribution(decision_logits(model, inputs, prefix_ids), label_ids)
    target = torch.full((log_probs.shape[0],), gt_index, device=log_probs.device)
    return F.nll_loss(log_probs, target)


def consistency_kl(log_p_intact: torch.Tensor, log_p_view: torch.Tensor) -> torch.Tensor:
    """KL(intact || view) over the admissible set (Roadmap Sec 4.2 consistency term).

    Both args are log-probs from :func:`label_distribution`. Intact view is the
    reference (its grounding is what we want the redacted view to match).
    """
    return F.kl_div(log_p_view, log_p_intact, reduction="batchmean", log_target=True)
