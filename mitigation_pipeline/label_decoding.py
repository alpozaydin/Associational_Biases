"""Admissible-set decision-token readout for the describer.

The parent repo (``utils/description.py`` / ``eval/raf_img_eval.py``) does
free-form ``generate()`` + regex parsing — there is **no** logit-level decision
mechanism. Roadmap Sec 4.2 needs one: a constrained read of the distribution
over an admissible label set at the *decision token*. This module builds it and
is shared by:

  - ``bite_tune.py``  -- task loss = NLL toward the GT label token (intact view).
  - ``train_consistency.py`` (Wk3) -- consistency loss = KL between the
    admissible-label distributions of the intact view ``x`` and redacted ``x'``.

The "decision token" is the position whose next-token logits pick the label.
With a generation-prompt-terminated input (``add_generation_prompt=True``), that
is the final input position: ``logits[:, -1, :]`` predicts the first answer
token. This matches the parent's "first admissible token" selection rule used
in the Grad-CAM notebook.

Tokenizer caveat (Mistral/SentencePiece): a label's first sub-token differs with
a leading space ("happiness" vs " happiness"). Mid-sequence the model emits the
space-prefixed variant, so we key on that by default — see ``label_token_ids``.
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


def label_token_ids(processor, labels=RAF_EMOTION_LABELS, *, leading_space=True) -> list[int]:
    """First sub-token id per label — the class set the decision token chooses over.

    :param leading_space: encode " happiness" not "happiness". True matches how a
        mid-sequence answer token is emitted (SentencePiece prefix), which is what
        the decision position predicts. Verify once for your processor.
    """
    tok = processor.tokenizer
    ids = []
    for label in labels:
        text = f" {label}" if leading_space else label
        sub = tok.encode(text, add_special_tokens=False)
        if not sub:
            raise ValueError(f"label {label!r} produced no tokens")
        ids.append(sub[0])
    if len(set(ids)) != len(ids):
        # First sub-tokens collide -> distribution can't separate those labels.
        raise ValueError(
            f"non-unique first sub-tokens for labels {labels}: {ids}. "
            "Pick labels with distinct leading tokens or score full strings."
        )
    return ids


def decision_logits(model, inputs) -> torch.Tensor:
    """Forward pass; return next-token logits at the decision position [B, vocab].

    Keeps the graph (no ``torch.no_grad``) so callers can backprop. ``inputs`` is
    the processor output already moved to the model device.
    """
    out = model(**inputs)
    return out.logits[:, -1, :]


def label_distribution(logits: torch.Tensor, ids: list[int]) -> torch.Tensor:
    """Renormalised log-probs over the admissible label set only [B, n_labels].

    Softmax is restricted to ``ids`` (constrained decoding), so probability mass
    on out-of-set tokens is discarded rather than treated as a competing class.
    """
    idx = torch.as_tensor(ids, device=logits.device)
    return F.log_softmax(logits.index_select(-1, idx), dim=-1)


def task_nll(model, inputs, ids: list[int], gt_index: int) -> torch.Tensor:
    """NLL of the ground-truth label at the decision token (scalar, batch-mean)."""
    log_probs = label_distribution(decision_logits(model, inputs), ids)
    target = torch.full((log_probs.shape[0],), gt_index, device=log_probs.device)
    return F.nll_loss(log_probs, target)


def consistency_kl(log_p_intact: torch.Tensor, log_p_view: torch.Tensor) -> torch.Tensor:
    """KL(intact || view) over the admissible set (Roadmap Sec 4.2 consistency term).

    Both args are log-probs from :func:`label_distribution`. Intact view is the
    reference (its grounding is what we want the redacted view to match).
    """
    return F.kl_div(log_p_view, log_p_intact, reduction="batchmean", log_target=True)
