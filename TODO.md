# TODO — Mitigating Demographic Bias Drift (Project 1)

Live task list. Plan lives in `Roadmap.md`; context in `CLAUDE.md`. Check items as done.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · ⛔ decision gate

---

## Week 1 — LoRA-bite spike + reproduce baseline
- [x] `lora_describer.py` — frozen LLaVA + swappable LoRA, 3 placement presets.
- [x] `label_decoding.py` — admissible-set decision-token readout (task NLL + consistency KL).
- [x] `bite_tune.py` — minimal task-only LoRA tune to move adapter off identity.
- [x] `gradcam_lora.py` — Grad-CAM on LoRA describer, stock-vs-adapted region delta.
- [x] `make_annotations.py` — RetinaFace → rename → hair seg → merge (fills repo glue gap).
- [ ] **Run** the spike on GPU (Colab A100 / Drive `cambridge_bias_mitigation`; ignore the v6 TPU — stack is CUDA-only):
  - [ ] pick a RAF probe subset (~300 imgs, stratified by emotion+gender; reuse `explainability_pipeline/random_image_selection/raf_image_selection.py`).
  - [ ] `make_annotations.py` → `annotations.json` + `hair_masks/`.
  - [ ] `gradcam_lora.py` **stock** (no `--adapter`) → baseline hair/face/bg activation.
  - [ ] `bite_tune.py --placement llm_only` (~300 steps, RAF subset; fp16 fits A100 80GB, no QLoRA).
  - [ ] `gradcam_lora.py --adapter <out> --compare` → read hair-activation delta.
- [ ] ⛔ **Gate (end Wk1):** does `llm_only` move layer-9 hair activation?
  - drops → proceed with `llm_only`.
  - no change → escalate to `llm_projector`, then `llm_projector_vision_late`; re-run gate.
- [ ] Reproduce parent baseline numbers (their Table 8 success-rate protocol) — the comparison everything is measured against.
- [ ] Sanity: verify `label_token_ids` leading-space assumption holds for the Mistral tokenizer (distinct first sub-tokens per RAF emotion).
- Reading: parent paper end-to-end (re-derive metrics); Lee et al. survey (arXiv:2309.14381); the two given links (ACM/Springer) — map each to data/training/deployment.

## Week 2 — Redaction + paired-view dataloader
- [ ] Concept-dependent redaction transform producing `x'` from `x` (Roadmap §3 table): hair→mask; facial skin→grayscale/jitter (emotion); hair+background→mask (activity). Reuse hair seg + RetinaFace.
- [ ] Paired-view dataloader yielding `(x, x', gt_label)` for RAF (and PHASE later).
- [ ] Precompute/cache masks (hair `.npy`, face bbox) for the train split.
- Reading: LoRA (Hu 2021) + a **2025-26 PEFT-for-VLM** survey; token-level Grad-CAM for LLaVA (2025).

## Week 3 — Consistency + LoRA training
- [ ] `train_consistency.py`: `L = L_task(x) + λ·L_consistency(x, x')` using `label_decoding.consistency_kl`.
- [ ] First emotion run on RAF-DB; checkpoint by val accuracy + val consistency loss.
- Reading: counterfactual data augmentation (Maudslay, Dinan); shortcut learning (Geirhos), GroupDRO/JTT.

## Weeks 4–5 — Train both datasets + related-work draft
- [ ] Train emotion on RAF-DB and PHASE; get accuracy + Grad-CAM moving the right way.
- [ ] Grad-CAM regional check at candidate checkpoints (confirm, don't drive, selection).
- [ ] **Related-work draft (2–3 pp)** placing us in data/training/deployment taxonomy.
- Reading: FER bias (Hosseini, Xu, Dominguez-Catena).

## Weeks 6–7 — Loop + isolation experiment ⛔
- [ ] Plug LoRA-LLaVA into full SD3.5 ↔ LLaVA loop.
- [ ] Sec 5.3 isolation: same seed set, baseline vs LoRA describer, frozen generator; drift metrics (Stuart-Maxwell, Cohen's κ, weighted Jaccard, DP).
- [ ] ⛔ **Gate (end Wk7):** drift-delta size → co-primary vs secondary billing for drift reduction.
- Reading: T2I fairness inference-time (2025-26) — relevant only if co-primary forces the Sec 7 generator arm.

## Weeks 8–9 — Activity generalization (+ optional generator arm)
- [ ] PHASE sports/caring activity recognition.
- [ ] Optional: deployment-time generator-side fix (Fair Diffusion / control-token) **only if** co-primary required.
- Reading: current describer SOTA — is LLaVA-NeXT still the right base vs Qwen-VL / InternVL successors?

## Weeks 10–11 — Ablations
- [ ] mask-only vs paired-consistency · hair-only vs hair+skin · LoRA placement · `λ` sweep · cross-dataset transfer (train RAF→test PHASE and reverse).

## Week 12 — Writing, figures, final eval.

---

## Open questions for supervisors (Roadmap §10)
1. Co-primary drift reduction: accept the data-contingent decision (isolation experiment sets billing), or require it up front (adds the Sec 7 generator arm, widens scope)?
2. Model base: keep LLaVA-NeXT/Mistral for comparability, or upgrade to a current VLM?
3. Activity scope: confirm generalization-only, deferred to Wk8–9, droppable if emotion-across-two-datasets fills the time.
