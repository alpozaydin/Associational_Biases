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
- [x] **Run** the spike on GPU (amax5 A100 80GB, env `assoc-bias`; not Colab):
  - [x] pick a RAF probe subset (280 imgs, 40/class from RAF test, seed=0 via `make_probe.py`).
  - [x] `make_annotations.py` → 280 annotations + hair masks (fixed to force TF-CPU to avoid cuDNN clash with torch).
  - [x] `gradcam_lora.py` **stock** → baseline `hair=0.294 face=0.341 background=0.366` (n=279).
  - [x] `bite_tune.py --placement llm_only` 300 steps on 350-img RAF-train subset; nll 1.39 → 0.82; adapter saved.
  - [x] `gradcam_lora.py --adapter <out> --compare` → adapted `hair=0.262 face=0.328 background=0.410` (n=277).
  - Fix in flight: `TokenLogitWrapper` had to preserve full pixel_values (5D anyres) + image_sizes so transformers>=4.48's strict token↔feature check passes; view 0 alone gets the grad.
- [x] ⛔ **Gate (end Wk1):** hair delta = **−0.032 (PASS)** — vision-tower Grad-CAM moved under `llm_only`. Half-good: hair dropped but shifted to *background*, not face. Direction is job of Wk3 consistency loss, not this bite spike. Proceed with `llm_only` for Wk2/3; escalate only if consistency-driven training keeps landing on background.
- [ ] Reproduce parent baseline numbers (their Table 8 success-rate protocol) — the comparison everything is measured against.
- [x] Sanity: token readout — assumption FAILED (all labels share leading space token `28705`); fixed via `label_decision_set` (shared-prefix strip + content-token readout). Cell 1 now passes.
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
