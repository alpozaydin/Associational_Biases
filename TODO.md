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
- [x] ⛔ **Gate (end Wk1):** vision-tower Grad-CAM moved but WRONG direction.
  - `llm_only` Δ: hair −0.032, face −0.013, background +0.044.
  - Escalated to `llm_projector` (also fixed a PEFT regex bug — `multi_modal_projector\..*` was catching GELUActivation; scoped to `linear_[12]`). Δ: hair −0.024, face −0.020, background +0.044.
  - Same failure mode both placements → **reach is not the constraint**. Task-only NLL has no signal that face > background; it just re-routes attention off hair onto whichever spurious cue is cheapest. Fix belongs in the loss (Wk3 consistency), not in placement. No point escalating to `+vision_late`.
  - Proceed with `llm_only` into Wk2/3 (cheaper adapter, same reach). Revisit placement only if consistency-driven training keeps landing wrong.
- [ ] Reproduce parent baseline numbers (their Table 8 success-rate protocol) — the comparison everything is measured against.
- [x] Sanity: token readout — assumption FAILED (all labels share leading space token `28705`); fixed via `label_decision_set` (shared-prefix strip + content-token readout). Cell 1 now passes.
- Reading: parent paper end-to-end (re-derive metrics); Lee et al. survey (arXiv:2309.14381); the two given links (ACM/Springer) — map each to data/training/deployment.

## Week 2 — Redaction + paired-view dataloader
- [x] Concept-dependent redaction transform `mitigation_pipeline/redaction.py`: modes `hair` / `hair+skin` (emotion) / `hair+bg` (activity, Wk8-9). Grayscale-in-place preserves emotion geometry. Verified visually on `test_2298` (fear).
- [x] Paired-view dataloader `mitigation_pipeline/paired_dataset.py` yielding `{x, x_prime, gt_index, img_id}` for RAF. Drops RetinaFace no-detect images at construction.
- [x] Sanity script `mitigation_pipeline/probe_paired.py` saves N side-by-side pairs as PNG for eyeball checks.
- [~] Precompute masks + bboxes for RAF-train stratified subset (200/class = 1400 imgs); bg annotation running. Full train (12k) after Wk3 first training loop confirms the pipeline.
- Reading: LoRA (Hu 2021) + a **2025-26 PEFT-for-VLM** survey; token-level Grad-CAM for LLaVA (2025).

## Week 3 — Consistency + LoRA training
- [x] `train_consistency.py`: `L = L_task(x) + λ·L_consistency(x, x')` on the paired-view dataset. Placement locked `llm_only` per Wk1. Intact side stop-grad on the KL (SimSiam-style asymmetric); `--loss-mode {kl,simsiam,byol}` stubbed for Sec 5.4 ablation.
- [x] First emotion run on RAF-DB (1395-img train subset, 1000 steps, λ=1.0, redact=hair). Result: Δhair=−0.033, Δface=−0.023, Δbg=+0.056. Numerically indistinguishable from Wk1's task-only bite (Δhair=−0.032) → consistency loss added no signal.
- [x] Watch cons-loss: stayed 0.005-0.06 across all 1000 steps, never trended down. **Case C confirmed (attention ≠ decision):** Grad-CAM says hair matters, but KL(f(x) || f(x_hair_masked)) is near-zero → label doesn't depend on hair at the decision-token level.
- [ ] **Escalate to embedding-space consistency (was Sec 5.4 ablation 6, now Wk4 primary).** SimSiam-style: `-cos(predictor(h_x_masked), stop_grad(h_x))` where `h_·` = penultimate LLM hidden state at the decision-token position. Predictor = 2-layer MLP. Rerun same gate. Paper story becomes "label-space fails because attention ≠ decision → embedding-space fixes it" — earns the fix by first documenting the failure.
- [ ] Checkpoint by val accuracy + val consistency loss (needs a held-out val split — TODO wire this up before longer runs).
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

## Weeks 10–11 — Ablations + optional SSL extension
- [ ] mask-only vs paired-consistency · hair-only vs hair+skin · LoRA placement · `λ` sweep · cross-dataset transfer (train RAF→test PHASE and reverse).
- [ ] **Consistency-space ablation** (SSL): KL on label distribution (ours) vs SimSiam-style cosine on the LLM penultimate embedding of the decision-token position vs SimSiam-style on the projector output. Tests whether label-space consistency is enough or whether invariance needs to live deeper. Trainer already carries `--loss-mode {kl,simsiam,byol}` flag — this is a config sweep, not a rewrite.
- [ ] **Anti-collapse regularizer** (SSL): KL only vs KL + VICReg-style variance term. Only run if the `λ` sweep shows val-accuracy dropping as `λ` grows (i.e. task signal traded for cheap consistency). Otherwise cite as considered-and-not-needed.
- [ ] **SSL on the SD3.5↔LLaVA loop** (Roadmap §11 stretch, Wk10-11 if time): SwAV or MoCo on loop-generated unlabeled images with the LoRA-LLaVA describer frozen, updating a second small LoRA slot. Bridges supervised view-consistency (RAF/PHASE) and self-supervised loop-drift correction. Natural feed into the paper's Discussion / Sec 7 generator-side option.

## Week 12 — Writing, figures, final eval.

---

## Open questions for supervisors (Roadmap §10)
1. Co-primary drift reduction: accept the data-contingent decision (isolation experiment sets billing), or require it up front (adds the Sec 7 generator arm, widens scope)?
2. Model base: keep LLaVA-NeXT/Mistral for comparability, or upgrade to a current VLM?
3. Activity scope: confirm generalization-only, deferred to Wk8–9, droppable if emotion-across-two-datasets fills the time.
