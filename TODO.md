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
- [x] **Wk4 escalate to embedding-space consistency (was Sec 5.4 ablation 6, promoted to primary).** SimSiam-style `-cos(predictor(h_x'), stop_grad(h_x))` on LLM last-layer hidden state at the decision-token position. Predictor = 2-layer Linear→LayerNorm→GELU→Linear MLP (kept fp32, backbone fp16). Result on 1395-img train, 1000 steps, λ=1.0, redact=hair, llm_only: Δhair=**−0.051**, Δface=**+0.001**, Δbg=+0.051. cons drove −0.67 → −0.97 (Wk3 KL stayed 0.005-0.06 → simsiam has 15× more signal). **Partial win:** hair drop 60% bigger than KL, face bleeding stopped (+0.001 vs Wk3 −0.023), but bg still absorbs the shift. Diagnosis: hair-only x' leaves bg as a free-lunch invariance-safe cue.
- [x] **Wk4b: simsiam with `--redact-mode hair+bg`, same config as Wk4.** Result: Δhair=−0.032, Δface=**−0.020**, Δbg=+0.052. **Regression vs Wk4** — face bleeding returned, hair drop shrank to Wk3/task-only level. Hypothesis: bigger perturbation triggers predictor-shortcut collapse (33M-param MLP + task-NLL label signal can hallucinate intact hidden from label + face features → backbone off the hook for face routing) and/or image-level "redacted vs intact" shortcut (large dark region distinguishes x' from x, LoRA picks up bg-color statistics on intact instead). **Current best remains Wk4 hair-only simsiam** (Δhair=−0.051, Δface=+0.001).
- [x] **Wk4c pre-work — probe acc + confusion sweep (advisor-mandated):** Ground-truth signal for adaptation quality. Results on 280-img probe:
  - Stock **62.14%** — 109/280 defaults to "surprise" (decoder-fallback pathology); surprise=1.00 is fake.
  - Wk4 hair simsiam **62.86%** — same fallback pattern shifted to "happy" (109/280); neutral=0.00 (class-conditional collapse).
  - Wk4b hair+bg simsiam **73.93%** (+11.8 pp = 3-SE gain) — **breaks the fallback**, meaningful column spread, worst confusions are semantically adjacent (anger↔disgust). Real classifier.
  - Task-only λ=0 **47.50%** (-14.6 pp) — worse overfit fallback (133→happy). **Kills the "supervised FT alone lifts acc" hypothesis**: FT without consistency is destructive on 1395-img subset.
  - **Hypothesis pending seed=1 confirmation**: simsiam consistency may act as a REGULARIZER, not just an invariance loss. Hair-only redaction (weak) recovers stock; hair+bg (strong) may prevent overfit + break decoder-fallback pathology. Wk4b CAM "regression" (Δface=-0.020) could reflect diffuse vision attention in a better classifier, not routing to bg content. Three of four configs collapsed to a fallback attractor (stock→surprise 109, Wk4 hair→happy 109, task-only→happy 133); Wk4b is the only one that didn't. Story rests on one seed.
  - **Prediction to log** (regularizer framing): Wk4b should also eventually collapse to a fallback under much longer training (e.g. 3000+ steps), just later than task-only. Not to run now, but recording so we're not surprised if that happens later.
- [x] **Wk4c step 1: Wk4b with `--seed 1`.** Result: acc **72.86%** (seed=0: 73.93%, Δ=1.1 pp within 1 SE). Confusion matrix confirms no-fallback pattern (all 7 columns non-empty: 26/35/68/36/27/28/60). Per-class balance actually TIGHTER at seed=1 (spread 0.28 vs 0.45). CAM: Δhair=-0.043, Δface=-0.003, Δbg=+0.046 (seed=0 was -0.032/-0.020/+0.052) — same qualitative pattern, face nearly flat. **Mechanism confirmed across 2 seeds.** Hypothesis promoted from "pending seed=1" to "supported": simsiam consistency + hair+bg redaction acts as regularizer that breaks decoder-fallback pathology on this 1395-img subset.
- [x] **Wk4c step 2: demographic-slice acc (FairFace gender+age on RetinaFace crops).** 280 probe imgs. Race axis skipped (`dima806/fairface_race_image_detection` 404). First look (all-GT) suggested Female-favored gap closure (seed=0 Female +19.0 pp vs Male +6.7 pp), but advisor caught two confounds:
  1. **Compositional confound**: stock's 109/280 surprise-fallback (pred_col=109 vs expected=40) inflates slices holding true-surprise images. Stock Male/young all-GT 0.714 was a fallback artifact — non-surprise-GT drops it to 0.590. The apparent "Male/young ceiling" evaporates under the filter. Reframe dropped.
  2. **Own-fallback asymmetry** (mild): Wk4b seed=1 confusion shows disgust +28, neutral +20 excess. Wk4b seed=0 shows happiness +17. Both are 2.5-4× smaller than stock's surprise +69, so non-surprise-GT filter fairly controls the dominant skew.
- [x] **Wk4c step 3: seeds 2+3 for differential stability (n=4 total).** Advisor call after n=2 gave gender-differential seed-swing 13.6 pp = 3× the mean +4.6 pp — could not claim demographic-gap-closing. n=4 verdict:
  - Global non-surprise-GT acc: stock **0.5583** → Wk4b seed-avg **0.7854** = **+22.7 pp** (all 4 seeds agree, range 0.746-0.813).
  - Per-seed non-surprise: seed=0 0.7750, seed=1 0.7458, seed=2 0.8083, seed=3 0.8125 (spread 6.7 pp).
  - Gender-differential (Δfemale-Δmale) per seed: **+0.114, -0.022, -0.030, -0.020**. Mean +0.010, sd 0.069, se 0.034, 95% CI [-0.057, +0.078] — straddles zero.
  - Seed=0's +11.4 pp Female-favored gap was outlier seed variance, not signal.
  - Non-surprise-GT per slice (n=4 avg): Female +0.233, Male +0.223, adult +0.238, young +0.228, Female/adult +0.236, Female/young +0.250, Male/adult +0.239, Male/young +0.212 — **uniform lift across all non-tiny slices**. Senior slices (n=12, 13) too small to interpret.
  - **Final story** (n=4 supports): SimSiam consistency + hair+bg redaction (a) prevents FT-alone catastrophic overfit (λ=0 → 47.5%), (b) breaks stock's decoder-fallback pathology (surprise +69 excess → ≤+28 in worst Wk4b seed), (c) uniformly lifts non-surprise-GT decoding by ~+23 pp across all sizable demographic slices. **Not** a demographic-gap-closing intervention: n=4 gender differential 95% CI centered on zero.
- [ ] **SKIP** `llm_projector` under simsiam+hair+bg (per advisor: headroom question, wrong priority — placement escalation only makes sense if seed variance confirms the current mechanism *and* demographic slice shows a gap to close). n=4 confirms mechanism (floor-lift) but no gap-closure signal → placement escalation remains unmotivated.
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
