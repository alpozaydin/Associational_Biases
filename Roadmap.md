# Project 1 -- Mitigating Demographic Bias Drift in Inter-Model Communication

Roadmap (12 weeks). Builds directly on Dogan et al., *Investigating Associational Biases in Inter-Model Communication of Large Generative Models* (arXiv:2601.22093), reusing its pipeline, datasets, and evaluation harness.

**Target venue: ICLR 2027** (deadline ~late Sep 2026, *unconfirmed — verify on iclr.cc*); aligns with the 3-month finish.

## 1. Framing and locked decisions

We harden the recognition arm (the image-to-text describer, LLaVA-NeXT/Mistral-7B) of an SD3.5 <-> LLaVA loop against demographic-correlated visual cues, using a swappable LoRA adapter on an otherwise frozen backbone. The adapter *is* the plug-and-play artifact: drop it onto a stock LLaVA, no retraining at deploy.

Locked:
1. Plug-and-play = swappable LoRA adapter on frozen backbone (reconciles "LoRA" with "plug-and-play").
2. Emotion first, on RAF-DB and PHASE. Activity (PHASE sports/caring) is a later generalization section, not core.
3. Primary deliverable is a recognizer robust to demographics. Loop-drift reduction is *earnable*, not asserted: an isolation experiment (Sec 5.3) decides whether it gets promoted to co-primary.

Not yet locked (for supervisors): whether drift reduction must be co-primary up front. If yes, we add a deployment-time generator-side fix (Sec 7), which is the only structurally honest route given the frozen generator.

## 2. Research questions

- RQ-A: Does consistency-based LoRA adaptation reduce the describer's reliance on demographic-correlated cues (hair, facial skin tone) while preserving emotion/activity recognition accuracy?
- RQ-B: Does the intervention change *visual grounding* (Grad-CAM regional activations shift off hair/skin toward concept-relevant regions), or only label behaviour?
- RQ-C: How much of the loop's demographic drift is attributable to the describer arm versus the frozen generator, and how much does the intervention remove?

RQ-B is the discriminating contribution. "The model stops attending to hair" is a stronger, more mechanistic result than a demographic-parity delta, and it separates this work from generic fairness-training papers.

## 3. Why uniform redaction fails, and the actual design

The brief says "mask faces, hair/skin, and background." That collides with the parent paper's own Table 7 / Sec 5.2:
- Face is the primary affect signal. Masking it destroys emotion supervision (RAF-DB is face-only -- nothing left).
- Background carries activity context (fields, courts, equipment); masking removes task signal for the one task where it helps.
- Hair is the clean case: region-separable and spurious for both tasks, yet substantially activated in Grad-CAM. This is the prime suspect.
- Facial skin tone is demographically loaded but lives *on* the face -- inseparable from expression by masking.

So redaction is concept-dependent and cue-dependent, not uniform:

| Region           | Emotion                                                       | Activity                |
| ---------------- | ------------------------------------------------------------- | ----------------------- |
| Hair             | mask                                                          | mask                    |
| Facial skin tone | perturb (grayscale / strong color jitter, geometry preserved) | n/a (face less central) |
| Background       | keep                                                          | mask                    |
| Body             | n/a (RAF-DB) / keep (PHASE)                                   | keep                    |

Masking-only training has a second, deeper flaw: it never teaches invariance when the cue *reappears* sharp at inference. The loop feeds LLaVA crisp SD3.5 faces it never saw masked, so the model can re-exploit the correlation. The fix is a **paired consistency objective**.

## 4. Method

### 4.1 Paired views
For each training image, construct two views:
- Intact view `x`: the original image.
- Transformed view `x'`: concept-dependent demographic removal per the table above (hair masked; facial skin tone grayscaled/jittered for emotion; hair+background masked for activity).

Segmentation reuses the parent repo's stack: RetinaFace for face boxes, MediaPipe for hair masks, PHASE body annotations.

### 4.2 Objective
Use the parent paper's constrained-decoding setup: prompt restricts output to an admissible label set, and we read the distribution at the decision token (their token-selection rule).

- Task term (intact view only, so expression learning is preserved): cross-entropy / NLL toward the ground-truth label token over the admissible set. RAF-DB and PHASE provide GT labels.
- Consistency term: penalize prediction change between views -- KL divergence between the admissible-label distributions at the decision token for `x` and `x'` (MSE on the admissible-token logit is a simpler fallback).

Total loss: `L = L_task(x) + λ · L_consistency(x, x')`. Sweep `λ`; this is the central ablation axis (Sec 6).

This is the parent paper's Sec 7.1 counterfactual substitution, moved to the representation level. It keeps the face for emotion and background for activity, attacks exactly the two cues the Grad-CAM evidence indicts, and trains invariance that survives the cue reappearing at inference.

*Honest limitation to state in the paper:* grayscale/jitter removes skin-tone colour but not face geometry, so ethnicity is only partially neutralized. Full skin-tone removal is impossible without destroying the affect signal. We do not claim to remove ethnicity cues, only skin-tone colour cues.

### 4.3 LoRA placement -- the highest-risk unknown
Grad-CAM in the parent paper is computed on vision-tower features (layer 9, pre-fusion). If we adapt only Mistral and leave the CLIP tower frozen, the visual tokens still encode hair/skin; we may redistribute the LLM's attention over those tokens without changing what they represent. Then RQ-B fails and the mechanistic claim collapses.

Week-1 spike resolves this before committing: adapt describer, re-run their Grad-CAM, check whether hair activation drops. If LLM-only adaptation doesn't move it, extend LoRA to the multimodal projector and late vision-tower layers. Decide placement empirically, not by assumption.

## 5. Experimental setup

### 5.1 Models and data
- Describer: LLaVA-NeXT + Mistral-7B-Instruct (parent paper's choice; keep for comparability).
- Generator: Stable Diffusion 3.5 Large, frozen.

Compute: LoRA-tuning a 7B VLM and running the loop are the feasibility-determining costs. The parent paper ran the loop on A100 HPC and explainability on an L4. Confirm equivalent access (a single 40-80GB GPU for LoRA training; the loop is the heavier job at eval). If only smaller GPUs are available, plan for gradient checkpointing / 4-bit base (QLoRA) and a subsampled loop-eval set -- decide this in week 1, since it bounds everything downstream.
- Datasets: RAF-DB (7 basic emotions, close-up), PHASE (emotions + activities, full-scene). Hold out a demographically-stratified validation split for model selection.
- Baseline (the comparison everything is measured against): frozen stock LLaVA, no adapter, on the parent paper's success-rate protocol (their Table 8). "Accuracy preserved" means within a pre-registered tolerance of this number; "grounding shifted" means relative to this model's Grad-CAM.
- Model selection: primary criterion is validation accuracy plus validation consistency loss (both cheap, computed every checkpoint). Grad-CAM regional activation is the expensive mechanistic check, so compute it only at a few candidate checkpoints, not continuously -- it confirms the choice rather than driving it.

### 5.2 Primary metrics
- Recognition: per-class accuracy / success rate, by gender (parent paper's protocol).
- Fairness: demographic parity gap, logistic-regression odds ratios (their Eq. 8).
- Mechanistic (RQ-B): token-conditioned Grad-CAM regional activations (hair/face/body/background), before vs after adaptation. Success = hair/skin activation drops, concept-relevant region activation rises, accuracy held.

### 5.3 The isolation experiment (decides co-primary billing, RQ-C)
This is the load-bearing experiment and answers the open co-primary question empirically.

Freeze the generator. Run the loop twice on the **identical seed set** -- once with baseline LLaVA, once with LoRA-LLaVA -- and measure drift in the *generated images* (Stuart-Maxwell, Cohen's kappa, weighted Jaccard, DP) for each. Holding the seeds and generator fixed means the only thing that varies is the describer, so the delta is its isolated marginal contribution to drift. (Fix generation seeds/noise where SD3.5 allows, to keep the comparison from absorbing sampling variance.)

- Large, significant delta across both datasets -> promote drift reduction to co-primary, now backed by an isolation design stronger than most fairness papers run.
- Small delta -> "drift is generator-dominated; the describer arm contributes X%, residual is the frozen generator" -> recognizer stays primary, residual is the finding.

*Causal caveat to keep in mind:* the parent paper (Sec 4.2) reports describer text is demographically near-neutral, so the main channel by which the describer can move generated demographics is the *concept label* it picks (e.g. "happy" -> SD3.5's female default). The experiment tests exactly that channel. Expect the effect to be modest and possibly sign-uncertain; that is information, not failure.

### 5.4 Ablations (the paper's backbone)
1. Mask-only vs paired-consistency (does consistency matter?).
2. Hair-only vs hair+skin perturbation (which cue carries the effect?).
3. LoRA placement: LLM-only vs +projector vs +late-vision (RQ-B mechanism).
4. `λ` sweep (accuracy/fairness trade-off curve).
5. Cross-dataset transfer: train RAF-DB, test PHASE and vice versa.
6. **Consistency space** (SSL-motivated): `KL on label distribution` (ours) vs `SimSiam-style cosine on the LLM penultimate embedding of the decision-token position` vs `SimSiam-style on the multi-modal projector output`. Tests whether label-space consistency is enough or whether invariance needs to live deeper in the network. Directly borrows from the SSL literature (BYOL/SimSiam) applied to a supervised, view-consistency setting.
7. **Anti-collapse regularizer** (SSL-motivated): `KL only` vs `KL + VICReg-style variance term`. If a `λ` sweep shows val-accuracy dropping as `λ` grows (i.e. the model is trading task signal for cheap consistency), the variance term is the mechanistic fix; if not, we cite it as considered-and-not-needed.

## 6. Reading track

This runs in parallel, not after. Roughly 3-5 papers/week, heavier in weeks 1-3.

Foundations and direct ancestors (mostly in the parent paper's bibliography, verifiable there):
1. Parent paper, end to end. Re-derive their metrics yourself.
2. Lee et al., *Survey of Social Bias in Vision-Language Models* (arXiv:2309.14381) -- one of your given links; taxonomy of intrinsic/extrinsic and mitigation families.
3. The two given links (ACM 5555/3716662.3716708 and Springer s43681-025-00721-9) -- read for positioning; map each to data-centric / training-time / deployment-time so you can place our method precisely.

Method components:
4. LoRA (Hu et al., 2021) and a current PEFT survey. *Search for 2025-2026 PEFT-for-VLM surveys -- this area moves fast, don't rely on a 2021 mental model.*
5. Grad-CAM (Selvaraju et al., 2017) and token-conditioned / VLM-adapted saliency. *Search "token-level Grad-CAM LLaVA 2025" for current adaptations.*
6. Counterfactual data augmentation / substitution (Maudslay et al.; Dinan et al.) -- cited in their Sec 7.1; our consistency loss is the representation-level version.
7. Shortcut learning / spurious correlations: Geirhos et al. (shortcut learning), and group-robustness methods (GroupDRO, JTT) as the conceptual neighbours of consistency training. Position us against these.

Domain and SOTA:
8. FER bias: Hosseini et al. (*Faces of Fairness*), Xu et al. (ECCV-W), Dominguez-Catena et al. -- all cited; establishes the female/happiness and skin-tone priors we target.
9. T2I debiasing for the generator-side option: Fair Diffusion (Friedrich et al., 2023), control-token conditioning. *Search "text-to-image fairness inference-time 2025-2026" for newer plug-and-play generator fixes -- relevant if co-primary forces Sec 7.*
10. *Current describer SOTA: search whether LLaVA-NeXT is still the sensible base or whether a 2025-2026 open VLM (Qwen-VL, InternVL successors) is the stronger comparability/upgrade choice. I am not confident the parent paper's model is still SOTA at time of reading -- verify before locking.*

Deliverable from the reading track: a 2-3 page related-work draft by week 5 positioning us in the data/training/deployment taxonomy, so the paper's framing is settled before results land.

## 7. Generator-side option (conditional, week 8+ stretch)

Only if supervisors require drift reduction as guaranteed co-primary. Add a deployment-time, fine-tuning-free generator fix (Fair Diffusion or demographic control-token prompting, their Sec 7.3 -- adapter-free, stays plug-and-play). Makes "we reduce loop drift" structurally defensible because it addresses the arm that actually injects demographics. Cost: a generator-side condition on every loop eval and scope past emotion-first. Not core.

## 8. Timeline

| Wk    | Experiment                                                                                                              | Reading   |
| ----- | ----------------------------------------------------------------------------------------------------------------------- | --------- |
| 1     | LoRA-bite spike: adapt describer, re-run Grad-CAM, check hair activation moves. Reproduce parent baseline numbers.      | Items 1-3 |
| 2     | Redaction/perturbation pipeline (reuse seg code). Build paired-view dataloader.                                         | Items 4-5 |
| 3     | Implement consistency+LoRA training; first emotion run (RAF-DB).                                                        | Items 6-7 |
| 4-5   | Train emotion on both datasets; get accuracy + Grad-CAM moving right. Related-work draft.                               | Item 8    |
| 6-7   | Plug into full SD3.5<->LLaVA loop; isolation experiment (Sec 5.3); drift metrics + residual. Decide co-primary billing. | Item 9    |
| 8-9   | Activity generalization (PHASE sports/caring). Optional generator-side arm if co-primary required.                      | Item 10   |
| 10-11 | Ablations (Sec 5.4).                                                                                                    | --        |
| 12    | Writing, figures, final eval.                                                                                           | --        |

The eval harness already exists in the parent repo, which de-risks weeks 6-7 substantially.

## 9. Decision gates and risks

- Gate (end wk 1): does describer adaptation move vision-tower Grad-CAM? No -> extend LoRA to projector/vision before proceeding. This is the single highest-risk assumption.
- Gate (end wk 7): isolation-experiment delta -> sets co-primary vs secondary billing.
- Risk: emotion accuracy degrades under aggressive skin-tone perturbation. Mitigation: `λ` sweep + selection criterion balances it; report the trade-off curve as a result, not a failure.
- Risk: consistency on near-neutral describer text yields tiny loop-drift effect. Mitigation: this is RQ-C's finding either way; framing absorbs it.

## 10. Open questions for supervisors

1. Co-primary drift reduction: accept the data-contingent decision (isolation experiment sets billing), or require it up front (then we add the Sec 7 generator arm and widen scope)?
2. Model base: keep LLaVA-NeXT/Mistral for comparability, or upgrade to a current VLM if reading shows the field has moved?
3. Activity scope: confirm it is generalization-only, deferred to wk 8-9, and may be dropped if emotion-across-two-datasets fills the time.

## 11. Extensions (post-primary result)

**SSL on the SD3.5<->LLaVA loop (Wk10-11 if time).** Once we plug into the full loop (Wk6-7), the generator produces an infinite stream of unlabeled images. This is the SSL sweet spot -- no labels, lots of data, at exactly the distribution we care about (loop-generated). Concretely: with the LoRA-LLaVA describer frozen (adapter loaded), run **SwAV** or **MoCo** using paired-view augmentations (intact + concept-redacted) on the loop's generated images, updating a second, small LoRA slot. This would be a fully self-supervised drift-correction pass -- adapter-free at deploy, still plug-and-play -- and a natural bridge into Sec 7's generator-side discussion. Positions the paper as bridging supervised view-consistency (RAF/PHASE) and self-supervised loop-drift correction. Concrete cost: an extra LoRA head, no data collection.