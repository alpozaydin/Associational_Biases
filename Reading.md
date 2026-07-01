# Reading list — Project 1 (PhD-preview grade)

Structured by priority (Layer 0 → Layer 11). Read Layer 1 first, keep Layer 0
open next to your notebook. Roadmap §6 has a compressed per-week version;
this file is the deeper backing.

Verify every citation yourself before writing it into the paper. Some are
approximate — the field moves faster than any static list. Where an exact
citation is uncertain, a search query is given.

---

## Layer 0 — How to read (before *what* to read)

Three-pass method:
- **Skim (10 min):** abstract, intro's last paragraph, figures, conclusion. Answer: what's the *shape* of the claim.
- **Structured (~1 h):** intro → method setup → main result → one crucial experiment. Skip proofs and related work.
- **Deep (2–3 h):** derive one equation by hand, re-run one experiment mentally with different numbers, write down 3 things the paper doesn't explain. Those are usually your future research questions.

One paper a day for 100 days beats 100 papers in a week — because you also write, re-derive, and argue.

**Tools:** Zotero (references), Obsidian or plain markdown (per-paper 1-page summaries), Semantic Scholar / Papers with Code for citation graphs.

---

## Layer 1 — Non-negotiable direct ancestors

Be able to teach these from memory.

1. **Dogan, Weiss, Patel, Cheong, Gunes** — *Investigating Associational Biases in Inter-Model Communication of Large Generative Models*. arXiv:2601.22093. **Re-derive every metric yourself**: Stuart-Maxwell, Cohen's κ (weighted, unweighted), weighted Jaccard, Krippendorff's α, demographic parity. Reproduce their Table 8 with our code.
2. **Lee et al.** — *Survey of Social Bias in Vision-Language Models*, arXiv:2309.14381. Taxonomy: intrinsic vs extrinsic bias, data-centric vs training-time vs deployment-time mitigation.
3. Supervisor-given links:
   - ACM `10.5555/3716662.3716708`
   - Springer `10.1007/s43681-025-00721-9`
   Map each into the Lee et al. taxonomy.

---

## Layer 2 — Method foundations (LoRA + VLM stack)

- **Hu et al. 2021** — *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. Derive the `ΔW = BA` decomposition; understand why low intrinsic rank is empirically enough.
- **Aghajanyan et al. 2021** — *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. ACL 2021. Theoretical motivation for why LoRA works at all.
- **Dettmers et al.** — *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023.
- **A 2025–2026 PEFT-for-VLM survey** — search Semantic Scholar for `parameter efficient fine-tuning vision-language 2025 survey`. Verify latest before citing.
- **Radford et al. 2021** — *CLIP: Learning Transferable Visual Models From Natural Language Supervision*. ICML 2021. Our vision tower.
- **Liu et al. 2023** — *Visual Instruction Tuning* (**LLaVA**). NeurIPS 2023.
- **Liu et al. 2024** — *LLaVA-NeXT / LLaVA-1.6* (**anyres**). Tech report. **The anyres section is the piece that broke our Wk1 Grad-CAM wrapper — read it carefully.**
- **Merullo et al.** — *Linearly Mapping from Image to Text Space*. ICLR 2023. Explains why LLaVA's multi-modal projector is a small MLP; deep insight into what LoRA on the projector actually shifts.

> ★ Insight — the connection between LoRA rank and the SVD of the true fine-tuning update `ΔW` is rarely spelled out. Derive it yourself: it's a 15-minute exercise that turns "LoRA-r" from a hyperparameter into an intuition — "we're committing to the top-r singular directions of the fine-tuning update".

---

## Layer 3 — Explainability / saliency for VLMs (RQ-B territory)

- **Selvaraju et al. 2017** — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017 (extended IJCV 2020). Derive the gradient-weighted feature map by hand.
- **Chefer et al. 2021** — *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers*. ICCV 2021. VLM-specific saliency: attention rollouts, gradient×attention.
- **Adebayo et al. 2018** — *Sanity Checks for Saliency Maps*. NeurIPS 2018. **Read before you claim RQ-B.** The paper that made everyone stop trusting saliency methods uncritically.
- **Token-level Grad-CAM for LLaVA 2024–2025** — search. Don't fabricate; verify.
- **Kim et al. 2018** — *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors* (**TCAV**). ICML 2018. Sharper form of "is `hair` a direction in activation space" than region-mask Grad-CAM.

---

## Layer 4 — Shortcut learning, spurious correlations, group robustness

- **Geirhos et al. 2020** — *Shortcut Learning in Deep Neural Networks*. Nature Machine Intelligence. The framing paper.
- **Sagawa et al. 2020** — *Distributionally Robust Neural Networks for Group Shifts* (**Group DRO**). ICLR 2020.
- **Liu et al. 2021** — *Just Train Twice* (**JTT**). ICML 2021. Simpler than GroupDRO, no group labels needed.
- **Kirichenko et al. 2023** — *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*. ICLR 2023.
- **Sohoni et al. 2020** — *No Subclass Left Behind*. NeurIPS 2020.

Our paired-view consistency is a *training-time* group-robustness method with a specific inductive bias about where the spurious signal lives (in the redacted region). This literature is your peer-comparison set.

---

## Layer 5 — Counterfactual augmentation / view invariance

- **Maudslay, Gonen, Cotterell, Teufel** — *It's All in the Name: Mitigating Gender Bias with Name-Based Counterfactual Data Substitution*. EMNLP 2019.
- **Dinan et al.** — *Queens are Powerful Too*. EMNLP 2020.
- **Kaushik, Hovy, Lipton** — *Learning the Difference That Makes A Difference With Counterfactually-Augmented Data*. ICLR 2020. Cleanest formal framing.

Our redaction transform is a counterfactual substitution at the *representation* level rather than the *token* level. Cite this line explicitly.

---

## Layer 6 — FER bias / affective computing fairness (AFAR territory)

- **Buolamwini + Gebru 2018** — *Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification*. FAccT 2018. Foundational.
- **Kärkkäinen + Joo 2021** — *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias Measurement and Mitigation*. WACV 2021.
- **Hosseini et al.** — *Faces of Fairness in FER* — verify exact citation before use.
- **Xu et al.** — ECCV Workshops on FER fairness (multiple recent; search `ECCV workshop facial expression fairness 2023 2024`).
- **Dominguez-Catena, Paternain, Galar 2023** — on dataset bias in facial expression recognition. Referenced by the parent paper's Sec 5.2.
- **Torralba + Efros 2011** — *Unbiased Look at Dataset Bias*. Older but sharp reference for the framing.

**Highest signal for your specific lab:** whatever Prof. Gunes' recent papers are. You'll be expected to know your supervisor's own line of work in detail.

---

## Layer 7 — Text-to-image fairness (Wk6-7 + Sec 7 generator arm)

- **Friedrich et al. 2023** — *Fair Diffusion: Instructing Text-to-Image Generation Models on Fairness*. The specific method Roadmap Sec 7 references.
- **Bianchi et al.** — *Easily Accessible Text-to-Image Generation Amplifies Demographic Stereotypes at Large Scale*. FAccT 2023.
- **Cho et al.** — *DALL-Eval*. Fairness eval protocols for T2I.
- **2024–2026 inference-time T2I fairness** — search. Fast-moving.

---

## Layer 8 — Self-supervised learning

Read in this order:

1. **SimCLR** (Chen et al., ICML 2020) — establishes the contrastive template.
2. **MoCo v1 → v2 → v3** (He, Chen et al.). Queue + momentum encoder; why it scales.
3. **BYOL** (Grill et al., NeurIPS 2020) — the "no negatives" pivot. **Then read** the follow-up *BYOL works even without batch statistics* (Richemond et al., 2020) which explains the actual mechanism.
4. **SimSiam** (Chen + He, CVPR 2021) — minimalist BYOL. Ablations are beautiful.
5. **VICReg** (Bardes, Ponce, LeCun, ICLR 2022) — variance-invariance-covariance decomposition. Directly borrowable into our loss.
6. **SwAV** (Caron et al., NeurIPS 2020) — swap prediction; bridges clustering and contrastive.
7. **DINO / DINOv2** (Caron et al., ICCV 2021; Oquab et al., 2023). **Read the attention visualizations** — DINO develops object-localization attention without supervision. Directly relevant to RQ-B.
8. **MAE** (He et al., CVPR 2022) — the masked-reconstruction alternative.

Recent survey: search `self-supervised representation learning 2024 survey`.

> ★ Insight — the BYOL / SimSiam collapse-avoidance story is subtle. Once thought to be about the EMA target network; later shown to be about batch-normalization statistics in the predictor. This kind of "the mechanism you named isn't the mechanism" story is directly relevant to Wk3 — attention isn't decision, and the consistency mechanism we named may not be the one that moves the needle.

> ★ Insight — DINO's attention maps are the aspirational target for RQ-B: self-supervised training makes the model attend to object *shape/silhouette*. That's a mechanistic result of exactly the shape RQ-B is chasing. Include DINO's mechanism section in your related work.

---

## Layer 9 — Fairness math + statistical evaluation

- **Barocas, Hardt, Narayanan** — *Fairness and Machine Learning* (free at fairmlbook.org). Chapters 2-4 for DP / EO / calibration definitions.
- **Hardt, Price, Srebro 2016** — *Equality of Opportunity in Supervised Learning*. NIPS 2016.
- **Pearl — Causality Ch. 3-4.** Counterfactual reasoning. Needed for any *causal* claim.
- **Dietterich 1998** — *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*. McNemar's test etc. Needed for stock-vs-adapter accuracy comparisons.
- **Efron + Tibshirani** — *An Introduction to the Bootstrap*, Ch 1-5. All our region-activation means need CIs.

---

## Layer 10 — Math to sharpen (fastest ROI for a PhD preview)

- **Information theory** — KL divergence, cross-entropy, mutual information, Jensen inequality. MacKay *Information Theory, Inference, and Learning Algorithms* Ch 2-3, or Cover & Thomas Ch 2. Do the exercises. Our whole loss lives here.
- **Linear algebra** — SVD, matrix rank, subspace decomposition, projection matrices. LoRA *is* SVD-restricted; understand what "ΔW lives in a low-rank subspace" means geometrically.
- **Optimization** — SGD, momentum, AdamW, gradient checkpointing. Nocedal & Wright Ch 3-4 for mechanics; Ruder's SGD-variants blog for the ML-side story.
- **Measure-theoretic probability** — 2-hour skim just so notation isn't scary. Grimmett & Stirzaker Ch 1-3.
- **Statistics** — bootstrap, permutation tests, McNemar, multi-comparison correction.

---

## Layer 11 — Doing PhD-style research

- **Cajal** — *Advice for a Young Investigator* (1897). 3-hour read; still lives.
- **John Schulman** — *An Opinionated Guide to ML Research* (blog). Short, high signal.
- **Andrej Karpathy** — *A Recipe for Training Neural Networks* (blog).
- **David Patterson** — *How to Give a Bad Talk* (also his *How to Give a Good Talk*). Standard references.
- **Papers with Code** as a habit — search a method → read the paper → find its ablation table → understand *which* ablation moved which number.

---

## A 4-week reading calendar aligned to Roadmap

- **Wk1-2 (now):** Layer 1 direct ancestors + Hu 2021 LoRA + LLaVA-NeXT anyres. Goal: fluently explain the parent paper's setup on a whiteboard.
- **Wk3-4:** Grad-CAM + Adebayo sanity checks + one VLM-specific saliency paper + shortcut/spurious literature. Goal: defensively explain what RQ-B is *not* claiming.
- **Wk5:** Counterfactual augmentation trio + FER bias trio. Related-work draft due end of Wk5 per roadmap.
- **Wk6-7:** T2I fairness alongside the loop-drift isolation experiment. Goal: know which generator-side interventions are viable if Sec 7 gets promoted.
- **Wk8-9+:** SSL literature — informs the Wk10-11 ablations.

---

## Highest-signal reading for our specific project

If time is short, this is the minimum:

1. Parent paper (Dogan et al. arXiv:2601.22093).
2. LLaVA-NeXT anyres section (verify the paper covers this or a follow-up).
3. Hu 2021 LoRA §2-3.
4. Selvaraju 2017 Grad-CAM §3.
5. Adebayo 2018 sanity checks §4 (their invariance tests).
6. Geirhos 2020 shortcut learning intro + Fig 1.
7. Sagawa 2020 GroupDRO problem statement.
8. SimSiam full paper (short + directly relevant to Sec 5.4 ablation 6).
9. DINO attention-visualisation figures (motivates RQ-B as a real research question).
10. Barocas-Hardt-Narayanan Ch 2 on parity metrics.

Everything else deepens; these ten give you the map.

---

## Reading log (fill in as you go)

Format: `YYYY-MM-DD | Paper | Pass depth | Notes`

- 2026-07-01 | Dogan et al. (parent) | skim | first orientation
