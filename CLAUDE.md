# CLAUDE.md — Project context

## Working rules (how Claude should operate here)
- **Ask, don't assume.** If something is unclear, ask before writing a single line. Never make silent assumptions about intent, architecture, or requirements.
- **Simplest solution first.** Always implement the simplest thing that could work. Do not add abstractions or flexibility that weren't explicitly requested.
- **Don't touch unrelated code.** If a file or function is not directly part of the current task, do not modify it, even if you think it could be improved.
- **Flag uncertainty explicitly.** If you are not confident about an approach or technical detail, say so before proceeding. Confidence without certainty causes more damage than admitting a gap.
- **Commit and push after every prompt.** When a request changes files, stage them, commit with a brief message (what + why), and push. Solo research repo → commit directly to `main`. Exclude secrets/scratch.

## What this is
Summer internship project, **AFAR Lab, University of Cambridge**, supervised by **Prof. Hatice Gunes**.
Builds directly on **Dogan et al., _Investigating Associational Biases in Inter-Model Communication of Large Generative Models_** (arXiv:2601.22093) — this repo is that paper's released code (`README.md`, `inter-model communication pipeline/`, `explainability_pipeline/`), which we reuse.

**Full plan: `Roadmap.md` (12 weeks). Live task list: `TODO.md`.** Read both before working.

## Original brief (Project 1 — Alp)
> The project focuses on mitigating **bias drift in inter-model communication pipelines** — systems where multiple models run one after another and pass their outputs to the next model (e.g. a vision model produces descriptions/features that another model uses to predict activity or affect). Because one model's output becomes the next model's input, small demographic biases can build up across steps, so the system starts relying more on demographic cues than on the person's actual activity or affect. To counter this, the project will use a **blurred/redacted training set** (masking faces, hair/skin, and background) and apply **lightweight LoRA adaptation**, so the model learns to stay anchored to stable, task-relevant cues like pose and motion, reducing the chance that demographic information drives or amplifies downstream decisions.
>
> Refs: arXiv:2601.22093 (most relevant — bias mitigation steps), arXiv:2309.14381, dl.acm.org/10.5555/3716662.3716708, link.springer.com/10.1007/s43681-025-00721-9

## What we actually build (locked, per Roadmap)
Harden the **describer arm** (LLaVA-NeXT + Mistral-7B) of an SD3.5 ↔ LLaVA loop against demographic-correlated cues via a **swappable LoRA adapter on a frozen backbone** — the adapter *is* the plug-and-play artifact.

- Method = **paired-view consistency**, not uniform redaction. For each image: intact view `x` + concept-dependent redacted view `x'` (hair masked; facial skin grayscaled/jittered for emotion; hair+background masked for activity). Loss `L = L_task(x) + λ·L_consistency(x, x')`.
- **Emotion first** (RAF-DB, then PHASE). Activity is later generalization.
- RQ-B (does Grad-CAM grounding shift **off hair/skin**?) is the discriminating contribution.
- Loop-drift reduction is *earnable*: the Sec 5.3 isolation experiment decides whether it's co-primary.

## Repo layout
- `inter-model communication pipeline/` — **parent paper code (reuse, don't fork)**: loop runner `main.py`, model init `utils/initialisation.py`, describer `utils/description.py`, eval/drift metrics `eval/` (stumax, cohen, jaccard, krippendorff, chi).
- `explainability_pipeline/` — **parent's** RetinaFace bboxes, MediaPipe hair seg, Grad-CAM notebook (layer-9 probe). Reused for redaction masks + the RQ-B Grad-CAM check.
- `mitigation_pipeline/` — **our new work**:
  - `lora_describer.py` — frozen LLaVA + swappable LoRA, 3 escalating placements (`llm_only` → `+projector` → `+vision_late`).
  - `label_decoding.py` — admissible-set **decision-token** readout (task NLL + consistency KL). See gotcha below.
  - `bite_tune.py` — Wk1 minimal LoRA tune to move the adapter off identity.
  - `gradcam_lora.py` — Wk1 decision gate: does adaptation move layer-9 hair activation? (stock vs adapted).
- `colab_runner.ipynb` — runs on Colab; data + checkpoints live on Google Drive root **`cambridge_bias_mitigation`** (RAF-DB not stored locally).

## Gotchas (read before editing)
1. **No constrained decoding in the parent repo.** `utils/description.py` / `eval/raf_img_eval.py` use free-form `generate()` + regex parsing — there is no logit/decision-token mechanism. Roadmap Sec 4.2's "distribution at the decision token over an admissible set" is **built by us** in `mitigation_pipeline/label_decoding.py`, not reused.
2. **Parent dir name has a space** (`inter-model communication pipeline`) → not importable as a package. `mitigation_pipeline/` therefore mirrors small bits of `initialisation.py` rather than importing it; keep the mirror in sync.
3. **Compute**: LoRA-tune fp16 base is fragile — `bite_tune.py` upcasts LoRA params to fp32 for the spike; real runs should use bf16 / QLoRA. Single 40–80GB GPU assumed.

## Current stage
**Week 1** (LoRA-bite spike + reproduce baseline). Scaffold for the spike exists; not yet run on hardware. Next gate (end Wk1): does describer adaptation move vision-tower Grad-CAM? No → escalate LoRA placement before proceeding.
