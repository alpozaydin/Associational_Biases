# Investigating Associational Biases in Inter-Model Communication of Large Generative Models

## Overview
This is the implementation accompanying the Journal of Artificial Intelligence Research  submission [Investigating Associational Biases in Inter-Model Communication of Large Generative Models](https://arxiv.org/abs/2601.22093). 
### Abstract
Social bias in generative AI can manifest not only as performance disparities but also as “associational bias”, whereby models learn and reproduce stereotypical associations between concepts and demographic groups, even in the absence of explicit demographic information (e.g., associating doctors with men). These associations can persist, propagate, and potentially amplify across repeated exchanges in inter-model communication pipelines, where one generative model’s output becomes another’s input. This concern is especially salient for human-centred perception tasks, such as human activity recognition and affect prediction, where inferences about behaviour and internal states can lead to errors or stereotypical associations that propagate into unequal treatment in sensitive deployments (e.g., wellbeing assessment or safety monitoring). In this work, we focus on concepts related to human activity and affective expression, and study how such associations evolve within an inter-model communication pipeline that alternates between image generation and image description. Using the RAF-DB and PHASE datasets, we quantify demographic distribution drift induced by model-to-model information exchange and assess whether these drifts are systematic using an explainability pipeline. Our results reveal demographic drifts toward younger representations for both actions and emotions, as well as toward more female-presenting representations, primarily for emotions. We further find evidence that some predictions are supported by spurious visual regions (e.g., background or hair) rather than concept-relevant cues (e.g., body or face). We also examine whether these demographic drifts translate into measurable differences in downstream behaviour, i.e., while predicting activity and emotion labels. Finally, we outline mitigation strategies spanning data-centric, training-time, and deployment-time (post-training) interventions, and emphasise the need for careful safeguards when deploying interconnected models in human-centred AI systems.
## Table of Contents

## Explainability Pipeline
```txt
.
├── gradcam_pipeline
├── hair_face_filtering
├── hair_segmentation
├── random_image_selection
├── retina_face
```

## Inter-model communication Pipeline

```txt
.
├── eval                        
│   ├── graphs                  # all the graphs for the paper
│   ├── cat_side_by_side.py     # RAF graphs (side by side)
│   ├── chi.py                  # calculate chi square statistic
│   ├── cosine_graphs.py        # plot cosine similarity graphs
│   ├── iters_img_eval.py       # text-based iteration graphs
│   ├── jaccard.py              # jaccard similarity metric
│   ├── krippendorff.py         # annotator agreement metric
│   ├── phase_img_eval.py       # script for llava to annotate phase images
│   ├── phase_sbs.py            # phase graphs
│   ├── raf_img_eval.py         # script for llava to annotate raf images
│   └── similarities.py         # cosine similarity computations
├── raf_utils                   # various util functions
│   ├── agg_raf_metadata.py     # aggregate raf metadata
│   ├── raf_agg.json            # aggregated original raf results
│   ├── raf_test_agg.json       # aggregated original raf results (test set)
│   └── raf_train_agg.json      # aggregated original raf results (train set)
├── utils                       # various util functions
│   ├── annotate.py             # manual annotation ui
│   ├── batch_commit.py         # helper script to commit huge numbers of files in batches
│   ├── bias_categories.py      # cleans and aggregates raf data
│   ├── coco.py                 # coco utils 
│   ├── data.py                 # for reading from configs files (used for text-based)
│   ├── description.py          # function for llava describing an image
│   ├── fix_iters.py            # script to rename some files
│   ├── hook.py                 # pre-commit hook
│   ├── initialisation.py       # initialising models
│   ├── json2csv.py             # converts some json data to csv
│   ├── merge_agg_results.py    # merge multiple aggregated results
│   ├── phase.py                # phase utils
│   ├── sample_for_anno.py      # randomly sample images for annotation
│   └── transform_raf_results.py# reformat old format of results to current format
├── image_gen.py                # generate images
├── main.py                     # main runner (for raf)
├── phase.py                    # main phase script 
├── poetry.lock                 # poetry stuff
├── pyproject.toml              # poetry stuff
└── README.md                   # README
```
## Citation

```
@misc{Dogan2026Investigating,  
  author        = {F.I. {Dogan} and Y. {Weiss} and K. {Patel} and J. {Cheong} and H. {Gunes}},  
  title         = {{Investigating Associational Biases in Inter-Model Communication of Large Generative Models}},   
  year          = {2026},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CY}
 }
```

## Acknowledgements

**Open Access:** The authors have applied a Creative Commons Attribution (CC BY) licence to any Author Accepted Manuscript version arising.
**Data Access Statement:** This study involved secondary analyses of existing datasets. All datasets are described and cited accordingly. 
**Funding:** The work of F. I. Dogan and H. Gunes were supported in part by CHANSE and NORFACE through the MICRO project, funded by ESRC/UKRI (grant ref. UKRI572).
**Contributions** Conceptualisation: FID, HG. Methodology: FID, YW, KP. Data curation: YW.  Investigation: FID, YW, KP. Software: YW, KP. Formal analysis \& Visualisation: FID, YW, KP. Resources: FID, JC, HG. Writing – original draft: FID, YW, KP, JC. Writing – review \& editing: FID, JC, HG. Supervision: FID, HG. Project administration: FID, HG. Funding acquisition: HG.
