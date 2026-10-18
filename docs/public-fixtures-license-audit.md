# Public fixture license audit

Date: 2026-10-18
Scope: which public receipt/invoice datasets may be vendored into
`fixtures/public/` and used to measure InvoiceLens.

## Why this document exists

The synthetic corpus (`fixtures/cjk_synthetic`) has known ground truth, but
it was drawn by this project. A measurement that only ever sees pages we
generated cannot catch a bias we do not know we have. Public datasets are
the independent half of the measurement — and "public" is not the same as
"licensed for what we are about to do with it". This project is MIT and
commercially aimed, so a dataset that says *research only* is out, and a
dataset with no license statement at all is out until one appears.

Method: the official source for each dataset was read directly (repository
README and LICENSE file, HuggingFace metadata, the challenge portal, the
paper). Where two independent places stated the license, that is recorded.
Inference is never used as a substitute for a statement.

## CORD v2 — INCLUDED

| Check | Source | Finding |
|---|---|---|
| Repository license file | github.com/clovaai/cord → `LICENSE-CC-BY` | Full CC BY 4.0 text, verbatim |
| HuggingFace metadata | huggingface.co/datasets/naver-clova-ix/cord-v2 | `license: Creative Commons Attribution 4.0 International License` |
| dataset_infos.json | same dataset, raw file | `"license": "Creative Commons Attribution 4.0 International License"` |

CC BY 4.0 permits commercial use and redistribution with attribution, so the
obligation is attribution, discharged by `fixtures/public/cord-v2/
ATTRIBUTION.md` (regenerated with the retrieval date on every fetch).

Two caveats recorded at fetch time, both from the upstream README:

- Only 1,000 images are published, not the 11,000 the paper describes:
  "Some class labels shown in the original paper were removed due to
  Indonesian legal issues. In particular, the `store_info`,
  `payment_info`, and `etc` fields have been removed from the target class
  to be published."
- The receipts are Indonesian (some English), which makes them the
  dot-as-thousands locale: `60.000` means 60,000. The eval harness passes
  the matching locale hint (`thousands="."`) for language `id`.

Only the 100-page **test** split is vendored — the split whose annotations
are meant to be held out, so measuring on it is measuring on unseen pages.

## SROIE — EXCLUDED (no license found)

Checked: the ICDAR 2019 challenge portal (rrc.cvc.uab.es, ch=13) and the
solution repository (github.com/zzzDavid/ICDAR-2019-SROIE).

- The repository carries an MIT LICENSE, but it covers the *solution code*
  ("Copyright (c) 2019 Niansong Zhang, Songjie Yang, Shengjie Xiu …
  the Software"), not the dataset.
- The challenge portal describes the dataset ("To address the potential
  privacy issue, some sensitive fields … of the receipts are blurred") and
  states no license or terms of use for the data itself.

Verdict: no license evidence for the dataset → not vendored. The privacy
blurring note is encouraging but is not a grant. What would change this: a
license or terms statement on the challenge portal, or an explicit
statement from the organizers.

## WildReceipt — EXCLUDED (source of truth not locatable)

Checked: GitHub repository search, the MMOCR documentation and repository
layout, the Ikomia dataset-loader plugin, and the originating paper.

- The paper is arXiv 2103.14470 ("Spatial Dual-Modality Graph Reasoning for
  Key Information Extraction", Sun et al., 2021) and says "Our code and
  dataset will be publicly released" — but states no license.
- The dataset circulates through the OpenMMLab mirror
  (`download.openmmlab.com/mmocr/data/wildreceipt.tar`); the MMOCR
  documentation pages that would state terms were not reachable at the
  documented URLs on 2026-10-18 (the docs tree has been restructured).
- The only GitHub repository found (`Ikomia-hub/dataset_wildreceipt`) is a
  third-party format converter; its LICENSE covers the plugin, not the
  data.

Verdict: the authoritative source could not be located, and a mirror with
no stated terms is not a basis for vendoring → not vendored. What would
change this: the original repository (or a license statement in the paper
or the mirror).

## Standing rule

A dataset enters `fixtures/public/` only with a license statement from an
authoritative source, recorded here with the URL it came from. "Widely
used" and "appears in many papers" are not license evidence. If a dataset
is excluded, the exclusion and the reason stay in this file — the next
person to want it should not have to re-derive the answer.
