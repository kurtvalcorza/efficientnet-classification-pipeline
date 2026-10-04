# EfficientNet-B0 Classification E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/efficientnet-classification-pipeline`  
**Notebook:** `tutorials/efficientnet_classification_colab.ipynb`  
**Reviewed commit:** `aa86bdb5a5618f0ff80b3df5abeae658f636c6eb` (`main`, confirmed with `gh api repos/kurtvalcorza/efficientnet-classification-pipeline/commits/main`)  
**Notebook Git blob:** `fbcb57dc13629f378e65825c44a56b853a76cdf1`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-26 (commit `9dce015`); the notebook last changed in `01cc3da`.  
**Finding prefix:** `EFN`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`. The notebook declares 2.1.

## Executive assessment

The default path is well built and reproduces exactly. The notebook carries `data.py` and `pipeline.py` verbatim, asserts the inline manifest against the module identity, stages and re-hashes the pinned `timm/efficientnet_b0.ra_in1k` snapshot, downloads a digest-pinned 400-image CIFAR-10 subset (`frog`, `truck`), measures majority-class, zero-shot ImageNet-mapping and untrained-head baselines before a bounded full fine-tune, evaluates on a held-out and an unseen split, probes blank and noise images before and after adaptation, and exports a SafeTensors adapter that reloads onto a fresh base within a stated tolerance. Its prose on softmax scores, the missing reject option and the small-sample limit is correct.

A direct CPU run of every code cell at the defaults reproduced the Kaggle record exactly:

| Measure | This review (CPU, pins, defaults) | Kaggle T4 record (blob `fbcb57dc`) |
|---|---|---|
| Code cells completed | 14/14 (178 s) | 14/14 on pass 2 (pass 1 stopped at the install guard) |
| Dataset | 400 records, 200 per class, 32×32 px, 100 pixel-identical groups | identical |
| Split | 280 / 60 / 60, train majority `frog` | identical |
| Held-out accuracy: majority / zero-shot / untrained / fine-tuned | 0.500 / 0.9667 / 0.5167 / 1.000 | identical |
| Unseen accuracy, fine-tuned | 1.000 | identical |
| Epoch losses | 0.4823, 0.1611, 0.0719, 0.0346, 0.0166 | identical |
| Adapted head on blank / noise | `truck` 0.509 / `truck` 1.0 | identical |
| Reload check | 60 images, tolerance 1e-4, equivalent | identical |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (EFN-M1).** The recorded run stopped at the install cell's stale-module guard and passed only after a restart; the release record reports it as PASSED.
2. **The held-out and unseen splits contain exact copies of training images, and the notebook tells the learner they cannot (EFN-M2).** The archive holds each `original_images` thumbnail and its `darkened_images` counterpart; 100 of these pairs are pixel-identical. 25 of 60 held-out and 19 of 60 unseen images are exact pixel copies of a training image, while Section 5 says "a random split is valid here because CIFAR-10 images are independent thumbnails". The repository's own records disclose the overlap; the notebook does not.
3. **The documented experiment corrupts the run (EFN-M3).** "Set `FREEZE_BACKBONE = True` and compare" — re-running the cell that holds the field trains a new head on the already fully fine-tuned backbone (reported as "head-only"), and the adapter check in Section 11 then fails with an `AssertionError`.
4. **Guided layer largely absent (EFN-M4).** Declared `GUIDED`, but there is no audience, how-to-use, roadmap, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template, and the 1,151 carried lines are not labelled as infrastructure.

The REL12 BYOD release gate is not recorded on a hosted runtime; this review ran the dataset branch with a compatible 12-image directory (passed through export and reload) and five incompatible inputs locally (§4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.1** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL; what a softmax over classes is; accuracy, balanced accuracy and a confusion matrix; why a baseline is needed before a score means anything" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA T4 documented, CPU also runs |
| Promised outcomes | Pinned install; carried modules; digest-verified snapshot; digest-verified CIFAR-10 subset validated before any model runs; ImageNet top-5 and `sample-sanity` report; zero-shot ImageNet-mapping baseline; blank/noise probes; stratified split; untrained-head floor; bounded fine-tune; held-out comparison by accuracy and balanced accuracy; unseen-split inference; adapter export and verified reload; five `outputs/` files; BYOD image and dataset branches through "the same validate → split → baselines → fine-tune → evaluate → export → reload stages as the sample"; "Try next" experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `0d8c5c6` |
| Release status | `Candidate` (`tutorials/README.md`, `STATUS.md`, `README.md`, `docs/release-verification.md`); REL12 pending |

### Evidence actually obtained

- **Source inspection.** All 31 cells (14 code; cells 5 and 7 are the carried `data.py` and `pipeline.py`). Also read: the generator and template, `data.py` (`read_class_archive`, `validate_dataset`, `split_dataset`), `pipeline.py` (`finetune`, `evaluate`, `save_artifact`, `apply_artifact`, `load_artifact`), `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/`.
- **Documented execution evidence.** `docs/release-verification.md`: Kaggle Tesla T4, 2026-09-26, **the reviewed blob**, clean model cache, image torch 2.10.0+cu128 / numpy 2.0.2; pass 1 `RuntimeError` in the install cell, pass 2 14/14. No Colab run, no BYOD dataset run and no experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), Python 3.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, timm 1.0.29, safetensors 0.8.0, huggingface-hub 0.36.2, numpy 2.5.3, pillow 11.3.0 — the notebook's pins (CPU wheel), taken read-only from another pipeline repository's `.venv`. Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook. The restart behaviour therefore rests on the documented run only.
  - **Clean assets:** empty `HF_HOME` and working directory; cell 9 fetched the snapshot at the pinned revision and `verify_snapshot` passed; cell 11 fetched and verified the archive.
  - **Probes:** P0 static; P1 every code cell at defaults; P2 pixel-digest overlap between the training split and the evaluation splits, and both models re-scored on the copy-free subset; P3 the "Try next" `FREEZE_BACKBONE = True` experiment, re-running cell 19 and the cells after it; P4 the BYOD cell with a compatible class-folder directory, a single image, and five incompatible inputs (path fields, no upload dialog).
- **Not verified:** a Colab run; the Colab upload dialog; the `PER_CLASS` experiment; GPU timing.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path (P1 14/14, identical to the hosted record; snapshot and archive digest-checked; reload equivalence asserted). Defects: the install pattern forces a restart (EFN-M1); `finetune` mutates the pipeline it is called on, so the documented experiment fine-tunes an already fine-tuned model and breaks the reload check (EFN-M3).
- **Scientific validity:** the split ignores the archive's duplicate pairs, so a third of each evaluation split is a copy of a training image (EFN-M2). On the copy-free part the fine-tuned head still scores 1.000 against zero-shot 0.943 (P2), so the headline ordering survives, but the notebook's justification of the split is false and the fine-tuned scores are not held-out measurements as presented.
- **Promise fulfilment:** default promises met. The "Try next" experiment does not deliver a head-only comparison (EFN-M3); BYOD dataset reload is not checked the way the sample's is (EFN-m1).
- **Learner experience:** accurate, candid prose (baselines first, "a difference of one or two images is within noise", no reject option), but the guided layer is missing (EFN-M4) and the experiments presuppose their outcome (EFN-m3).
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2 (EFN-M1); SPL5 (EFN-M2); REL12 BYOD evidence absent from the release record. SHOULD deviations: SPL10 (EFN-M2); GDL10, UX5, VER4 (EFN-M3); GDL1–GDL7, GDL9–GDL14, UX8 (EFN-M4); DAT19/UX10 (EFN-m2).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 14/14 | Section 1 says the cell "stops with a restart instruction" | **Not met** (EFN-M1) |
| Digest-verified pinned snapshot | cell 9 | 3/3 files verified | clear | Met |
| Pinned labelled sample validated before any model | cell 11 | 400 records accepted; finding "100 group(s) of pixel-identical images" | the finding is printed but never discussed | Met; finding unexplained (EFN-M2) |
| Stratified split, valid held-out | cell 13 | 280/60/60; ID-disjointness asserted; 25/60 held-out and 19/60 unseen are pixel copies of training images | "valid here because CIFAR-10 images are independent thumbnails" | **Not met** (EFN-M2) |
| ImageNet top-5 + zero-shot baseline | cell 13 | frog thumbnails → `pick`, `platypus`; zero-shot 0.9667 | grouped-mass rule explained | Met |
| Blank/noise probes | cells 15, 23 | ImageNet head low top scores; adapted head `truck` 1.0 on noise | explained | Met |
| Untrained-head floor, fine-tune, held-out comparison | cells 17–21 | 0.5167 → 1.000; losses printed | "within noise" caveat given; loss framed as optimisation evidence | Met (validity per EFN-M2) |
| Unseen-split inference | cell 23 | 1.000 on 60 | "never used for training" — but 19 are copies | Met, overstated (EFN-M2) |
| Adapter export and verified reload | cell 25 | 360 tensors; 60-image equivalence | "loading succeeding is not the check" | Met |
| "Try next: `FREEZE_BACKBONE = True` and compare … scores and runtime" | cell 19 field | P3: head trained on the fine-tuned backbone; cell 25 `AssertionError`; no runtime printed | none | **Not met** (EFN-M3, EFN-m3) |
| BYOD dataset through the same stages | cell 29 | P4: 12-image directory → validate, split, fine-tune, export, reload ran; reload not compared; no unseen inference | contract stated first | Partly met (EFN-m1) |
| BYOD image | cell 29 | P4: top-5 + `not-measurable` report | stated | Met |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install, read the carried package, stage and verify the snapshot | run cells | printed identity and verified-file count |
| Download, verify and validate the dataset | run cell | manifest printed; duplicate finding not interpreted |
| Read top-5; build a zero-shot baseline; probe blank/noise | run cells, read output | outputs readable; no prediction prompt |
| Split with stratification; fine-tune; compare against baselines; classify unseen | run cells, read table | no checkpoint asks the learner to interpret the table |
| Export, reload and verify the adapter | run cell | equivalence asserted |

The objectives are operations the code performs (GDL5); the only learner-controlled activity is the "Try next" paragraph, which breaks the run as written (EFN-M3).

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | Each stage is introduced and two "What to look for" notes plus "Expect roughly chance" and "Read the loss as optimisation evidence only" set expectations. Missing: audience, how-to-use, roadmap, task contract, glossary, prediction prompts, checkpoints, troubleshooting, conclusion template; carried cells (1,151 lines) unlabelled and uncollapsed (EFN-M4). The duplicate-group finding printed by cell 11 is never explained, and Section 5 asserts the opposite (EFN-M2). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 14/14 after a restart (EFN-M1). Direct: 14/14 at defaults in 178 s with a clean model cache and a fresh archive download; every value in the table above equals the Kaggle record; five outputs written. No Colab run. |
| **Active learning** | Direct (P3) | After the default run, `FREEZE_BACKBONE = True` and a rerun of cell 19 → 21 → 23 → 25: cell 19 reports `trainable_parameters` 2,562 of 4,010,110 and calls it a frozen run, but the stem weights are the fine-tuned ones (not ImageNet); losses start at 0.0054; cell 21 prints fine-tuned 1.000 again; cell 25 raises `AssertionError` because the head-only adapter is reloaded onto the pristine base. The `PER_CLASS` experiment was not run. |
| **Reuse and recovery** | Direct (P4, path fields) + source; Colab upload dialog not verified | Compatible 12-image (6 per class) directory: validated, split 8/4, untrained balanced accuracy 0.50, fine-tuned 0.75, adapter exported and reloaded (10 s). Single image: top-5 and `not-measurable`. Incompatible: one-image class → "every class needs at least 2 images to split; short classes: {'b': 1}"; non-image member → "b\2.png: not a decodable image (…)"; missing directory → `FileNotFoundError … is not a directory`; a text file given to the image branch → raw `UnidentifiedImageError` with no corrective action (EFN-m2). |

## 5. Findings

### Major

#### EFN-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, `_INSTALL_GUARD` lines 50–72 and the install-cell assembly around line 468; Section 1 prose in `tools/notebook_template.py`); `docs/release-verification.md` Recorded executions row and the status lines in `STATUS.md`, `README.md`, `tutorials/README.md`.
- **Observed issue:** the cell `pip install`s seven pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. The opening cell promises that **Run all** in a fresh runtime installs the pins and completes every stage.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle or Colab image hits an error in the first code cell and must restart and run again; RUN1, RUN10 and ENV6 forbid this, and the recorded "PASSED (default path)" rests on a restart-dependent run.
- **Evidence:** documented — Kaggle T4 run of blob `fbcb57dc`, pass 1 stopped with `cuda-bindings 12.9.4 → 13.4.3, numpy 2.0.2 → 2.5.3`, pass 2 14/14 "post-restart". Source — probe P0: `pip_install_in_kernel: true`, `uses_uv: false`, restart instruction present.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart is needed. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Implement it in `tools/build_notebook.py`, regenerate, re-qualify with a one-pass hosted run, and stop reporting a restart-dependent run as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab runtime completes every code cell in one pass with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/efficientnet_classification_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### EFN-M2 — The evaluation splits contain exact copies of training images, and Section 5 says the split is valid because the images are independent

- **Cell/section:** cell 12 (Section 5 prose; `tools/notebook_template.py` line 151), cell 13 (split and ID-only leakage assertion), cell 22 ("never used for training"), cell 11 (duplicate finding printed, not discussed); `data.py` `split_dataset`.
- **Observed issue:** `Cleanlab/cifar-10-subset` stores each thumbnail under `original_images/` and a counterpart under `darkened_images/`; for 100 of the 200 pairs in the sample the two files are pixel-identical (P2). `split_dataset` is a per-record random split, and the leakage assertion compares ids, which differ between the two folders. The notebook says: "A random split is valid here because CIFAR-10 images are independent thumbnails."
- **Consequence:** 25 of 60 held-out and 19 of 60 unseen images are exact copies of a training image, so the fine-tuned row in the comparison table and the "unseen" score are partly memorisation checks. The learner is taught a false reason for a valid split. The headline ordering does survive: on the copy-free subset the fine-tuned head scores 1.000 (35 images, 30 frog / 5 truck) against zero-shot 0.943 — but that subset is no longer balanced and was not what the notebook reported.
- **Evidence:** direct (P2, CPU): `pairs_pixel_identical` 100 of 200; held-out `exact_pixel_copy_in_train` 25/60, same-numbered counterpart in train 46/60; unseen 19/60 and 36/60; copy-free re-scores as above. Documented: `docs/release-verification.md`, `STATUS.md` and `README.md` already record the 46/60 counterpart overlap and that the split is not duplicate-aware.
- **Recommended correction:** split by source image: group each `original_images/<class>/image_N` with its `darkened_images` counterpart (or by pixel digest) and keep each group on one side; add a pixel-digest disjointness assertion beside the id assertion; replace the "independent thumbnails" sentence with an explanation of the pairs and what grouping prevents; have cell 11's prose interpret the duplicate finding. Alternatively read only `original_images/`. Re-record the comparison afterwards.
- **Acceptance check:** after the fix, no held-out or unseen image has a pixel digest equal to any training image (assertion in the notebook passes), no `original_images`/`darkened_images` pair is split across sides, and the Section 5 text no longer claims independence of the archive's images.
- **Spec:** SPL5 (MUST), SPL3, SPL10.

#### EFN-M3 — The documented "head-only" experiment fine-tunes the already fine-tuned model and breaks the reload check

- **Cell/section:** Interpretation "Try next" (`tools/notebook_template.py` line 454); cell 19 (`FREEZE_BACKBONE` field and `adapter.finetune`), cell 25 (reload assertion); `pipeline.py` `finetune` ("mutating this pipeline"), `save_artifact` (omits `frozen_prefixes`).
- **Observed issue:** "Set `FREEZE_BACKBONE = True` and compare the held-out scores and runtime of a head-only fine-tune." The field lives in cell 19, and no rerun scope is given. Re-running cell 19 calls `finetune` on the same `adapter` that the default run already fully fine-tuned, so the "head-only" run starts from the adapted backbone and head. `save_artifact` then writes only the head (the backbone is treated as equal to the base), and `load_artifact` overlays it on the pristine ImageNet backbone, so the reloaded model is a different model.
- **Consequence:** the learner's only documented experiment reports a frozen fine-tune that is not one (held-out 1.000 again, inherited from the full fine-tune) and ends with an unexplained `AssertionError` in Section 11; any conclusion about head-only versus full fine-tuning is wrong.
- **Evidence:** direct (P3, CPU): rerun of cell 19 reported `freeze_backbone: true`, 2,562 of 4,010,110 trainable, first epoch loss 0.0054 (default run epoch 1: 0.4823); `stem_weight_equals_imagenet_base: false`; cell 21 fine-tuned 1.000; cell 25 `AssertionError` at the label/score comparison.
- **Recommended correction:** make the experiment start from a fresh re-headed base: either move the `FREEZE_BACKBONE` field into cell 17 and state "re-run cells 17–27", or create the adapter inside cell 19 (`adapter = EfficientNetPipeline.from_pretrained(..., class_names=SAMPLE_CLASSES, seed=SEED)` before `finetune`). Keep the default run's comparison row so the two policies are shown side by side, and state the rerun range for `PER_CLASS` too (cells 11–27).
- **Acceptance check:** after a default Run all, following the "Try next" instruction exactly yields a frozen run whose backbone weights equal the ImageNet base after training, its own held-out row beside the full fine-tune row, and a passing reload check in cell 25.
- **Spec:** GDL10, UX5, VER4, FT5.

#### EFN-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 3, 5, 7, end of notebook. Generator: `tools/notebook_template.py` and `tools/build_notebook.py` section assembly.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output task contract, no glossary (MBConv, squeeze-and-excitation, softmax mass, zero-shot mapping, balanced accuracy, frozen backbone, BatchNorm statistics, adapter), no prediction before the baselines, the fine-tune or the comparison, no interpretation checkpoint with a sample answer, no troubleshooting (Hub download failure, digest mismatch, CPU time, out-of-memory, BYOD errors), no conclusion template. Cells 3, 5 and 7 (52 + 341 + 810 lines of infrastructure) carry no **Infrastructure** label and no `cellView: form`.
- **Consequence:** a self-paced learner gets accurate prose but little help deciding what matters, predicting what normal output looks like, or stating a conclusion; 1,151 lines of carried code dominate the scroll.
- **Evidence:** source inspection; probe P0 `guided_markers` (how_to_use, roadmap, glossary, audience, predict_prompt, troubleshooting, conclusion_template, infrastructure_label all false; the `checkpoint` match is the word "checkpoint" for model weights, not a learner checkpoint), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC 2.2's guided-mode requirements: audience and how-to-use, roadmap, task contract, glossary, a prediction before Sections 5, 8 and 9, "What to notice" after each principal stage, collapsible checkpoint answers, a Predict → Change one thing → Run → Observe → Explain activity built on EFN-M3's fix, troubleshooting, and a conclusion scaffold; title cells 3, 5, 7 `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL7 and GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3, 5 and 7 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL7, GDL9–GDL14, UX8.

### Minor

#### EFN-m1 — The BYOD dataset branch does not run "the same stages as the sample"

- **Cell/section:** cell 28 prose ("the same validate → split → baselines → fine-tune → evaluate → export → reload stages as the sample"), cell 29 dataset branch (`tools/notebook_template.py` around lines 381–431).
- **Observed issue:** the branch splits two ways (no unseen split), prints only balanced accuracy, performs no new-data inference, and calls `load_artifact` without comparing the reloaded predictions, whereas the sample path checks equivalence within a tolerance. It does not check pixel duplicates across its split either (EFN-M2).
- **Consequence:** a user's adapter is "exported and reloaded" without evidence that the reload reproduces the model, and the user sees less than the sample shows.
- **Evidence:** direct (P4): the 12-image directory ran to "BYOD adapter exported and reloaded" in 10 s with untrained 0.50 / fine-tuned 0.75 balanced accuracy and no comparison; source inspection for the missing stages.
- **Recommended correction:** reuse the sample cells' functions for BYOD: the same three-way split, accuracy plus balanced accuracy and confusion matrix, inference on the unseen split, and the same tolerance check on reload; or narrow the prose to what runs.
- **Acceptance check:** with a compatible BYOD directory, the branch prints the same metric set as Section 9 and a reload-equivalence line with its tolerance, or the prose lists exactly the stages it runs.
- **Spec:** DAT13, DAT14, VER4.

#### EFN-m2 — The BYOD image branch fails on a non-image with a raw exception, and the upload path can pick a stale file

- **Cell/section:** cell 29 image branch (`Image.open(image_path)`; `next(iter(sorted(_upload_into(...).iterdir())))`).
- **Observed issue:** a file that is not an image raises `PIL.UnidentifiedImageError: cannot identify image file '…notes.png'` with no corrective action (the dataset branch wraps the same failure as "not a decodable image"). The upload path takes the first sorted file in `outputs/byod/image/`, which keeps earlier uploads, so a second upload whose name sorts later is ignored.
- **Consequence:** an unhelpful error on the commonest mistake, and silent reuse of an earlier image.
- **Evidence:** direct (P4) for the exception; source inspection for the upload ordering (dialog not run).
- **Recommended correction:** open the image through `_open_image` (or catch `OSError`) and say what file types are accepted; use the files returned by this upload, not the directory listing, or clear the directory first.
- **Acceptance check:** a text file given to the image branch produces a `ValueError` naming the file and the accepted formats; a second upload of a different file is the one classified.
- **Spec:** DAT19, UX10.

#### EFN-m3 — "Try next" presupposes outcomes and asks for a runtime the notebook does not print

- **Cell/section:** Interpretation "Try next" (`tools/notebook_template.py` line 454).
- **Observed issue:** "compare the held-out scores and runtime of a head-only fine-tune" — no cell prints a duration (`finetune` returns none); "lower `PER_CLASS` and watch how quickly the fine-tuned model falls back toward the zero-shot baseline" states the result in advance.
- **Consequence:** the learner cannot make the runtime comparison without instrumenting the code, and is told what to see instead of predicting and checking.
- **Evidence:** source inspection; probe P0 `finetune_returns_elapsed_time: false`.
- **Recommended correction:** print wall time in cell 19 (and record it in the result JSON); phrase the experiments as predictions to test.
- **Acceptance check:** cell 19 prints the fine-tune duration, and the "Try next" text contains no predetermined outcome.
- **Spec:** GDL10, UX5.

### Suggestions

- **EFN-S1 — Show where fine-tune and zero-shot disagree.** The two held-out frogs the zero-shot mapping misses are the whole measured difference; printing them (with their ImageNet top-1) would turn the comparison into an interpretation activity.
- **EFN-S2 — Declare the current spec.** The notebook and docs declare NOTEBOOK_SPEC 2.1; regenerate against 2.2 when the template is revised.
- **EFN-S3 — Name the adapted head's noise answer in the conclusion.** `truck` 1.0 on uniform noise is the strongest illustration of the no-reject-option limit; the Interpretation section could cite it.

## 6. Readiness

**Needs revision.** Open Majors EFN-M1 to EFN-M4; unresolved MUSTs RUN1, RUN10, ENV6, REL2 (EFN-M1) and SPL5 (EFN-M2); REL12 BYOD evidence not recorded on a hosted runtime. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob, a hosted BYOD positive and negative run (REL12), and a re-recorded comparison on a duplicate-aware split.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped):** the default path (14/14, numbers identical to Kaggle), the duplicate overlap and the copy-free re-scores, the freeze-experiment failure, the BYOD positive directory and five refusals.
- **Verified from documented execution:** the install-cell restart on Kaggle T4.
- **Inferred from source:** the upload-dialog behaviour (EFN-m2), GPU behaviour of the experiment, and the effect of the proposed fixes.
- **Most likely to be wrong:** EFN-M2's severity. On the copy-free subset the fine-tuned head still beats zero-shot, so one could argue it is a Minor explanation defect; it is graded Major because the notebook teaches a false justification for the split and violates SPL5.

Probe ZIP: `efficientnet_classification_colab_Review_Probes.zip` (`run_probes.py`, `results.json`, `source_manifest.json`).
