# Weight and sample-data provenance and hosting

- Upstream: `timm/efficientnet_b0.ra_in1k`
- Revision: `1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a`, pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest-listed file at it, cross-checked each LFS file against the SHA-256 the Hub records, and wrote the commit and digests into the manifest and `src/efficientnet_classification_pipeline/pipeline.py`.
- Weight format: SafeTensors (`model.safetensors`, 21,355,344 bytes, SHA-256 `d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd`). The Hub repository also holds `pytorch_model.bin` (21,441,905 bytes), a pickle checkpoint with the same weights; it is not in the manifest and is never staged or loaded.
- Manifest: `weights/efficientnet-b0-ra-in1k/dimer-base-manifest.json` (3 files: `README.md`, `config.json`, `model.safetensors`; `totalBytes` 21360641). Every entry carries its SHA-256.
- Committed copy: `config.json` (578 bytes) is committed as the Hub serves it at the pinned commit, so the offline tests can check the architecture, the tag and the preprocessing against timm's built-in configuration. The pin tool replaced it with the bytes downloaded at the pinned commit before hashing (they were byte-identical), so the digests describe the pinned bytes.
- Upstream weight license: Apache-2.0 (the checkpoint's `README.md` front matter and the Hub's licence tag). The model was trained on ImageNet-1k, whose images carry their own terms of access.
- Hosting: Apache-2.0 permits use, modification, distribution and commercial use, subject to keeping the licence and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored).
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files that are absent, at the pinned revision; `verify_snapshot()` then checks every file before any load. `weights/**` is marked `-text` in `.gitattributes`, so Windows `core.autocrlf` cannot rewrite the committed files and break their digests.
- Loader trust boundary: `timm.create_model("efficientnet_b0.ra_in1k", pretrained=False, num_classes=1000)` builds the architecture without contacting the Hub, and `load_state_dict(..., strict=True)` loads the verified SafeTensors file. No pickle is deserialised and no model-repository code runs.

## Tutorial sample data

- Dataset: `Cleanlab/cifar-10-subset`, file `CIFAR-10-subset.zip`, at commit `bb5a7aabf1d14d2d1e3e49d0d8f917bda3622f75`.
- Size and digest: 986,707 bytes, SHA-256 `66f90a4f87d865e8eb653b62f10e754684075a32314177de76832349d4b1fb19`. `fetch_sample_archive` refuses any other bytes and has no fallback.
- Licence: MIT (the dataset card). The images are CIFAR-10 images (Krizhevsky, 2009); the tutorial keeps the `frog` and `truck` folders.
- Hosting: the archive is downloaded at runtime into a working directory (`data/`, git-ignored) and is not redistributed by this repository.
