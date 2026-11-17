# UrbanTwin Sim2Real: final LUMPI and V2X submissions

**Team:** dipper13579  
**Author:** KuanLin Chen — Department of Industrial and Systems Engineering, Chung Yuan Christian University (中原大學工業與系統工程學系).

**Paper:** “Fixing the Twin, Not the Detector: Sim-to-Real Roadside LiDAR Detection from Digital-Twin Data,” IEEE BigData 2026 Cup. Paper link: **TBD**.

This repository reconstructs the two selected submissions from synthetic-trained OpenPCDet members. It contains code, recipes and verification evidence; datasets, real-derived maps, prediction caches and weights are external.

| Track | Submission | Combined | Detection mAP | Result source |
|---|---:|---:|---:|---|
| LUMPI | 1052645 / WINc3 | 0.5541 | 0.2047 | First place; Codabench submission 1052645 |
| V2X-Real | 1054488 / G77 H3 | 0.8102 | 0.5013 | Codabench submission 1054488 |

These are historical leaderboard results, not scores recomputed by the local dummy-reference checker. `VALIDATION.md` records exact cached replay and format checks. `MEMBERS.md` maps every checkpoint to its recipe and parent; `RELEASE_ASSETS.md` supplies SHA-256 values for separately distributed artifacts.

## Environment and installation

Run the commands below from the root of this release checkout unless a command changes directory.

Measured execution: Ubuntu 22.04 under WSL2; Python 3.10.22; PyTorch 2.11.0+cu128 / CUDA 12.8; spconv-cu126 2.3.8; NumPy 1.26.4; RTX 5060 Laptop GPU (8 GB). The installed spconv CUDA suffix differs from torch's CUDA suffix; this records the environment actually used. `provenance/environment.json` and `requirements.txt` record dependencies. The complete install was not re-executed on a clean OS during release verification.

```bash
conda create -n urbantwin_release python=3.10
conda activate urbantwin_release
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
git clone https://github.com/open-mmlab/OpenPCDet.git OpenPCDet
git -C OpenPCDet checkout 233f849829b6ac19afb8af8837a0246890908755
export PCDET_ROOT="$PWD/OpenPCDet"
# Choose private storage locations. Do not commit them or their contents.
mkdir -p ../urbantwin_work
export WORK_ROOT="$(realpath ../urbantwin_work)"
export RAW_ROOT="$WORK_ROOT/raw"
export KIT_ROOT="$WORK_ROOT/starting_kit"
mkdir -p "$WORK_ROOT"
source scripts/env.sh
bash scripts/install_overlay.sh
# A CUDA compiler/toolkit matching the local build is required for extensions.
export TORCH_CUDA_ARCH_LIST=12.0
export CUDA_CACHE_MAXSIZE=4294967296
python -m pip install SharedArray torch-scatter
python -m pip install --no-build-isolation -e "$PCDET_ROOT"
```

If extension wheels are unavailable for the selected torch/CUDA combination, build from source with the activated CUDA toolkit; the install commands are a recipe, not a claim of a tested clean install. The overlay retains upstream Apache-2.0 notices. `scripts/install_overlay.sh` checks the upstream commit before copying the modified files.

## Obtain inputs separately

- Official challenge and starter-kit links: https://urbanity-lab.github.io/urbantwin-sim2real-challenge/ . Extract each official kit to KIT_ROOT/lumpi and KIT_ROOT/v2x, including scoring_program, local_eval.py, forbidden_frames.txt, detection_test_frames and detection_test_frame_ids.json.
- UrbanTwin synthetic downloads: UT-LUMPI https://doi.org/10.7910/DVN/D9SSWD and UT-V2X https://doi.org/10.7910/DVN/N6N4UR . Download/extract the pcd and lbl trees to RAW_ROOT/UT-LUMPI/extracted/UCF-DT-LUMPI-Seq-1 and RAW_ROOT/UT-V2X-Real-IC/extracted/UCF-DT-V2X-Real-Seq-1. The locally stored Dataverse metadata declares **CC0 1.0** for both; see the versioned snapshot fields in `provenance/dataset_licenses.json`. The live metadata endpoint was blocked during preparation, so review the current landing-page terms when downloading. This repository's Apache license does not replace dataset terms.
- LUMPI: https://data.uni-hannover.de/dataset/lumpi . Obtain Measurement 4/6 Label.csv from its official links and place them at WORK_ROOT/data/lumpi_real/M4_Label.csv and M6_Label.csv. `scripts/prepare_real_lumpi.sh` fetches selected allowed point frames into RAW_ROOT/LUMPI-real and constructs local calibration sets.
- V2X-Real: https://github.com/ucla-mobility/V2X-Real . Obtain val.zip through its documented download and place it at RAW_ROOT/V2X-Real/val.zip. `tools/guard_v2x.py` binds this archive and the forbidden list to the prior canonical-identity audit; changed inputs require a new audit before use.
- Checkpoints: arrange the individually verified assets at WORK_ROOT / each `checkpoint` path in `configs/members.json`. Use the filenames and SHA-256 values in `RELEASE_ASSETS.md` to verify separately distributed checkpoints.

Optional UrbanTwin range downloader (or download through the website if access is restricted):

```bash
python tools/fetch_ut.py fetch --doi 10.7910/DVN/D9SSWD --folders lbl,pcd --out "$RAW_ROOT/UT-LUMPI/extracted"
python tools/fetch_ut.py fetch --doi 10.7910/DVN/N6N4UR --folders lbl,pcd --out "$RAW_ROOT/UT-V2X-Real-IC/extracted"
```

## Synthetic preparation, training and inference

```bash
source scripts/env.sh
for dataset in lumpi lumpi_v2 lumpi_v3b v2x v2x_v2 v2x_v3 v2x_dense; do
  python tools/prepare_training.py "$dataset"
done
python tools/train_member.py --dry-run
# Example root member; then follow parent dependencies in MEMBERS.md.
python tools/train_member.py --member lumpi/A_cp20
```

Preparation writes train/val infos, synthetic point files and GT sampling databases under WORK_ROOT. Synthetic indices below 9000 train; 9000–9999 validate. Training outputs follow OpenPCDet's output/config/extra_tag/ckpt layout. Copy the checkpoint at the selected saved epoch in `configs/members.json` to its listed WORK_ROOT path before training descendants or running inference. Set `--member track/name` separately for each member. The PR1_R14 wrapper installs the procedural rider hook. Only synthetic points/boxes enter training; `COMPLIANCE.md` describes real-derived statistics and inference calibration. Retraining is stochastic; exact historical weights are supplied through the external asset inventory.

```bash
python tools/infer_members.py lumpi --list-only
python tools/infer_members.py v2x --list-only
python tools/infer_members.py lumpi
python tools/infer_members.py v2x
```

These generate the fixed member/TTA caches in the paths expected by the replay scripts. The V2X defaults use 0/x/y/xy flips; truck and pedestrian also use rotations +5/-5/+10/-10 degrees. HC pedestrian runs use top 400 per source. LUMPI uses the U, h and h2 limits explicitly defined in `infer_members.py`; cap and ordering are part of the recipe.

## Rebuild allowed real calibration inputs and site priors

```bash
bash scripts/prepare_real_lumpi.sh
python tools/prepare_v2x_real.py
for part in 4 4r 6 6r; do
  python tools/infer_members.py lumpi --frames "$WORK_ROOT/data/lumpi_real/points_M$part" --member-root "$WORK_ROOT/map_members/M$part"
done
bash scripts/rebuild_lumpi_priors.sh
python tools/rebuild_v2x_prior_predictions.py --infer
bash scripts/prepare_realism.sh lumpi
bash scripts/prepare_realism.sh v2x
```

LUMPI uses three distinct data roles:

- **Earlier labeled validation sets:** M4, M6, M4r and M6r, with 152, 102, 76 and 110 frames respectively. Each frame is at least 300 frame indices from every forbidden ID; the implementation excludes distances ≤300 in both measurements. They support validation, model selection, earlier size statistics and site priors.
- **M4w labeled in-window validation:** 100 annotated Measurement 4 frames within the test time window, each at least three frame indices from every forbidden ID (the code excludes distances ≤2). M4w is used for validation, model selection and box-size statistics, including two-wheeler footprints and wide-truck rules. The final `wide_blob`, `tw_refit`, `near` and `k_person` settings depend on this validation/statistical use. These are permitted non-forbidden validation labels, not hidden test labels.
- **Unlabeled density calibration:** density matching reads allowed point clouds without their annotations. The final LUMPI recipe uses the point-only view of the same 100 M4w frames (`points_M4w`); this is a separate use, not a claim of a disjoint frame set. The histogram generator does not read labels. V2X uses 309 allowed point clouds for density calibration and reserves 50 for the realism check.

Every official forbidden frame is excluded from validation, selection and calibration. V2X uses 359 allowed infrastructure validation frames, with 309 density-calibration and 50 realism-check frames. Its local s0..s5 IDs are a different namespace from official IDs; the guard uses hashes tied to the original scenario-path audit, not string intersection. Real labels are never training targets. Site priors use allowed real points and recurring model predictions, without shipping any map or frame list. Same-pose temporal validation does not establish independent-site generalization.

The V2X helper rebuilds only the pedestrian portion of the historical T_pp map input: other classes do not enter this prior. The original shared 8000 cap was nonbinding. The 359-frame V2X map input was rebuilt exactly from member caches. Twelve LUMPI map smoke cases reproduce the exact box multiset and static-cell support; historical equal-score cross-class order differs in ten cases and is immaterial to these occupancy maps. Full LUMPI map regeneration was not rerun across every real frame. Training and GPU inference were not rerun; exact final replay uses existing local member caches and calibration inputs.

## Exact cached replay and packaging

Use an existing private WORK_ROOT containing the generated member caches, original allowed calibration points/labels/infos, LUMPI OGP maps, V2X T_pp input, and the official kits. `provenance/replay_inputs.json` inventories the external dependency hashes. No original workspace Python or shell modules are imported by the release pipeline; only external input artifacts and the pinned OpenPCDet/official scorer are used.

```bash
# Clone the local release, without importing the development repository history.
git clone --no-hardlinks . ../urbantwin_replay
cd ../urbantwin_replay
# Set WORK_ROOT, RAW_ROOT, KIT_ROOT, PCDET_ROOT to your existing private inputs.
source scripts/env.sh
RUN_ROOT="$PWD/work/lumpi" bash scripts/reproduce_lumpi.sh
RUN_ROOT="$PWD/work/v2x" bash scripts/reproduce_v2x.sh
# A declaration must be supplied by the submitting owner.
python tools/make_submission.py --track lumpi --predictions work/lumpi/predictions.json --synthetic-dir "$WORK_ROOT/outputs/lumpi/synthetic50_win1m" --declaration "$WORK_ROOT/declaration.txt" --out work/submission_lumpi_WINc3.zip
python tools/make_submission.py --track v2x --predictions work/v2x/predictions.json --synthetic-dir "$WORK_ROOT/outputs/v2x/synthetic50_exact1m" --declaration "$WORK_ROOT/declaration.txt" --out work/submission_v2x_G77_H3.zip
```

Verify each reconstructed prediction against its historical archive:

```bash
python tools/verify_predictions.py --actual work/lumpi/predictions.json --expected "$WORK_ROOT/submissions/submission_lumpi_WINc3_win1m.zip"
python tools/verify_predictions.py --actual work/v2x/predictions.json --expected "$WORK_ROOT/submissions/submission_v2x_G77_H3.zip"
```

Place those historical archives at the indicated private paths, or pass their actual location.

Alternatively set DECLARATION before either replay script to pack and check immediately. The packer invokes the official checker and requires valid_submission=1. Dummy-reference detection scores are not leaderboard estimates. Archive bytes can change with declaration text and zip timestamps; exact detection equality and the historical archive hashes are reported separately.

## Outputs and time budget

| Step | Expected output | Time guidance |
|---|---|---|
| Downloads | Separate RAW_ROOT and official kits | Network-dependent; not measured during release verification |
| Synthetic preparation | Seven dataset variants, infos and sampling DBs | Budget tens of minutes to hours; not rebenchmarked |
| Training | 48 selected member/ancestor checkpoints | Budget days on one 8 GB GPU; depends on epoch counts, not rerun |
| Member and map inference | Fixed JSON cache hierarchy | Hours across all TTAs/real sets; not rebenchmarked |
| Density matching | 50 synthetic .bin frames per track | Measured separately where available in VALIDATION.md |
| Cached fusion/postprocess | Two predictions.json files | Exact measured run evidence in VALIDATION.md |
| Packaging | zip and official valid_submission=1 | Typically under a minute per track; measured record in verification/format_checks.json |

The estimates distinguish unmeasured full regeneration from measured release checks. `MANIFEST.md` lists each public file's source, purpose and size. `SANITIZE_REPORT.md` records the all-tree/history scan; `RELEASE_CHECKLIST.md` lists what remains for owner-controlled publication. Code license: Apache-2.0 (`LICENSE`, `NOTICE`).
