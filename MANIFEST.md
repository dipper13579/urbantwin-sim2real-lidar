# Release file manifest

This inventory describes the public source tree. Weights, datasets, real-derived frame lists/maps, cached predictions, signed declarations and submission archives are external. Internal agent progress and original private source references are retained outside this repository.

Source aliases identify provenance, not executable paths: SOURCE_ROOT denotes the private original project; RAW_ROOT denotes separately obtained datasets. Neutral WORK_ROOT paths in the member and asset inventories describe the reproducible input layout. No original development history is included in the public branch.

## Submission trace

- LUMPI 1052645 / WINc3: `scripts/reproduce_lumpi.sh` runs the adopted base chain, Person iii, Car F_v34/F_v36 + F_v35, and Bicycle F_v36/F_v34. `win_stack.py` applies the fixed recipe; `final_rules.py` and the Bicycle-only merge yield the selected final predictions. M4w supplies labeled validation/model-selection and box-size statistics; its point-only view separately supplies unlabeled density calibration.
- V2X-Real 1054488 / G77 H3: `scripts/reproduce_v2x.sh` invokes the fixed vehicle, pedestrian/rider and truck pipelines in `reproduce_v2x.py`, with geometry/fusion and quality functions in `v2x_fusion.py` and `v2x_quality.py`. Special-member caches and checkpoints use `outputs/v2x/final/`. Source ordering, caps and recipe values are unchanged.
- Training genealogy and epochs: `MEMBERS.md` and `configs/members.json`; synthetic preparation: `prepare_training.py` and the OpenPCDet overlay; realism frames: `scripts/prepare_realism.sh`; packing/official checker: `make_submission.py`.
- External artifact identities: `provenance/release_assets.json` and `provenance/replay_inputs.json`. Fresh-clone replay results and runtime source hashes: `verification/` and `VALIDATION.md`.

## File inventory

Sizes are delivered file bytes. Git storage is checked separately. Original private analysis references removed from public comments remain in the author handoff; they are not presented as available public scripts.

| File | Purpose | Source | Bytes |
|---|---|---|---:|
| `.gitignore` | Repository support metadata | Release-generated | 113 |
| `COMPLIANCE.md` | Labeled validation, unlabeled calibration, training and exclusion disclosure | Release-generated documentation | 6536 |
| `LICENSE` | Apache-2.0 license | SOURCE_ROOT/OpenPCDet/LICENSE (Apache-2.0) | 11357 |
| `MANIFEST.md` | Public per-file provenance and size inventory | Release-generated inventory / documentation | 25940 |
| `MEMBERS.md` | Member configuration, parent and saved epoch map | Release-generated documentation | 5235 |
| `NOTICE` | Author copyright and preserved upstream notices | Release-generated documentation | 672 |
| `README.md` | Reproduction instructions, authorship, paper and results | Release-generated documentation | 13135 |
| `RELEASE_ASSETS.md` | External artifact filenames, sizes, hashes and members | Release-generated documentation | 7261 |
| `RELEASE_CHECKLIST.md` | Owner publication checklist | Release-generated documentation | 1284 |
| `SANITIZE_REPORT.md` | Privacy/data/size and one-commit-history scan | Release-generated inventory / documentation | 1936 |
| `VALIDATION.md` | Fresh-clone replay, hash, format and scope record | Release-generated inventory / documentation | 3306 |
| `configs/NEW_V_training.yaml` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 5040 |
| `configs/PR1_R14_training.yaml` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 5924 |
| `configs/R24_truck_training.yaml` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 5059 |
| `configs/R28_truck_training.yaml` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 5059 |
| `configs/lumpi_centerpoint.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint.yaml | 2549 |
| `configs/lumpi_centerpoint_v11.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v11.yaml | 4231 |
| `configs/lumpi_centerpoint_v16.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v16.yaml | 4236 |
| `configs/lumpi_centerpoint_v19.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v19.yaml | 4661 |
| `configs/lumpi_centerpoint_v22.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v22.yaml | 5379 |
| `configs/lumpi_centerpoint_v23.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v23.yaml | 5389 |
| `configs/lumpi_centerpoint_v24.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v24.yaml | 4436 |
| `configs/lumpi_centerpoint_v25.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v25.yaml | 4783 |
| `configs/lumpi_centerpoint_v26.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v26.yaml | 5384 |
| `configs/lumpi_centerpoint_v27.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v27.yaml | 5344 |
| `configs/lumpi_centerpoint_v3.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v3.yaml | 3274 |
| `configs/lumpi_centerpoint_v30.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v30.yaml | 4514 |
| `configs/lumpi_centerpoint_v34.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v34.yaml | 3513 |
| `configs/lumpi_centerpoint_v35.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v35.yaml | 5456 |
| `configs/lumpi_centerpoint_v36.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v36.yaml | 3396 |
| `configs/lumpi_centerpoint_v4.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v4.yaml | 3289 |
| `configs/lumpi_centerpoint_v7.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v7.yaml | 3636 |
| `configs/lumpi_centerpoint_v8.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v8.yaml | 3477 |
| `configs/lumpi_centerpoint_v9.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_centerpoint_v9.yaml | 4242 |
| `configs/lumpi_dataset.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/lumpi_dataset.yaml | 1907 |
| `configs/lumpi_person_members.txt` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/outputs/lumpi/win/person_iii.txt | 138 |
| `configs/lumpi_recipe.json` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/outputs/lumpi/win/recipe1.json | 87 |
| `configs/members.json` | Fixed model/data or final-pipeline configuration | Release checkpoint and log inventory | 27761 |
| `configs/v2x_centerpoint.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint.yaml | 2372 |
| `configs/v2x_centerpoint_v10.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v10.yaml | 4338 |
| `configs/v2x_centerpoint_v12.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v12.yaml | 5398 |
| `configs/v2x_centerpoint_v15.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v15.yaml | 5395 |
| `configs/v2x_centerpoint_v16.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v16.yaml | 5255 |
| `configs/v2x_centerpoint_v2.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v2.yaml | 3038 |
| `configs/v2x_centerpoint_v21.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v21.yaml | 5398 |
| `configs/v2x_centerpoint_v22.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v22.yaml | 5395 |
| `configs/v2x_centerpoint_v24.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v24.yaml | 5528 |
| `configs/v2x_centerpoint_v25.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v25.yaml | 5893 |
| `configs/v2x_centerpoint_v26.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v26.yaml | 5848 |
| `configs/v2x_centerpoint_v27.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v27.yaml | 5708 |
| `configs/v2x_centerpoint_v28.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v28.yaml | 5851 |
| `configs/v2x_centerpoint_v29.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v29.yaml | 5848 |
| `configs/v2x_centerpoint_v3.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v3.yaml | 3116 |
| `configs/v2x_centerpoint_v30.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v30.yaml | 5981 |
| `configs/v2x_centerpoint_v33.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v33.yaml | 6040 |
| `configs/v2x_centerpoint_v36.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v36.yaml | 6078 |
| `configs/v2x_centerpoint_v4.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v4.yaml | 3205 |
| `configs/v2x_centerpoint_v5.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v5.yaml | 3465 |
| `configs/v2x_centerpoint_v6.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v6.yaml | 3920 |
| `configs/v2x_centerpoint_v7.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v7.yaml | 4068 |
| `configs/v2x_centerpoint_v8.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v8.yaml | 4196 |
| `configs/v2x_centerpoint_v9.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_centerpoint_v9.yaml | 4336 |
| `configs/v2x_dataset.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_dataset.yaml | 1841 |
| `configs/v2x_dense_statistics.json` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 1112 |
| `configs/v2x_pedestrian.json` | Fixed model/data or final-pipeline configuration | Selected final-model development source; exact original reference retained in the private author handoff | 1322 |
| `configs/v2x_voxelrcnn.yaml` | Fixed model/data or final-pipeline configuration | SOURCE_ROOT/OpenPCDet/tools/cfgs/urbantwin/v2x_voxelrcnn.yaml | 6603 |
| `openpcdet_overlay/LICENSE` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/LICENSE | 11357 |
| `openpcdet_overlay/pcdet/datasets/__init__.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/__init__.py | 3166 |
| `openpcdet_overlay/pcdet/datasets/augmentor/data_augmentor.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/augmentor/data_augmentor.py | 15860 |
| `openpcdet_overlay/pcdet/datasets/urbantwin/__init__.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/urbantwin/__init__.py | 0 |
| `openpcdet_overlay/pcdet/datasets/urbantwin/synth_clutter.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/urbantwin/synth_clutter.py | 8947 |
| `openpcdet_overlay/pcdet/datasets/urbantwin/synth_person.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/urbantwin/synth_person.py | 11040 |
| `openpcdet_overlay/pcdet/datasets/urbantwin/urbantwin_dataset.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/datasets/urbantwin/urbantwin_dataset.py | 8242 |
| `openpcdet_overlay/pcdet/models/detectors/detector3d_template.py` | Pinned OpenPCDet overlay or retained upstream license | SOURCE_ROOT/OpenPCDet/pcdet/models/detectors/detector3d_template.py | 19513 |
| `provenance/dataset_licenses.json` | Source/input/environment/license inventory | RAW_ROOT/UT-LUMPI/dataverse_metadata.json; RAW_ROOT/UT-V2X-Real-IC/dataverse_metadata.json (license fields only) | 1367 |
| `provenance/environment.json` | Source/input/environment/license inventory | Release-generated documentation | 784 |
| `provenance/origins.json` | Source/input/environment/license inventory | Release-generated inventory / documentation | 13450 |
| `provenance/release_assets.json` | Source/input/environment/license inventory | Release SHA-256 inventory of original artifacts | 13744 |
| `provenance/replay_inputs.json` | Source/input/environment/license inventory | Release-generated inventory / documentation | 64957 |
| `provenance/v2x_validation_guard.json` | Source/input/environment/license inventory | Private development source; original reference retained in the author handoff outside this repository | 625 |
| `requirements.txt` | Measured Python package versions | Release-generated documentation | 225 |
| `scripts/env.sh` | Portable fixed workflow: env | Release-generated | 576 |
| `scripts/install_overlay.sh` | Portable fixed workflow: install overlay | Release-generated | 292 |
| `scripts/lumpi_chain.sh` | Portable fixed workflow: lumpi chain | SOURCE_ROOT/setup/win_chain.sh (portable paths) | 10338 |
| `scripts/lumpi_final_sources.sh` | Portable fixed workflow: lumpi final sources | SOURCE_ROOT/setup/win_var.sh (adopted branches only) | 3110 |
| `scripts/lumpi_functions.sh` | Portable fixed workflow: lumpi functions | SOURCE_ROOT/setup/win_chain.sh (constants and functions) | 5579 |
| `scripts/prepare_real_lumpi.sh` | Portable fixed workflow: prepare real lumpi | Release-generated | 1073 |
| `scripts/prepare_realism.sh` | Portable fixed workflow: prepare realism | SOURCE_ROOT/setup/win_pack_c2.sh; setup/12_v2x_eval.sh (final synthesis commands) | 799 |
| `scripts/rebuild_lumpi_priors.sh` | Portable fixed workflow: rebuild lumpi priors | Private development source; original reference retained in the author handoff outside this repository | 3015 |
| `scripts/reproduce_lumpi.sh` | Portable fixed workflow: reproduce lumpi | Release-generated | 1411 |
| `scripts/reproduce_v2x.sh` | Portable fixed workflow: reproduce v2x | Release-generated | 445 |
| `tools/bg_prior.py` | Point-level background prior for a fixed roadside site. Voxel occupancy over unlabeled non-forbidden real frames | SOURCE_ROOT/tools/bg_prior.py | 3917 |
| `tools/box_sizes.py` | Class-level box sizes for LUMPI from box-size statistics of real validation labels (non-forbidden frames of the test window, | SOURCE_ROOT/tools/box_sizes.py | 3648 |
| `tools/calibrate_sizes.py` | Post-hoc box-size calibration of predictions toward real LUMPI annotation conventions. | SOURCE_ROOT/tools/calibrate_sizes.py | 3912 |
| `tools/class_resolve.py` | Two-wheeler vs pedestrian resolution for LUMPI. Real LUMPI labels a cyclist as one Bicycle box (rider included, | SOURCE_ROOT/tools/class_resolve.py | 4871 |
| `tools/ensemble.py` | Weighted box fusion of several predictions.json files (same frames). | SOURCE_ROOT/tools/ensemble.py | 3554 |
| `tools/fetch_ut.py` | Selective download from split zips over HTTP (UrbanTwin on Harvard Dataverse, or any URL). | SOURCE_ROOT/tools/fetch_ut.py | 12260 |
| `tools/final_rules.py` | Final label-free rules on a stacked LUMPI package (outputs/lumpi/win/prereg.md 2c).  Every frame is processed on its own, from | SOURCE_ROOT/tools/final_rules.py | 4889 |
| `tools/guard_v2x.py` | Fail closed unless inputs match the canonical-identity audit; no real IDs stored here. | Release-generated | 787 |
| `tools/infer_members.py` | Generate adopted test member caches with original TTA ordering and export limits. | Release-generated | 3454 |
| `tools/lumpi_real.py` | Helpers for the real LUMPI data used for calibration/validation only (never for training). | SOURCE_ROOT/tools/lumpi_real.py | 2931 |
| `tools/lv_geo.py` | Long-vehicle geometry for LUMPI (Truck / Bus): refit the boxes to the points they sit on and re-score the class from the measured | SOURCE_ROOT/tools/lv_geo.py | 12064 |
| `tools/make_submission.py` | Pack a Codabench submission for the UrbanTwin Sim2Real challenge, then run the | SOURCE_ROOT/tools/make_submission.py | 6704 |
| `tools/merge_predictions.py` | Class-routed merge of predictions.json files: take each class from the run you choose. | SOURCE_ROOT/tools/merge_predictions.py | 1559 |
| `tools/ped_rules.py` | Label-free point-feature score rules for pedestrian boxes (V2X). Two thirds of the detector's pedestrian boxes | SOURCE_ROOT/tools/ped_rules.py | 4287 |
| `tools/postprocess.py` | Class-level post-processing of predictions.json, tuned on the real validation set. | SOURCE_ROOT/tools/postprocess.py | 6294 |
| `tools/predict.py` | Run a trained OpenPCDet model and write detections in the challenge's predictions.json format. | SOURCE_ROOT/tools/predict.py | 9606 |
| `tools/prepare_lumpi_rare.py` | Rare-class (Bus / Motorcycle) real LUMPI validation sets from the frames fetched by setup/18_fetch_rare.sh. | SOURCE_ROOT/tools/prepare_lumpi_rare.py | 3193 |
| `tools/prepare_lumpi_real.py` | Turn downloaded real LUMPI frames into a challenge-like validation / realism-calibration set. | SOURCE_ROOT/tools/prepare_lumpi_real.py | 3747 |
| `tools/prepare_lumpi_win.py` | In-window real LUMPI validation set M4w (frames fetched by setup/132_fetch_win.sh) and organizer-named GT for the older sets. | SOURCE_ROOT/tools/prepare_lumpi_win.py | 4574 |
| `tools/prepare_training.py` | Rebuild the exact synthetic dataset variants. No real points are training inputs. | Release-generated | 1282 |
| `tools/prepare_ut.py` | Convert a downloaded UrbanTwin sequence into the training layout used by UrbanTwinDataset. | SOURCE_ROOT/tools/prepare_ut.py | 11804 |
| `tools/prepare_v2x_real.py` | Real V2X-Real validation / realism-calibration set from val.zip (infrastructure LiDAR -1 only). | SOURCE_ROOT/tools/prepare_v2x_real.py | 5315 |
| `tools/realism_match.py` | Density-matched synthetic frames for the realism score. | SOURCE_ROOT/tools/realism_match.py | 8591 |
| `tools/rebuild_v2x_prior_predictions.py` | Rebuild the adopted 359-frame pedestrian static-map input; no real data vendored. | SOURCE_ROOT/setup/120_v2x_rot.sh (fixed v3029rotR10 branch) | 2326 |
| `tools/refine_extent.py` | Point-based length refinement of large-vehicle boxes (Bus / Truck): measure the frame's points along each | SOURCE_ROOT/tools/refine_extent.py | 4056 |
| `tools/refine_ground.py` | Per-object ground height: put each box bottom at the local ground (from the frame's own points) + a per-class | SOURCE_ROOT/tools/refine_ground.py | 4750 |
| `tools/refine_points.py` | Point-based refinement of small boxes (pedestrians): pull each box center toward the median of the frame's | SOURCE_ROOT/tools/refine_points.py | 3281 |
| `tools/release_paths.py` | User-owned inputs live outside the public checkout. | Release-generated | 460 |
| `tools/reproduce_v2x.py` | Reproduce H3 from cached per-member predictions and local point/prior inputs. | Private development source; original reference retained in the author handoff outside this repository | 10219 |
| `tools/runtime_config.py` | Resolve portable YAML with the actual OpenPCDet parser. | Release-generated | 508 |
| `tools/sanitize.py` | Scan the complete release tree and every reachable Git blob/commit. | Release-generated | 4193 |
| `tools/select_lumpi_frames.py` | Reconstruct adopted frame selections, writing only private local name lists. | Private development source; original reference retained in the author handoff outside this repository | 1962 |
| `tools/select_rare_frames.py` | Pick extra real LUMPI frames rich in rare classes (Bus, Motorcycle) for a separate validation set. | SOURCE_ROOT/tools/select_rare_frames.py | 3315 |
| `tools/static_prior.py` | Static false-positive prior for a fixed roadside site: poles, bushes and sign posts get detected as Person / | SOURCE_ROOT/tools/static_prior.py | 3051 |
| `tools/synth_rider.py` | R14 procedural rider augmentation. Task-local, synthetic training only. | Selected final-model development source; exact original reference retained in the private author handoff | 16761 |
| `tools/train_member.py` | Train one inventory member, or parse all recipes without training. | Release-generated | 2309 |
| `tools/tw_object.py` | Two-wheeler objects: share the confidence of a bike-shaped object between the classes that label it (label-free rules). | SOURCE_ROOT/tools/tw_object.py | 5432 |
| `tools/tw_refit.py` | Two-wheeler box refit: heading / centre from the point cluster under the box (label-free) + label-statistics footprint. | SOURCE_ROOT/tools/tw_refit.py | 6359 |
| `tools/v2x_fusion.py` | Adopted deterministic WBF and pedestrian geometry; no search code. | Private development source; original reference retained in the author handoff outside this repository | 4257 |
| `tools/v2x_quality.py` | v2x_quality | Private development source; original reference retained in the author handoff outside this repository | 5413 |
| `tools/van_geo.py` | Van geometry re-score (label-free): a van is a vehicle whose roof is clearly above a car's and below a lorry's. | SOURCE_ROOT/tools/van_geo.py | 3802 |
| `tools/verify_predictions.py` | Compare every ordered record and byte against predictions.json or a submission ZIP. | Release-generated exact JSON/byte comparison | 933 |
| `tools/vert_fit.py` | Point-fitted vertical placement of boxes (label-free, uses only the frame's own points). On V2X real val the | SOURCE_ROOT/tools/vert_fit.py | 3686 |
| `tools/wide_blob.py` | Wide tall objects for LUMPI (Truck): a box for objects that are wider than a road vehicle's body (working machines, vehicles with | SOURCE_ROOT/tools/wide_blob.py | 6881 |
| `tools/win_stack.py` | Inference geometry / calibrated size modules on a finished LUMPI package, in one fixed order, driven by a recipe (JSON). | SOURCE_ROOT/tools/win_stack.py | 3986 |
| `verification/asset_inventory.json` | Inspectible verification evidence | Release-generated validation evidence | 220 |
| `verification/config_parse.json` | Inspectible verification evidence | Release-generated validation evidence | 8102 |
| `verification/format_checks.json` | Inspectible verification evidence | Release-generated validation evidence | 859 |
| `verification/lumpi_map_smoke.json` | Inspectible verification evidence | Release-generated validation evidence | 3489 |
| `verification/lumpi_replay.log` | Inspectible verification evidence | Release-generated validation evidence | 2928 |
| `verification/lumpi_selections.json` | Inspectible verification evidence | Release-generated validation evidence | 570 |
| `verification/nms_equivalence.json` | Inspectible verification evidence | Release-generated | 311 |
| `verification/prediction_comparison.json` | Inspectible verification evidence | Release-generated validation evidence | 1719 |
| `verification/runtime_checks.json` | Inspectible verification evidence | Release-generated validation evidence | 187 |
| `verification/runtime_code_binding.json` | Inspectible verification evidence | Release-generated validation evidence | 17078 |
| `verification/sanitize.json` | Inspectible verification evidence | Release-generated validation evidence | 699 |
| `verification/synthetic_rebuild.json` | Inspectible verification evidence | Release-generated validation evidence | 348 |
| `verification/v2x_guard_check.json` | Inspectible verification evidence | Release-generated validation evidence | 625 |
| `verification/v2x_map_rebuild.json` | Inspectible verification evidence | Release-generated validation evidence | 189 |
| `verification/v2x_replay.log` | Inspectible verification evidence | Release-generated validation evidence | 1591 |
