# Training inputs and real-data use

Training points and object crops originate only from UrbanTwin synthetic data or procedural generators. Original real clouds, annotations, real-derived frame sets, maps and member predictions are not included in this repository. Training logs and checkpoint genealogy are summarized in MEMBERS.md. No new training was performed during this release preparation.

Real data informed more than sizes: validation/model selection, procedural shape/density design, density histograms, static/occupancy priors and inference-time geometry. These uses are disclosed explicitly rather than described as synthetic-only inference. The applicable competition rules and dataset terms remain separate from the code license.

## Statistics and fixed derived parameters

| Use | Source / statistic | Implementation / stored parameters |
|---|---|---|
| Earlier LUMPI size calibration | Per-class median l,w,h from allowed M4 labels within x/y ±40; at least 50 examples; 300-frame exclusion; ratio to synthetic train medians | `calibrate_sizes.size_tables`; Car=set and adopted class rules in lumpi_chain/functions |
| M4w labeled in-window validation and statistics | 100 annotated Measurement 4 frames within the test time window, each at least three frame indices from every forbidden ID; validation, model selection and box-size statistics for two-wheeler footprints and wide-truck rules | `prepare_lumpi_win.py`, `box_sizes.py`, `tw_refit.py`, `wide_blob.py`, `win_stack.py`; final `wide_blob`, `tw_refit`, `near`, `k_person` depend on this use |
| LUMPI synthetic convention | Fixed shift (-1.25,.25,.04); Bicycle/Motorcycle/Van/Bus dimensions; Car point and box ratios | `prepare_training.py`; per-member YAML |
| LUMPI unlabeled density matching | Mean 3D occupancy histogram from the point-only view of 100 allowed M4w frames; annotations are not read; 1 m xy cells, explicit nonuniform z bins | `scripts/prepare_realism.sh`, `realism_match.py` |
| V2X density matching | Mean 3D occupancy histogram over 309 of the 359 allowed validation frames; 1 m xy cells, fixed z bins | Same files; remaining 50 frames are a realism check |
| V2X dense synthetic reconstruction | W_a (132 allowed frames) median dimensions for vehicle and truck; raw UT medians and ratios | All aggregate values and counts in `configs/v2x_dense_statistics.json`; object_keep=2.3, 50000 point cap |
| Synthetic walkers | Observed approximate dimensions, distance-dependent point count/decay/spread, ground-height range, near/far density, occlusion and sampling ranges | `openpcdet_overlay/pcdet/datasets/urbantwin/synth_person.py`; exact SYNTH_PERSON/SYNTH_OCCLUSION, dropout and z-shift settings in each YAML |
| Synthetic clutter | Approximate pole/blob/ground geometry and density, procedural randomness, no real crops | `synth_clutter.py` and per-member YAML |
| Synthetic riders | Target fraction .085; Bicycle/Scooter/Motorcycle mixture .5/.45/.05; target lwh (1.82,.75,1.72), (1.09,.68,1.81), (1.87,.88,1.84) | `tools/synth_rider.py`, `PR1_R14_training.yaml`; procedural points only |
| Static prior | Fraction of allowed real frames with model detections above score .3 in 3x3 neighborhoods of .5 m BEV cells | `static_prior.py`; LUMPI U/rot/TB map chains; V2X 359-frame pedestrian T_pp chain |
| Background prior | Occupied voxel frequency across allowed real frames, evaluated inside test detections | `bg_prior.py`; cell .3 m in V2X; explicit LUMPI chain args |
| Inference geometry | Point medians, local ground/vertical extents, point-count/height/occupancy features, radial and near-range shifts | refine_points/extent/ground, vert_fit, ped_rules, wide_blob, tw_refit, final_rules |
| V2X fixed box calibration | Vehicle (4.55,2.04,1.70), radial -.10; pedestrian (.66,.69,1.72), z+.09, radial+.08; truck scale (1.15,1.08,.95), bottom held | `reproduce_v2x.py`, `v2x_fusion.py` |
| Selected postprocessing | Validation-chosen dimensions, thresholds, score multipliers, yaw consensus and class rules | Fixed scripts/configs only; V2X nested quality setting in `configs/v2x_pedestrian.json`; LUMPI `lumpi_recipe.json` and chain arguments |

`realism_match.py` starts from synthetic UrbanTwin frames and resamples to the aggregate real histogram. Exact mode can generate uniform synthetic filler in underfilled cells; no real point is copied into the submitted synthetic frames. LUMPI adds the documented shift/noise. Synthetic realism frames are outside the training split.

## Exclusion and limits of the evidence

The earlier labeled LUMPI validation sets M4, M6, M4r and M6r are each at least 300 frame indices from every forbidden ID; the implemented exclusion is distance ≤300 in both measurements. `calibrate_sizes.py` uses this older margin for its class medians. Separately, M4w contains 100 labeled Measurement 4 frames inside the test time window, each at least three frame indices from every forbidden ID (distance ≤2 is excluded). M4w supports validation, model selection and box-size statistics used by the final geometry/score recipe; its labels never enter detector training. Unlabeled density calibration reads only the allowed point clouds: the final LUMPI recipe reuses M4w point clouds without reading their labels, while V2X uses its 309-frame calibration subset. These roles are distinguished even where point-cloud frame identities overlap. Selection code reconstructs local file lists, which remain outside Git. The legacy CLASS_NAMES mapping used in size calibration is retained for exact arithmetic; ORG_CLASS_NAMES and to_org handle the organizer Van/Unknown naming for validation labels.

V2X local scene indices must not be compared directly with official IDs. The prior audit mapped all 100 forbidden IDs to original scenario/-1/frame paths and found zero intersections with the whole val archive and its 359-frame subset. `guard_v2x.py` verifies both archive and forbidden-list digests before extracting or computing labels. It intentionally rejects different versions, which require a fresh canonical-path audit. W_a/W_b frame collections are excluded; only the fixed W_a dimension aggregates required for synthetic reconstruction are included.

Temporal groups from the same sensor pose support exposed validation/regression checks, not independent-site evidence. Exact cached replay demonstrates the published code reproduces the selected predictions from the listed caches; it does not establish deterministic retraining, a clean-system installation or an independent leaderboard rerun.
