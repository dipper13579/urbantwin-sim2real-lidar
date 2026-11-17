# Release verification — 2026-11-17

Both pipelines were rerun from a fresh local clone using cached member predictions and the documented neutral input layout, through fusion, postprocessing, packaging and the official local checker.

| Track | Frames / boxes | Verified predictions SHA-256 | Official valid_submission | Replay + pack seconds |
|---|---:|---|---:|---:|
| lumpi | 50 / 123344 | `975d977e057edf24e590ea4d7d138a71d78b7858ca974a158d7f911817291f61` | 1 | 1441.478 |
| v2x | 50 / 66738 | `1db3c468a3b05189cf15b14dee95fc2812d5d203e7760157b6c95b9dadf423e4` | 1 | 126.171 |

Each SHA-256 equals the reviewed historical submission: LUMPI 1052645 and V2X-Real 1054488. The predictions entry inside each newly produced ZIP is byte-identical to the reconstructed JSON. Local ZIP bytes differ because the checks use an anonymous unsigned declaration fixture and new archive metadata. The official checker verifies format with dummy reference data; it does not recompute leaderboard results.

`verification/prediction_comparison.json` records start/end times and hashes. `verification/format_checks.json` records package validity. Full sanitized logs are in `verification/lumpi_replay.log` and `verification/v2x_replay.log`.

## Code and configuration binding

The runtime files used by the fresh clone match the delivered files byte-for-byte; `verification/runtime_code_binding.json` lists their SHA-256 values. Comments/documentation were corrected and the special-member path prefix was renamed to `outputs/v2x/final/`. An AST comparison against the reviewed baseline confirmed that Python executable logic changed only in that path prefix. All 48 member configurations parsed successfully using the actual OpenPCDet loader. Python and Bash syntax checks passed. Runtime hashes are used instead of a self-referential commit hash because evidence and documentation are finalized after the replay; the delivered public branch is a single initial commit.

The neutral private input layout links to the original immutable caches during this verification. No original predictions, weights or data were overwritten. Input artifact digests remain listed in `provenance/replay_inputs.json`; original private development paths are not required by the public scripts.

## Additional retained evidence and limitations

Earlier verification established byte-identical regeneration of both 50-frame synthetic realism sets, exact 359-frame V2X pedestrian-prior reconstruction, exact LUMPI frame selection, and 12 LUMPI prior smoke cases with identical box multisets/static cells. The JSON records are retained as prior evidence; those extra reconstruction checks were not repeated during this documentation/path correction. Equal-score cross-class list order differs in ten of the historical LUMPI map smoke cases, without changing their occupancy support.

No detector retraining, GPU member-cache regeneration, clean-OS installation, full all-frame LUMPI map regeneration, leaderboard submission or publication was performed in this verification. Labeled M4w frames were used for validation, selection and box-size statistics as disclosed in `COMPLIANCE.md`; density matching reads the allowed point clouds without annotations. Same-pose temporal validation does not establish independent-site generalization.
