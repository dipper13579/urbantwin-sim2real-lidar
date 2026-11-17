# Member training map

48 checkpoints, including training ancestors and V3 for the V2X static prior. All are external assets. Epoch is read from each trusted local checkpoint; parent and schedule are traced through source training logs. The selected saved epoch is authoritative. See `configs/members.json` for artifact paths, hashes, logs and seeds. No training was rerun during release verification.

| Member | Role | Config | Parent | Planned / saved epochs | Fixed seed |
|---|---|---|---|---:|---|
| lumpi/A_cp20 | final member | `configs/lumpi_centerpoint.yaml` | scratch | 20 / 20 | False |
| lumpi/F_v3 | final member | `configs/lumpi_centerpoint_v3.yaml` | F_v8 | 8 / 20 | False |
| lumpi/F_v8 | final member | `configs/lumpi_centerpoint_v8.yaml` | F_v4 | 8 / 8 | False |
| lumpi/F_v4 | final member | `configs/lumpi_centerpoint_v4.yaml` | scratch | 30 / 30 | False |
| lumpi/F_v7 | final member | `configs/lumpi_centerpoint_v7.yaml` | F_v4 | 8 / 8 | False |
| lumpi/F_v9 | final member | `configs/lumpi_centerpoint_v9.yaml` | F_v7 | 8 / 8 | False |
| lumpi/F_v11 | final member | `configs/lumpi_centerpoint_v11.yaml` | F_v9 | 8 / 8 | False |
| lumpi/F_v16 | final member | `configs/lumpi_centerpoint_v16.yaml` | F_v11 | 3 / 3 | False |
| lumpi/F_v19 | final member | `configs/lumpi_centerpoint_v19.yaml` | F_v8 | 8 / 8 | False |
| lumpi/F_v22 | final member | `configs/lumpi_centerpoint_v22.yaml` | F_v11 | 6 / 6 | False |
| lumpi/F_v23 | final member | `configs/lumpi_centerpoint_v23.yaml` | F_v9 | 6 / 6 | False |
| lumpi/F_v24 | final member | `configs/lumpi_centerpoint_v24.yaml` | F_v4 | 6 / 6 | False |
| lumpi/F_v25 | final member | `configs/lumpi_centerpoint_v25.yaml` | F_v7 | 6 / 6 | False |
| lumpi/F_v26 | final member | `configs/lumpi_centerpoint_v26.yaml` | F_v16 | 6 / 6 | False |
| lumpi/F_v27 | final member | `configs/lumpi_centerpoint_v27.yaml` | F_v16 | 6 / 6 | False |
| lumpi/F_v30 | final member | `configs/lumpi_centerpoint_v30.yaml` | F_v4 | 6 / 6 | False |
| lumpi/F_v34 | final member | `configs/lumpi_centerpoint_v34.yaml` | F_v4 | 8 / 8 | False |
| lumpi/F_v35 | final member | `configs/lumpi_centerpoint_v35.yaml` | F_v27 | 6 / 6 | False |
| lumpi/F_v36 | final member | `configs/lumpi_centerpoint_v36.yaml` | F_v3 | 8 / 8 | False |
| v2x/V25 | final member | `configs/v2x_centerpoint_v25.yaml` | V12 | 8 / 8 | False |
| v2x/V12 | final member | `configs/v2x_centerpoint_v12.yaml` | V10 | 8 / 8 | False |
| v2x/V10 | training ancestor | `configs/v2x_centerpoint_v10.yaml` | V9 | 8 / 8 | False |
| v2x/V9 | training ancestor | `configs/v2x_centerpoint_v9.yaml` | V8 | 8 / 8 | False |
| v2x/V8 | training ancestor | `configs/v2x_centerpoint_v8.yaml` | V7 | 8 / 8 | False |
| v2x/V7 | final member | `configs/v2x_centerpoint_v7.yaml` | V6 | 8 / 8 | False |
| v2x/V6 | final member | `configs/v2x_centerpoint_v6.yaml` | V5 | 8 / 8 | False |
| v2x/V5 | final member | `configs/v2x_centerpoint_v5.yaml` | V4 | 8 / 8 | False |
| v2x/V4 | final member | `configs/v2x_centerpoint_v4.yaml` | scratch | 20 / 20 | False |
| v2x/V21 | final member | `configs/v2x_centerpoint_v21.yaml` | V10 | 8 / 8 | False |
| v2x/V26 | final member | `configs/v2x_centerpoint_v26.yaml` | V22 | 8 / 8 | False |
| v2x/V22 | final member | `configs/v2x_centerpoint_v22.yaml` | V9 | 8 / 8 | False |
| v2x/V27 | final member | `configs/v2x_centerpoint_v27.yaml` | V16 | 8 / 8 | False |
| v2x/V16 | training ancestor | `configs/v2x_centerpoint_v16.yaml` | V8 | 8 / 8 | False |
| v2x/V28 | final member | `configs/v2x_centerpoint_v28.yaml` | V21 | 8 / 8 | False |
| v2x/V30 | final member | `configs/v2x_centerpoint_v30.yaml` | V24 | 8 / 8 | False |
| v2x/V24 | training ancestor | `configs/v2x_centerpoint_v24.yaml` | V12 | 8 / 8 | False |
| v2x/V33 | final member | `configs/v2x_centerpoint_v33.yaml` | V29 | 8 / 8 | False |
| v2x/V29 | final member | `configs/v2x_centerpoint_v29.yaml` | V15 | 8 / 8 | False |
| v2x/V15 | final member | `configs/v2x_centerpoint_v15.yaml` | V9 | 8 / 8 | False |
| v2x/VR1 | final member | `configs/v2x_voxelrcnn.yaml` | V7 | 1 / 1 | False |
| v2x/B_cp16_dens | final member | `configs/v2x_centerpoint.yaml` | scratch | 16 / 16 | False |
| v2x/V2 | final member | `configs/v2x_centerpoint_v2.yaml` | V15 | 8 / 20 | False |
| v2x/V36 | final member | `configs/v2x_centerpoint_v36.yaml` | V30 | 8 / 8 | False |
| v2x/NEW_V | final member | `configs/NEW_V_training.yaml` | V30 | 4 / 4 | True |
| v2x/R24_truck | final member | `configs/R24_truck_training.yaml` | V36 | 4 / 4 | True |
| v2x/R28_truck | final member | `configs/R28_truck_training.yaml` | V29 | 4 / 4 | True |
| v2x/PR1_R14 | final member | `configs/PR1_R14_training.yaml` | V30 | 4 / 4 | True |
| v2x/V3 | static-prior member | `configs/v2x_centerpoint_v3.yaml` | V30 | 8 / 24 | False |

Training order is a topological traversal of `parent`; it is not table order. Train a parent first, copy its selected saved epoch to the `checkpoint` location under WORK_ROOT, then train its child. The special R24 vehicle (NEW_V), R24 truck, R28 truck and PR1_R14 recipes use seed 666 and four epochs. Unfixed random seeds, procedural generators and GPU kernels mean that retraining is not expected to reproduce historical weights bit for bit.
