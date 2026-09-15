# Predictive Temporally Structured World Transition U

RoboCasa first-round implementation scaffold aligned with
[`docs/plans/Predictive_Temporally_Structured_World_Transition_U_RoboCasa_v0.1.md`](docs/plans/Predictive_Temporally_Structured_World_Transition_U_RoboCasa_v0.1.md).

## Goal

Learn a **predictive** temporally structured latent

\\[
U_t = [u_t^1, u_t^2, u_t^3, u_t^4]
\\]

from **history + language only**:

\\[
U_t = P_\\theta(I_{t-M:t}, S_{t-M:t}, L)
\\]

Future RGB / state / action are **supervision only** — they never enter the U predictor.

## Default window (v0.1 §7 / §48)

| Item | Value |
|------|-------|
| History | 0.5 s, \\(N_h=4\\) RGB frames |
| Future | 1.5 s, \\(H=30\\) @ 20 Hz |
| \\(K\\) | 4 equal intervals |
| Views | `left`, `wrist` (eye optional later) |
| \\(d_U\\) | 512 |
| Visual encoder | Frozen DINOv3 + 4×4 pool / view |

## Package layout

```
pts_wtu/          # library
configs/pts_wtu/  # data + Stage A/B configs
scripts/pts_wtu/  # index / export / train / probes
docs/plans/       # experiment plan v0.1
```

## Implementation order (§47)

1. Window dataset  
2. Offline DINOv3  
3. Normalization  
4–7. History encoders + U predictor + decoders  
8. Stage-A training (`L_V + L_S`)  
9–10. Frozen-U probes + U intervention  
11–12. Action head + Stage B (dirs reserved)  
13–16. Baselines / counterfactual / rollout — later  

## Install

```bash
pip install -e ".[train]"
```

## Data

Point `configs/pts_wtu/data_robocasa.yaml` at your local RoboCasa365 root.
Large features and checkpoints stay **out of git** (`artifacts/`).

## Status

This commit is a **scaffold**: module signatures and configs only.
Training logic will be filled following §47.
