# Spectral bias sweep (sphere, spherical harmonics)

Steps until the seed-averaged rho_l (recovered fraction of target degree l) first reaches 0.9 (`-` = not within the run); final = value at the last step. Checkpoints are log-spaced, so each step count is good to about 25%.

## lr 0.0003

| model | params | runs | s/run | l=1 | l=2 | l=3 | l=4 | l=5 | l=6 | l=7 | l=8 | final min_l | final MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP ReLU, 6x138 | 97,996 | 10 | 37 | 47 | 73 | 112 | 149 | 198 | 263 | 404 | 538 | 1.00 | 6.26e-03 |
| MLP ReLU 6x256 (paper) | 330,241 | 10 | 71 | 27 | 47 | 73 | 97 | 129 | 198 | 304 | 404 | 1.00 | 2.26e-03 |
| CGENN, x as vector + frame | 98,697 | 10 | 178 | 81 | 211 | 382 | 484 | 692 | 877 | 1113 | 1412 | 1.00 | 3.58e-03 |

## lr 0.0001

| model | params | runs | s/run | l=1 | l=2 | l=3 | l=4 | l=5 | l=6 | l=7 | l=8 | final min_l | final MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP ReLU, 6x138 | 97,996 | 10 | 37 | 112 | 172 | 228 | 304 | 350 | 466 | 621 | 826 | 1.00 | 4.39e-03 |
| MLP ReLU 6x256 (paper) | 330,241 | 10 | 71 | 63 | 97 | 129 | 172 | 228 | 304 | 404 | 538 | 1.00 | 3.62e-03 |
| CGENN, x as vector + frame | 98,697 | 10 | 178 | 237 | 545 | 988 | 1253 | 1590 | 2017 | 2558 | 2881 | 1.00 | 3.69e-03 |

## lr 0.001

| model | params | runs | s/run | l=1 | l=2 | l=3 | l=4 | l=5 | l=6 | l=7 | l=8 | final min_l | final MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLP ReLU, 6x138 | 97,996 | 10 | 37 | 20 | 36 | 55 | 84 | 112 | 172 | 263 | 350 | 1.00 | 4.74e-03 |
| MLP ReLU 6x256 (paper) | 330,241 | 10 | 71 | 10 | 20 | 41 | 63 | 97 | 129 | 198 | 228 | 0.99 | 1.51e-02 |
| CGENN, x as vector + frame | 98,697 | 10 | 178 | 28 | 81 | 147 | 211 | 301 | 382 | 484 | 614 | 1.00 | 2.82e-03 |
