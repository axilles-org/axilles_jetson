| model | overall | overground_all | treadmill | walk | turn | rampascent | rampdescent | transition | stand | params | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | 0.820 | 0.715 | 0.868 | 0.829 | 0.827 | 0.851 | 0.802 | 0.626 | -0.625 | 296,481 | 1.67 |
| tcn_emb | 0.849 | 0.733 | 0.902 | 0.868 | 0.814 | 0.842 | 0.831 | 0.660 | -0.663 | 298,453 | 1.62 |
| tcn_emb (mean embedding) | 0.846 | 0.734 | 0.897 | 0.849 | 0.824 | 0.863 | 0.822 | 0.659 | -0.648 | 298,453 | 1.62 |
| film | 0.860 | 0.753 | 0.908 | 0.872 | 0.844 | 0.828 | 0.862 | 0.666 | -0.252 | 300,533 | 1.68 |
| film (mean embedding) | 0.846 | 0.751 | 0.890 | 0.850 | 0.845 | 0.858 | 0.831 | 0.657 | -0.168 | 300,533 | 1.68 |
| film_ctx | 0.820 | 0.659 | 0.893 | 0.864 | 0.847 | 0.828 | 0.832 | 0.671 | -3.213 | 351,585 | 2.56 |
| moe | 0.834 | 0.736 | 0.879 | 0.843 | 0.833 | 0.835 | 0.833 | 0.658 | -0.463 | 307,224 | 1.78 |

R² per mode on held-out subjects (last-timestep prediction, as deployed). Turns are scored only on steps that land on force plates; R² is not meaningful for stand (torque barely varies), use RMSE.

| model | overall | overground_all | treadmill | walk | turn | rampascent | rampdescent | transition | stand |
|---|---|---|---|---|---|---|---|---|---|
| tcn | 0.214 | 0.257 | 0.187 | 0.211 | 0.233 | 0.226 | 0.196 | 0.306 | 0.262 |
| tcn_emb | 0.196 | 0.249 | 0.161 | 0.186 | 0.242 | 0.233 | 0.181 | 0.292 | 0.265 |
| tcn_emb (mean embedding) | 0.198 | 0.248 | 0.166 | 0.199 | 0.235 | 0.216 | 0.186 | 0.292 | 0.264 |
| film | 0.189 | 0.240 | 0.156 | 0.183 | 0.221 | 0.243 | 0.164 | 0.290 | 0.230 |
| film (mean embedding) | 0.198 | 0.240 | 0.171 | 0.198 | 0.221 | 0.220 | 0.181 | 0.293 | 0.222 |
| film_ctx | 0.214 | 0.281 | 0.168 | 0.189 | 0.220 | 0.243 | 0.180 | 0.287 | 0.422 |
| moe | 0.205 | 0.248 | 0.180 | 0.203 | 0.229 | 0.238 | 0.180 | 0.293 | 0.249 |

RMSE (N·m/kg) per mode on held-out subjects.

MoE gate weights by mode (moe):

| mode | expert 0 | expert 1 | expert 2 | expert 3 |
|---|---|---|---|---|
| treadmill | 0.31 | 0.25 | 0.25 | 0.19 |
| walk | 0.34 | 0.22 | 0.24 | 0.19 |
| turn | 0.31 | 0.23 | 0.26 | 0.20 |
| rampascent | 0.23 | 0.21 | 0.35 | 0.21 |
| rampdescent | 0.31 | 0.21 | 0.25 | 0.23 |
| transition | 0.29 | 0.21 | 0.27 | 0.23 |
| stand | 0.25 | 0.18 | 0.28 | 0.29 |
