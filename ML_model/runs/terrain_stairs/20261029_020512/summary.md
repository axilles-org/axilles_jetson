| model | overall | overground_all | treadmill | walk | turn | rampascent | rampdescent | stairascent | stairdescent | transition | stand | params | CPU ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | 0.835 | 0.759 | 0.880 | 0.843 | 0.843 | 0.859 | 0.823 | 0.832 | 0.829 | 0.703 | -0.515 | 296,481 | 2.32 |
| tcn_emb | 0.865 | 0.779 | 0.916 | 0.890 | 0.845 | 0.860 | 0.856 | 0.795 | 0.818 | 0.734 | -0.457 | 298,453 | 2.36 |
| tcn_emb (mean embedding) | 0.849 | 0.772 | 0.894 | 0.856 | 0.839 | 0.871 | 0.828 | 0.881 | 0.818 | 0.721 | -0.474 | 298,453 | 2.36 |
| film | 0.855 | 0.760 | 0.912 | 0.873 | 0.837 | 0.839 | 0.856 | 0.771 | 0.811 | 0.715 | -0.611 | 300,533 | 2.41 |
| film (mean embedding) | 0.841 | 0.761 | 0.889 | 0.847 | 0.839 | 0.862 | 0.815 | 0.849 | 0.818 | 0.709 | -0.525 | 300,533 | 2.41 |
| film_ctx | 0.846 | 0.768 | 0.893 | 0.854 | 0.842 | 0.820 | 0.831 | 0.820 | 0.801 | 0.707 | 0.052 | 351,585 | 3.62 |
| moe | 0.840 | 0.763 | 0.885 | 0.852 | 0.842 | 0.853 | 0.831 | 0.830 | 0.819 | 0.710 | -0.470 | 307,224 | 2.49 |

R² per mode on held-out subjects (last-timestep prediction, as deployed). Turns are scored only on steps that land on force plates; R² is not meaningful for stand (torque barely varies), use RMSE.

| model | overall | overground_all | treadmill | walk | turn | rampascent | rampdescent | stairascent | stairdescent | transition | stand |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn | 0.203 | 0.234 | 0.178 | 0.203 | 0.222 | 0.220 | 0.185 | 0.204 | 0.172 | 0.265 | 0.253 |
| tcn_emb | 0.184 | 0.224 | 0.149 | 0.169 | 0.220 | 0.219 | 0.167 | 0.226 | 0.177 | 0.251 | 0.248 |
| tcn_emb (mean embedding) | 0.194 | 0.227 | 0.168 | 0.194 | 0.225 | 0.210 | 0.183 | 0.172 | 0.178 | 0.257 | 0.250 |
| film | 0.190 | 0.233 | 0.153 | 0.183 | 0.227 | 0.235 | 0.167 | 0.238 | 0.181 | 0.260 | 0.261 |
| film (mean embedding) | 0.199 | 0.233 | 0.172 | 0.200 | 0.225 | 0.218 | 0.190 | 0.193 | 0.178 | 0.263 | 0.254 |
| film_ctx | 0.196 | 0.230 | 0.169 | 0.196 | 0.223 | 0.248 | 0.181 | 0.211 | 0.186 | 0.263 | 0.200 |
| moe | 0.200 | 0.232 | 0.175 | 0.197 | 0.223 | 0.225 | 0.181 | 0.205 | 0.177 | 0.262 | 0.250 |

RMSE (N·m/kg) per mode on held-out subjects.

Excluded test trials (ankle goniometer beyond ±60°, sensor fault): AB30_stair_3_l_01_01, AB30_stair_3_l_01_03

MoE gate weights by mode (moe):

| mode | expert 0 | expert 1 | expert 2 | expert 3 |
|---|---|---|---|---|
| treadmill | 0.21 | 0.38 | 0.26 | 0.16 |
| walk | 0.21 | 0.38 | 0.24 | 0.16 |
| turn | 0.22 | 0.36 | 0.27 | 0.16 |
| rampascent | 0.19 | 0.28 | 0.37 | 0.16 |
| rampdescent | 0.23 | 0.35 | 0.25 | 0.17 |
| stairascent | 0.20 | 0.30 | 0.33 | 0.17 |
| stairdescent | 0.27 | 0.33 | 0.23 | 0.17 |
| transition | 0.23 | 0.32 | 0.27 | 0.18 |
| stand | 0.28 | 0.25 | 0.24 | 0.23 |
