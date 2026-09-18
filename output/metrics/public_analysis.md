# SmartDoc public-sample analysis

The primary image metric is comparison with the same-frame `dewarped` reference. The `ground-truth` comparison is secondary because it also contains blur, illumination, and source-content differences.

| sample | target W×H | corner RMSE (px) | dewarped PSNR (dB) | ground-truth PSNR (dB) | valid ratio |
|---|---:|---:|---:|---:|---:|
| smartdoc_card01 | 638×1010 | 2.807e-13 | 39.45 | 11.83 | 1.0000 |
| smartdoc_paper01 | 2480×3508 | 1.218e-12 | 34.44 | 11.33 | 1.0000 |
| smartdoc_poster01 | 2167×3072 | 2.118e-12 | 33.48 | 7.77 | 1.0000 |
| smartdoc_receipt01 | 1797×5770 | 2.141e-12 | 31.35 | 8.79 | 1.0000 |
| smartdoc_screen01 | 3000×2250 | 1.102e-12 | 36.80 | 8.54 | 1.0000 |

Interpretation: corner RMSE at machine precision validates the four-point homography fit. Dewarped PSNR measures implementation consistency, while ground-truth PSNR should not be used alone to attribute error to the rectification algorithm.
