# Frozen EaTR parent for DGQC transfer

- Source run: `artifacts/canonical_b128_restart/eatr/seed2023/eatr`
- Source artifact at freeze time: `best_map.pt`
- Frozen checkpoint: `eatr_plain_b128_best_map.pt`
- SHA256: `301c68139d32daa84421a345b227bf74195ee5940111b5ecb388bf1824dcf3bc`
- Validation snapshot: `mAP=8.73`, `G-mIoU@3=3.51`, `mR@5=12.85`, `mR+@5=1.37`
- Freeze time: 2026-07-22 22:51 Asia/Shanghai

This is an immutable parent snapshot for a matched cross-backbone transfer study. The live
canonical b128 EaTR training may continue improving after this point; all children in this study
must still use this exact checkpoint so their comparison remains attributable.
