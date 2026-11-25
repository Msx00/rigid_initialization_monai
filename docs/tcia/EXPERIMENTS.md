# Reference experiment

The TCIA profile was selected by patient-equal validation Target TRE across 30 combinations: PCA gain 0, 0.025, 0.05, 0.075 or 0.1; smoothing 0, 5 or 10; 4000 surface samples; 150 or 200 ICP iterations. All 30 ran on the same 124 validation pairs. The selected setting was 0 / 0 / 4000 / 150. The previous 0.1 / 0 / 7000 / 100 setting gave 7.49 mm on the same validation patients. Increasing the selected iteration limit to 200 gave 7.42 mm and Dice 0.8306.

| Split | Pairs | Patients with matched Targets | Patient-equal TRE before → after (mm) | Mean pair Dice before → after | Mean ASSD after (mm) | Mean HD95 after (mm) |
|---|---:|---:|---:|---:|---:|---:|
| Train | 747 | 498 | 46.08 → 7.35 | 0.0709 → 0.8263 | 2.27 | 6.12 |
| Validation | 124 | 82 | 41.04 → 7.32 | 0.0875 → 0.8293 | 2.26 | 5.99 |
| Test | 373 | 248 | 46.30 → 7.47 | 0.0847 → 0.8233 | 2.33 | 6.23 |
| All | 1244 | 828 | 45.65 → 7.38 | 0.0767 → 0.8257 | 2.28 | 6.14 |

The splits contain 498, 83 and 249 distinct patients. In total 1241 pairs have 1588 matching Targets. All pairs contribute gland metrics, including cases with rotation or source review flags. Before registration, the transform is identity in the native physical coordinate systems.

**Target TRE** measures the distance between native ROI volume centroids of corresponding Targets after mapping the MR point to US. Patient-equal TRE averages Targets within each pair, pairs within each patient, and then patients. The fitted gland centroid is evaluated separately. Test median patient TRE was 6.13 mm; 61 of 248 patients exceeded 10 mm.

**Gland Dice** uses a full-support grid with the output grid spacing and phase so that distant native glands are included in the baseline. **ASSD** averages the two directed mean surface distances on the output grid. **HD95** is the 95th percentile of the pooled bidirectional surface distances within each pair; the table averages these per-pair values.
