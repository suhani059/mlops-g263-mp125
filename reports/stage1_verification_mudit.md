\# Stage 1 Data Verification Report



\*\*Name:\*\* Mudit Punetha  

\*\*Group:\*\* G263  

\*\*Problem:\*\* MP125 - City Bus Delay (Patna)



\## Objective

To independently verify the seven data-quality findings reported in findings.yaml using train.csv and holdout\_features.csv.



\## Verification Results



| Data Issue | Observation | Status |

|---|---|---|

| Sentinel Value | 55 rows contain 9999 in the hour column. | Confirmed |

| Duplicate Rows | 0 duplicate rows after excluding record\_id. | Confirmed |

| Leaking Column | Only delay\_min is absent from holdout\_features.csv. | Basic check passed |

| Spelling Variants | Weather values represent three categories: clear, fog, rain. | Confirmed |

| Unit Mixup | Depot-zone medians show no obvious unit-scale differences. | No obvious issue |

| Stuck Sensor | driver\_exp\_yr remains 5 across 126 rows between 2026-03-29 and 2026-04-19. | Confirmed for this period |

| Target Typos | 12 unusually large delay values, ranging from 137 to 282, appear above a gap in the target distribution. | Suspected 10x errors |



\## Conclusion

The seven findings in findings.yaml are broadly consistent with the independent checks performed. The unit-mixup and target-typo findings would benefit from additional validation before treating them as definitive.

