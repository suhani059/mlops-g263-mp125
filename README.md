# MP125 - City Bus Delay (Patna)

Group: **G263**  |  Problem: **MP125**  |  Faculty: Yadit Kumar  |  Batch: 6

| Enrolment no. | Name |
|---|---|
| S24CSEU0623 | SUBHAMKUMAR |
| S24CSEU0567 | SuhaniVij |
| S24CSEU0644 | Karanveer |
| S24CSEU0597 | MuditPunetha |

Predict `delay_min` (regression, MAE) for Patna City Bus Service trips.

## Stage 1: data checks
```
pip install -r requirements.txt
dvc pull                      # or copy the zip's CSVs into data/raw/
PYTHONPATH=src python -m mp125.run_checks
```
Results are written to `reports/data_checks.json` and copied into `reports/findings.yaml`.
