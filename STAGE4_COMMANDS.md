# Stage 4 - commands (PowerShell, inside the repo folder)

git pull
python -m pytest -q                      # 24 passed
$env:PYTHONPATH="src"; python -m mp125.monitor

git add .
git commit -m "Stage 4: drift, per-group MAE, retrain v2, promotion, holdout predictions"
git push
