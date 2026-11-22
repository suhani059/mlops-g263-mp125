# Stage 2 - run on your own machine (PowerShell, inside the repo folder)

pip install -r requirements.txt
$env:PYTHONPATH="src"; python -m pytest -q
$env:PYTHONPATH="src"; python -m mp125.train

git add .
git commit -m "Stage 2: cleaning, MLflow training, registry, tests"
git push
