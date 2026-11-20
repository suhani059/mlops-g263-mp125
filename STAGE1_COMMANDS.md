# Stage 1 - commands to run on your own machine

```bash
# 1. Create a PUBLIC repo on GitHub, add all 4 members as collaborators, then:
git clone https://github.com/<you>/<repo>.git && cd <repo>
# copy the contents of G263_MP125/ into the repo (do NOT open CSVs in Excel)

# 2. DVC
pip install dvc
git init  # if not cloned
dvc init
dvc add data/raw/train.csv data/raw/later.csv
# holdout_features.csv is committed directly (the handbook layout lists it without .dvc)
git add .dvc data/raw/*.dvc data/raw/.gitignore data/raw/holdout_features.csv \
        README.md params.yaml requirements.txt src reports .gitignore
git commit -m "Stage 1: DVC, data checks, findings"
git push

# 3. Verify the hash matches your original file
sha256sum data/raw/train.csv   # bd0f491c45e2a1685d72d951309615bf0642b6dac6f1462fe9bb9640b5ca7063
```
Each member should make at least one commit from their own GitHub account.
