# GitHub Publishing Guide

These steps assume Windows PowerShell and a verified local OpsMind v1.0.2 checkout.

## 1. Verify the project

```powershell
python -m pip install -e ".[dev]"
.\scripts\verify-local.ps1
```

Start the API once and confirm:

```powershell
opsmind serve
```

In a second PowerShell:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/health"
```

Expected version: `1.0.2`.

## 2. Initialize Git

```powershell
git init
git branch -M main
git add .
```

## 3. Run the repository preflight

```powershell
.\scripts\github-preflight.ps1
```

The script fails if staged files include local secrets, model weights, generated reports, vector-store persistence, updater backups, or unexpectedly large files.

Review manually too:

```powershell
git status
```

You should not see `.env`, `.venv`, `*.pt`, `*.joblib`, `*.safetensors`, `data/vector_store/`, runtime JSON files under `data/output/`, or updater backup directories staged for commit.

## 4. Commit

```powershell
git commit -m "feat: release OpsMind AI v1.0.2"
```

## 5. Create the GitHub repository

Recommended repository name: `opsmind-ai`.

Create an empty public repository on GitHub. Do not initialize it with another README or `.gitignore`, because those files already exist locally.

Then add the remote GitHub gives you:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/opsmind-ai.git
git push -u origin main
```

## 6. Verify CI

Open the repository's **Actions** tab. The `ci` workflow should run for Python 3.11 and 3.13. Both matrix jobs should pass:

```text
Install
↓
pip check
↓
compileall
↓
pytest
```

Do not describe the project as having CD unless an automated deployment workflow is added later.
