# How to Upload This Repo to GitHub

## Step 1 — Create a GitHub account and repository

1. Go to https://github.com and sign in (or create an account)
2. Click **New repository** (+ icon, top right)
3. Fill in:
   - Repository name: `kma-classifier`
   - Description: `Automated extraction and classification of policy commitments from environmental policy documents`
   - Visibility: **Private** (recommended — contains research code)
   - Do NOT initialise with README (you already have one)
4. Click **Create repository**
5. Copy the repository URL shown (e.g. `https://github.com/yourusername/kma-classifier.git`)

---

## Step 2 — Install Git on your machine

**Ubuntu/Debian:**
```bash
sudo apt install git
```

**macOS:**
```bash
brew install git
```

**Windows:** Download from https://git-scm.com/download/win

---

## Step 3 — Configure Git (first time only)

```bash
git config --global user.name "Muhammad Yasar Khan"
git config --global user.email "yasar.khan@universityofgalway.ie"
```

---

## Step 4 — Initialise and push the repo

```bash
# Navigate to the repo folder
cd /path/to/kma-classifier

# Initialise git
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: KMA classifier pipeline v1.0"

# Link to GitHub
git remote add origin https://github.com/yourusername/kma-classifier.git

# Push to GitHub
git branch -M main
git push -u origin main
```

GitHub will ask for your username and password. For password use a
**Personal Access Token** (not your account password):
1. GitHub → Settings → Developer Settings → Personal Access Tokens → Tokens (classic)
2. Generate new token → select `repo` scope → copy the token
3. Use it as the password when prompted

---

## Step 5 — Verify on GitHub

Go to `https://github.com/yourusername/kma-classifier` and confirm all files are there.

---

## Step 6 — Pushing future changes

After making changes to any file:

```bash
git add .
git commit -m "Brief description of what changed"
git push
```

---

## Step 7 — On the GPU server

To keep the server code in sync with GitHub:

```bash
# First time — clone the repo to the server
ssh yaskha@140.203.155.230
git clone https://github.com/yourusername/kma-classifier.git ~/files/kma-classifier

# Future updates — pull latest code
cd ~/files/kma-classifier
git pull
```

---

## What NOT to commit

The `.gitignore` already excludes:
- `*.pt` files (trained model weights — too large, use Git LFS or store separately)
- `*.pkl` files (UMAP reducer)
- `exported_files/` (PDF extractions with policy content)
- `annotated_excel_file/` (reviewed annotations)
- `training_data.csv` (generated data)

### Storing model files separately

Option A — **Git LFS** (for files up to 2GB):
```bash
git lfs install
git lfs track "*.pt" "*.pkl"
git add .gitattributes
git add level_classifier.pt class_classifier.pt umap_reducer.pkl
git commit -m "Add trained model files"
git push
```

Option B — **Zenodo** (for academic sharing):
Upload model files to https://zenodo.org with a DOI for citation.

---

## Adding a License

```bash
# MIT License is recommended for academic open-source
curl -o LICENSE https://raw.githubusercontent.com/nicholasgasior/template-mit/master/LICENSE
# Edit LICENSE to add your name and year
git add LICENSE
git commit -m "Add MIT License"
git push
```

---

## Making the repo public (when ready to publish)

1. GitHub → repository → Settings → Danger Zone → Change visibility → Public
2. Add a proper description and topics: `nlp`, `policy-analysis`, `peatlands`, `classification`
3. Pin it to your GitHub profile
