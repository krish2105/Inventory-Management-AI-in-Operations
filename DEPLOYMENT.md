# Deploying the dashboard on Streamlit Community Cloud (free)

1. **Repository** — code is on GitHub: `https://github.com/krish2105/Inventory-Management-AI-in-Operations` (branch `main`).
   Required files are at the repo root: `requirements.txt`, `.streamlit/config.toml`, `data/processed/sku_daily_demand.csv`.
2. Go to **https://share.streamlit.io** → *Sign in with GitHub* (account `krish2105`) → authorise Streamlit.
3. Click **Create app** → **Deploy a public app from GitHub**.
4. Fill in:
   - Repository: `krish2105/Inventory-Management-AI-in-Operations`
   - Branch: `main`
   - Main file path: `apps/streamlit/app.py`
   - App URL: e.g. `krishna-ai-inventory` (→ `https://krishna-ai-inventory.streamlit.app`)
5. **Advanced settings** → Python version **3.12** (or 3.13). No secrets are needed.
6. Click **Deploy**. The first build installs the packages (≈3–5 min). The first page load computes forecasts for 40 SKUs (≈10 s); later loads are cached.
7. **Sharing with the professor:** app *Settings → Sharing*. To follow the "no sharing with peers" rule, set the app to **Only specific people can view** and invite the instructor's email, or keep it public (the dataset is public UCI data; nothing private is stored). Send the `.streamlit.app` link.
8. **Updating:** every `git push` to `main` redeploys automatically.

Troubleshooting: *Manage app → Logs* shows build errors. If a package fails to install, choose Python 3.12 and *Reboot app*. If the app sleeps after inactivity, click *Yes, get this app back up*.

## Google Colab (notebook)
Upload `submissions/Laboratory Work 1/Krishna_Mathur_AS25DXB018_AI_in_Operations_Laboratory_Work_1.ipynb` to Colab (File → Upload notebook) → *Runtime → Run all* (the first cell clones this repo automatically) → *Share* → restrict to the instructor → paste the link into `Google_Colab_Link.txt`.
