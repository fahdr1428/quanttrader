# Deploying Quant Lab Pro

This guide covers running Quant Lab Pro on your own computer and publishing it as a public research demo
on **Streamlit Community Cloud** (free). Nothing in this repository has been deployed yet.

---

## 1. Before you publish: checklist

Work through this list before making the GitHub repository public or deploying it.

- [ ] **Review every file** in the repository (`git ls-files`). Nothing should contain credentials,
      personal data, machine-specific paths, cached market data or databases. `.gitignore` already
      excludes `.venv/`, `data/*` (except `.gitkeep`), `*.db`, `.env*` and `.streamlit/secrets.toml`.
- [ ] **Run the tests and the app locally** (see the README) and confirm live Yahoo Finance downloads
      work from your network — this build was verified in a sandbox without market-data access.
- [ ] **Replace the screenshots** in `docs/screenshots/` with screenshots of real data (they were
      captured on a synthetic test file) and update the captions in the README.
- [ ] **Choose a licence** (for example MIT or Apache-2.0), add a `LICENSE` file, and declare it in
      `pyproject.toml`. Without one, others may view but not legally reuse the code.
- [ ] Keep `QLP_STORAGE` **unset** (or `session`) on the public deployment, so each visitor's
      experiments stay private to their own browser session.
- [ ] Decide whether to keep the default first-visit auto-load of SPY (it costs one Yahoo request per new
      visitor when caches are cold).
- [ ] Check the **About** text in `app.py` and the disclaimer in `ui/components.py` read the way you want.

---

## 2. Local development (macOS)

```bash
git clone https://github.com/fahdr1428/quanttrader.git
cd quanttrader
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
python -m pytest                       # all tests should pass
QLP_STORAGE=sqlite streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with `Ctrl+C`. Leave the virtual environment with
`deactivate`.

Configuration is through environment variables only; there are no secrets.

| Variable | Default | Purpose |
|---|---|---|
| `QLP_STORAGE` | `session` | `session` = per-browser in-memory records; `sqlite` = persistent local file |
| `QLP_DATA_DIR` | `<repo>/data` | Where the price cache and the SQLite database are written |

---

## 3. Publish the code on GitHub

The repository already exists at `github.com/fahdr1428/quanttrader` (private). When you have completed
the checklist above:

1. Merge the development branch into your default branch (via a pull request).
2. On GitHub: **Settings → General → Danger Zone → Change repository visibility → Public** — only when
   you are ready. Streamlit Community Cloud can also deploy from a private repository if you grant it
   access, which lets you test the deployment first.

---

## 4. Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with your GitHub account and authorise
   Streamlit to access the repository.
2. Click **Create app → Deploy a public app from GitHub** (wording may differ slightly over time).
3. Fill in:
   - **Repository:** `fahdr1428/quanttrader`
   - **Branch:** your default branch (e.g. `main`)
   - **Main file path:** `app.py`
   - **App URL:** choose a subdomain, e.g. `quant-lab-pro`
4. Open **Advanced settings**:
   - **Python version:** 3.12 or 3.13 (the project requires ≥ 3.11; it was tested on 3.13).
   - **Secrets:** leave empty — the app needs none.
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically and starts `app.py`. The first
   build takes a few minutes.

**What the platform uses from the repository**

| File | Role |
|---|---|
| `app.py` | Entry point (multipage navigation is defined inside it) |
| `requirements.txt` | Python dependencies, installed with pip |
| `.streamlit/config.toml` | Theme, upload limit (20 MB), telemetry off |
| `pages/`, `ui/`, `quantlab/` | Application code |

`requirements-lock.txt` is not used by the platform; it records the exact versions tested locally.

### Verify the deployment

1. Open the app URL. The Overview should load SPY automatically, or show a clear *"Market data
   unavailable"* message if Yahoo Finance blocks the server (see troubleshooting).
2. Check the context strip shows the data source, last observation and cache status.
3. Open Backtest → *Accounting reconciliation*: all seven checks should pass.
4. Save an experiment on Notes, open the app in a private window, and confirm the experiment is **not**
   visible there (session-private storage).

---

## 5. Free-tier constraints

Streamlit Community Cloud limits change over time; check the current documentation. Practically:

- **Limited memory and CPU per app.** The app keeps memory modest (daily data only, bounded caches,
  ≤150 sensitivity and ≤100 walk-forward combinations). Large walk-forward grids on 20+ years of data take
  several seconds.
- **Apps sleep after a period without traffic** and take a short while to wake. The first visitor after
  a sleep triggers a cold start and fresh data downloads.
- **Ephemeral disk.** The on-disk price cache is lost on restart; data is simply re-downloaded. Never
  rely on `QLP_STORAGE=sqlite` on the cloud: the file would be shared by all visitors and lost on restart.
- **Shared outbound IP addresses.** Yahoo Finance frequently rate-limits or blocks requests from cloud
  providers. The app handles this gracefully (cached data is used and labelled stale, or a clear error is
  shown, with CSV upload as a fallback), but live downloads on the hosted demo may be intermittent.
- **No availability guarantee.** Present the deployment as a research demo, not a service.

---

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| *"Could not reach Yahoo Finance (network error or blocked connection)"* | The server cannot reach Yahoo (firewall, proxy, or Yahoo blocking the host's IP range). Retry later, or upload a CSV via **Data source → CSV file**. |
| *"Yahoo Finance is rate-limiting requests"* | Too many requests from the same IP. Wait a few minutes; previously cached data is used and labelled *stale*. |
| *"No price history was found for 'XYZ'"* | Unknown or delisted symbol, or no trading in the chosen dates. Yahoo uses `BRK-B` rather than `BRK.B` and `^GSPC` for indices. |
| *"The loaded data failed validation"* | Too few usable observations (fewer than 30) or malformed columns. See the **Data** page for the full report. |
| Build fails installing dependencies | Select Python 3.12/3.13 in Advanced settings; reboot the app from the dashboard to rebuild. |
| App shows *"This app has encountered an error"* | Open **Manage app → Logs** in the Streamlit dashboard; please file an issue with the trace. |
| Saved experiments disappeared | Expected in `session` storage when the browser session ends. Export JSON from **Notes → Import / export** to keep them. |
| Walk-forward says "not enough for one fold" | Shorten the training/test windows or extend the date range. |

---

## 7. Security and privacy notes

- No API keys, credentials or secrets are required, read or stored. Free-text fields that look like
  credentials are rejected.
- No broker connectivity, order placement, leverage or live trading exists in the code.
- Usage telemetry is disabled in `.streamlit/config.toml`; the app sends no analytics.
- Public deployments store experiments only in each visitor's own in-memory session.
- Uploaded CSV files are processed in memory and are not written to disk.
