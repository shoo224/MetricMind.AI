# MetricMind AI — Intelligent Metrics Anomaly Detection and Root-Cause Analysis

**Milestone 2 (M2):** a working, tested, reproducible anomaly-detection pipeline on real server
metrics, built with a Clean/Hexagonal architecture. M2 = ingestion → timestamp alignment →
validation → EDA → Isolation Forest → evaluation against ground truth → figures.

> Per-component explanations and viva preparation: **[docs/VIVA_GUIDE.md](docs/VIVA_GUIDE.md)**

---

## 1. Project overview
MetricMind AI watches several performance metrics of a server (CPU, memory, network response
times), learns what normal behaviour looks like, and flags time steps that deviate. Later
milestones add deep-learning detection, root-cause analysis, an API and a dashboard.

## 2. Problem statement
Operations teams cannot watch dashboards around the clock, and fixed thresholds ("alert if CPU >
80%") miss faults that are abnormal only in *combination* or *context*, while flooding people with
false alarms on harmless spikes. We need a data-driven detector that learns normal behaviour from
history without requiring labelled anomalies (labels are rare and expensive in production).

## 3. Objectives
1. Ingest separate per-metric time-series files and align them on timestamp into one multivariate dataset.
2. Validate and clean the data without silently hiding problems.
3. Explore the data (distributions, correlations, time-series plots).
4. Detect anomalies with an unsupervised model (Isolation Forest) and score each time step.
5. Evaluate honestly against the dataset's labelled fault events.
6. Persist model, scaler, data and results; make every step reproducible from the CLI.
7. Keep the design extensible (LSTM autoencoder, root-cause analysis, API, database, dashboard).

## 4. Architecture
Clean/Hexagonal: business logic in the centre, technology at the edges, dependencies point inward.

```
interfaces (CLI; future FastAPI)  ──►  application (use cases, services)  ──►  domain (entities, ports)
                                                                                   ▲
infrastructure (pandas CSV, scikit-learn, joblib, matplotlib) ── implements ───────┘
```

| Layer | Contains | May import |
|---|---|---|
| `domain` | `Metric`, `MetricSeries`, `LabeledEvent`, `Anomaly`, `AnomalyResult`, `ChronologicalSplit`, `EvaluationReport`; **ports** (`MetricDataLoader`, `AnomalyDetector`, `FeatureScaler`, `ModelRepository`, …) | stdlib + numpy only |
| `application` | Services (validate, merge, impute, label, split, preprocess, evaluate) and use cases (prepare, EDA, train, evaluate) | domain |
| `infrastructure` | `CSVMetricDataLoader`, `IsolationForestDetector`, `SklearnStandardScaler`, `JoblibModelRepository`, `CsvJsonStore`, `MatplotlibFigureRenderer` | domain, application ports |
| `interfaces` | CLI commands and the **composition root** (`wiring.py`) | everything |

These rules are enforced by an automated test (`tests/unit/test_architecture.py`).

**Key decisions**
* *Ports & adapters.* The use cases only know `AnomalyDetector`, not scikit-learn. An LSTM
  autoencoder is a new adapter plus one line in `wiring.py`.
* *Domain uses numpy, not pandas.* numpy arrays are the neutral numeric interchange type; pandas is
  confined to application/infrastructure (the application layer uses DataFrames as its tabular type).
* *Dataset facts live in `configs/metrics.toml`,* not in code. A different dataset needs a different file, not different code.
* *Composition root.* Only `interfaces/cli/wiring.py` chooses concrete classes.
* *No premature abstraction.* `infrastructure/monitoring` and `interfaces/api` are empty
  placeholders for M3+; nothing meaningless was added to fill them.

## 5. Folder structure
```
metricmind-ai/
├── src/metricmind/{domain,application,infrastructure,interfaces,config}/  main.py
├── configs/metrics.toml        dataset description (files, sentinel, sampling, event logs)
├── data/{raw,processed,external}/   raw CSVs in, processed CSVs out
├── models/                     scaler.joblib, isolation_forest.joblib, model_metadata.json
├── reports/{figures,results,logs}/
├── scripts/                    prepare_data.py  run_eda.py  train_model.py  evaluate_model.py
├── tests/{unit,integration,fixtures}/
├── docs/VIVA_GUIDE.md          .env.example  pyproject.toml  docker-compose.yml (placeholder)
```

## 6. Dataset requirements (verified by inspecting the supplied files)
Place the files so that `data/raw/` contains:
```
cpu_usage.csv  memory_usage.csv  memory_used.csv
network_response_time_a.csv  network_response_time_b.csv  network_response_time_diff.csv
event_logs/event_logs_{cpu,memory,network}.csv
```
What the data actually is:
* 6 metric files, header `Time,Value`, **40,320 rows each**, **15 s sampling**, one week
  (2025-12-11 01:00:00 → 2025-12-18 00:59:45). Timestamps are identical across files; no duplicates, no gaps.
* **Sentinel `-99999`** marks a failed probe: 1 value in `response_time_a`, 12 in `_b`, 13 in `_diff`
  (`diff = a − b` everywhere else). They are treated as missing.
* **`memory_used` ≈ 77 × `memory_usage`** (correlation 1.000) and **`response_time_diff = a − b`** → redundant; kept for EDA, excluded from the model.
* **Labels = 42 injected fault events** (cpu 13, memory 12, network 17) with start/end timestamps
  (UTC; `start_point`/`end_point` are row indices and agree with the timestamps to within 7.4 s).
  Events never overlap. Labelled rows: 8,799 of 40,320 (**21.8 %**).
* Units are not documented by the dataset; `configs/metrics.toml` marks them "(inferred)".

## 7. Setup
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                      # optional; defaults work
```
Python ≥ 3.11. Works on a normal laptop (full pipeline ≈ 12 s here).

## 8. Configuration
`.env` / environment variables (see `.env.example`): `DATA_RAW_DIR`, `DATA_PROCESSED_DIR`,
`MODEL_DIR`, `REPORT_DIR`, `TRAIN_FRACTION`, `MAX_FFILL_STEPS`, `DUPLICATE_POLICY`,
`MODEL_FEATURES`, `IF_N_ESTIMATORS`, `IF_CONTAMINATION`, `IF_MAX_SAMPLES`, `IF_RANDOM_STATE`,
`LOG_LEVEL`. Relative paths resolve against the project root; nothing is machine-specific.

## 9–11. Run the pipeline
```bash
python scripts/prepare_data.py     # 1. load → validate → merge → impute → label → chronological split → save
python scripts/run_eda.py          # 2. statistics + 4 EDA figures
python scripts/train_model.py      # 3. fit scaler + Isolation Forest on TRAIN rows → save to models/
python scripts/evaluate_model.py   # 4. load model → score TEST rows → metrics + figures
```
Equivalent: `metricmind prepare|eda|train|evaluate`. Add `-v` to show INFO logs on the console
(the log file `reports/logs/metricmind.log` always records INFO).

## 12. Expected outputs
| Path | Content |
|---|---|
| `data/processed/metrics.csv, train.csv, test.csv` | merged cleaned dataset + `is_anomaly`, `anomaly_category`, `event_id` |
| `data/processed/dataset_summary.json` | row counts, validation, missing-value and label statistics |
| `models/*.joblib, model_metadata.json` | scaler, detector, features, hyper-parameters, threshold, library versions |
| `reports/results/evaluation_metrics.json` | all metrics + confusion matrix + sensitivity table |
| `reports/results/test_predictions.csv` | per-row score, prediction, truth |
| `reports/figures/eda_*.png` | CPU, memory, response times, correlation heatmap |
| `reports/figures/{confusion_matrix,anomaly_scores,anomalies_*}.png` | evaluation figures |

## 13. Algorithms
* **Forward-fill imputation (≤ 4 steps):** uses only past values, so no future leakage.
* **StandardScaler:** `z = (x − μ)/σ`, fitted on training rows only.
* **Isolation Forest (scikit-learn):** builds many random trees that split the feature space with random
  cut points. Rare, different points are isolated in fewer splits (short average path). The anomaly
  score is a normalised path length; **higher = more anomalous** (we flip scikit-learn's sign). The
  threshold is the `1 − contamination` quantile of training scores. Unsupervised: **labels are never used for training.**
* Model features: `cpu_usage, memory_usage, response_time_a, response_time_b` (redundant columns excluded).

## 14. Evaluation metrics
Positive class = anomaly. **Precision** = TP/(TP+FP); **Recall** = TP/(TP+FN); **F1** = harmonic
mean. Accuracy is reported but not primary (21.8 % of rows are anomalous, so "always normal" already scores 78.2 %).
Also: confusion matrix, ROC-AUC and average precision (threshold-free), **event-level recall**
(did we flag ≥ 1 row of each fault?) and **recall per fault category**. Metrics are implemented
from the confusion matrix and cross-checked against scikit-learn in the unit tests.

## 15. M2 implementation status and actual results
All M2 requirements are implemented. Results on the held-out **test set** (the last 30 % of the week,
12,096 rows, 2025-12-15 22:36 → 2025-12-18 00:59), model trained on the first 70 % with default
hyper-parameters (no tuning on test labels):

| Metric | Value |
|---|---|
| Rows flagged | 458 (3.8 %) vs 2,641 labelled (21.8 %) |
| Confusion matrix [[TN, FP],[FN, TP]] | [[9425, 30], [2213, 428]] |
| **Precision** | **0.934** |
| **Recall** | **0.162** |
| **F1** | **0.276** |
| Accuracy | 0.815 (trivial all-normal baseline: 0.782) |
| ROC-AUC / Average precision | 0.747 / 0.573 |
| Events detected (lenient: ≥ 1 flagged row inside the window) | 10 / 10 |
| Recall by category | cpu 0.29, memory 0.09, network 0.16 |

**How to read this honestly.** The model is *conservative*: when it raises an alarm it is almost
always right (precision 0.93), but it misses most labelled rows. Reasons, verified in the data:
1. Labels are **fault windows**, not individual anomalous points. Many rows inside a window look normal
   (network windows are only ~35–62 % abnormal; some windows show no visible effect — see `reports/figures/anomalies_*.png`).
2. Faults are ~22 % of the *training* data, so elevated values are not rare there; Isolation Forest assumes anomalies are rare.
3. `contamination = 0.05` flags ≈ 5 % of rows by design. `reports/results/flag_rate_sensitivity.csv`
   shows the trade-off (flag 20 % → precision 0.53, recall 0.49). That table uses test labels, so it is **analysis only**, not a tuning step.
4. Clearly extreme values *are* caught: 100 % of test rows with memory > 40 %, 91 % with response_time_b > 3000.
5. Natural spikes outside any window are labelled "normal", which costs precision in principle; here only 30 false positives occurred.

## 16. Limitations and future work
* Point-wise model: no temporal context (a slow memory leak is only seen once the level is high). → **LSTM autoencoder** (M3) models sequences.
* Unsupervised training on contaminated data. → train on normal-only periods, or tune `contamination` on a **validation** split.
* One week of data, one server; interval-level labels only; units undocumented.
* No root-cause analysis yet (which metric drove an alarm) → M3: per-feature contribution / reconstruction error.
* Not real-time; the architecture supports it (new `MetricDataLoader` adapter for a stream, FastAPI in `interfaces/api`, TimescaleDB as a new `TabularStore`, Docker via the placeholder compose file).

## 17. Testing
```bash
pytest                                  # 57 tests (unit + integration), synthetic data only
pytest --cov=metricmind --cov-report=term-missing
```
Covers CSV loading/validation errors, timestamp parsing, merging (gaps, duplicates, unsorted),
missing values (causality), labelling, chronological splitting, scaler leakage, detector behaviour and
persistence round-trip, metrics vs scikit-learn, architecture rules, an end-to-end synthetic pipeline,
and CLI error handling. Tests were also *mutation-checked*: deliberately introducing bugs (backward
fill, swapped split, flipped score sign, fitting the scaler on test data, …) makes specific tests fail.

## 18. Development commands
```bash
ruff check .          # lint
ruff format .         # format
pytest -q             # tests
metricmind train -v   # run a step with INFO logs
```
