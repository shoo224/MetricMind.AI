# Viva Guide — MetricMind AI (M2)

Each block: **What / Why here (layer) / Interacts with / Say in the viva**.

## A. Components

### Domain entities (`domain/entities`, `domain/value_objects`)
* **What:** `Metric` (what is measured), `MetricSeries` (its timestamps + values), `LabeledEvent` (a known fault interval), `Anomaly`/`AnomalyResult` (detector output), `ChronologicalSplit` (the train/test rule), `EvaluationReport` (metrics).
* **Why domain:** they are business concepts independent of any library; if pandas disappears they stay.
* **Interacts:** produced by infrastructure loaders, consumed by application services.
* **Say:** "Entities describe the problem, not the technology. That is why they use no pandas or sklearn, and an architecture test enforces it."

### Ports (`domain/ports`)
* **What:** interfaces — `MetricDataLoader`, `EventLabelLoader`, `AnomalyDetector`, `FeatureScaler`, `ModelRepository`.
* **Why:** dependency inversion — the application depends on the *abstraction*; infrastructure implements it.
* **Say:** "To add an LSTM autoencoder I write one class implementing `AnomalyDetector` and change one line in `wiring.py`. No use case changes."

### CSVMetricDataLoader (`infrastructure/data`)
* **What:** reads a metric CSV, checks columns, parses timestamps *strictly*, rejects non-numeric values and empty files, converts the `-99999` failed-probe sentinel to NaN.
* **Why infrastructure:** it is the only code that knows about CSV/pandas.
* **Say:** "Bad data raises a clear `DataValidationError`; it is never silently coerced. Duplicates/gaps are *reported* by the validator, not hidden here."

### SeriesValidator + MetricMerger (`application/services`)
* **What:** validator detects duplicates, unsorted data, missing values, wrong sampling interval. Merger outer-joins all series on timestamp, sorts, inserts explicit NaN rows for missing grid timestamps, applies the duplicate policy (`error` by default).
* **Why:** the six metrics live in six files; a multivariate model needs them in one table aligned in time. Misaligned rows would compare CPU at 10:00 with memory at 10:05 and corrupt every conclusion.
* **Say:** "Outer join so nothing is dropped; gaps become visible NaN rows; strategy is documented and deterministic."

### MissingValueHandler
* **What:** forward-fill up to 4 steps (1 min), drop rows still missing.
* **Why forward-fill, not interpolation:** interpolation uses *future* values → leakage. Forward-fill is causal. (Real data: 26 sentinel values, all repaired.)

### GroundTruthLabeler
* **What:** turns 42 fault intervals into per-row `is_anomaly`, `anomaly_category`, `event_id` by timestamp (inclusive bounds; event logs are UTC like the metrics).
* **Say:** "I verified that `start_point` row indices and `start_time` timestamps agree within 7.4 s, and the plots show big bursts inside shaded windows. No labels are invented."

### ChronologicalSplitter
* **Why not random split:** shuffling lets the model train on the future and test on the past → over-optimistic results and unrealistic for monitoring. First 70 % trains, last 30 % tests; an assertion checks no overlap.

### FeaturePreprocessor + StandardScaler
* **What:** selects model features and standardises them; **scaler fitted on train only**, saved, reused at evaluation.
* **Honest nuance:** Isolation Forest is largely insensitive to feature scale (random cut points between min/max). Scaling is kept because the pipeline is shared with the planned LSTM autoencoder, which *does* need it.
* **Why no clipping:** extreme values are the anomalies; clipping would hide them.
* **Why 4 features:** `memory_used`≈77×`memory_usage` and `diff = a−b` are redundant; including them would double-weight the same signal.

### IsolationForestDetector (`infrastructure/ml`)
* **What:** adapter over scikit-learn; `fit`, `score_samples` (higher = more anomalous), `predict` (1 = anomaly), `threshold`, configurable hyper-parameters.
* **Why Isolation Forest:** unsupervised (no labels needed), fast, handles multivariate data, no distributional assumptions, strong baseline. Weakness: no temporal context.

### AnomalyEvaluator
* **What:** precision/recall/F1/confusion matrix from first principles; ROC-AUC, average precision; event-level and per-category recall.
* **Say:** "Unit tests prove my numbers equal scikit-learn's."

### Use cases + composition root
* `PrepareData`, `RunEda`, `TrainModel`, `EvaluateModel` orchestrate services; `wiring.py` is the only place concrete classes are chosen; CLI files are thin.

## B. Viva questions

1. **Why anomaly detection?** Manual thresholds miss multivariate/contextual faults and cause alert fatigue; labels are scarce so unsupervised learning is practical.
2. **Why this dataset?** Real multi-metric server telemetry at 15 s resolution with *known injected faults* → ground truth for honest evaluation.
3. **What does each metric mean?** `cpu_usage` CPU load; `memory_usage` % of RAM; `memory_used` absolute memory (≈ proportional); `response_time_a/b` network latency to two probe targets; `diff` = a − b. (Units are not documented — inferred.)
4. **Why timestamp alignment?** Metrics are in separate files; a row must describe one instant across all metrics.
5. **Why multivariate?** Several measurements per time step; faults often show as a *combination* (e.g. CPU and latency together).
6. **Why Isolation Forest?** See above.
7. **Why chronological split?** Avoids future leakage; mimics deployment (train on past, detect on future).
8. **What is the anomaly score?** Normalised average path length to isolate a point, sign-flipped so higher = more anomalous; flagged if above a threshold set by `contamination`.
9. **How are anomalies evaluated?** Against fault-window labels: precision, recall, F1, confusion matrix, plus ROC-AUC/AP, event recall, per-category recall.
10. **Your recall is 0.16 — is it bad?** It reflects (a) window labels where many rows look normal, (b) faults making up 22 % of training data, (c) a conservative 5 % contamination. Precision is 0.93 and every test fault event had at least one row flagged (10/10 — a lenient criterion). Obvious extremes are caught (memory > 40 %: 100 %). I did not tune on the test set; the sensitivity table is analysis only.
11. **Why not report accuracy?** 81.5 % vs 78.2 % for "always normal" — misleading on imbalanced labels.
12. **Limitations?** Point-wise, no temporal context, contaminated training data, one week/one server, interval-level labels, no root cause yet.
13. **Why LSTM autoencoder next?** It learns normal *sequences* and flags high reconstruction error → catches gradual leaks and temporal patterns, and the per-metric error enables root-cause analysis.
14. **Real-time later?** New `MetricDataLoader` for streams, FastAPI in `interfaces/api`, TimescaleDB as a new store, the trained model is already reloadable (no retraining at start-up).
15. **Data leakage — where prevented?** Chronological split; scaler fitted on train only (test-guarded by a unit test with a scaler that raises on `fit`); causal forward-fill; labels never used in training.
