# Explainable Machine Learning for Drinking Water Quality Assessment in Gilgit-Baltistan

An explainable water-quality intelligence dashboard for Gilgit-Baltistan, Pakistan. The project brings together reported results from multiple studies, makes the source evidence filterable, and provides transparent risk rules alongside an exploratory machine-learning laboratory.

The application is designed for research exploration and communication. It is not a replacement for laboratory testing, public-health action, or regulatory certification.

## Contents

- [Project goals](#project-goals)
- [Features](#features)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation and usage](#installation-and-usage)
- [Dashboard guide](#dashboard-guide)
- [Data and schema](#data-and-schema)
- [Risk scoring](#risk-scoring)
- [Explainable AI lab](#explainable-ai-lab)
- [Data-quality behavior](#data-quality-behavior)
- [Limitations and responsible use](#limitations-and-responsible-use)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License and attribution](#license-and-attribution)

## Project goals

This project aims to:

1. Make reported drinking-water evidence from Gilgit-Baltistan easier to inspect.
2. Highlight microbial and physicochemical indicators associated with unsafe records.
3. Keep risk categories traceable to reported compliance fields.
4. Demonstrate explainable modeling workflows without presenting a small aggregated dataset as a validated predictor.

## Features

The Streamlit dashboard provides:

- District, year, and risk-band filters.
- Study-level risk summaries, charts, and comparison tables.
- Water-source and seasonality views.
- Descriptive statistics, correlations, data-completeness charts, and a turbidity versus E. coli view when enough values are available.
- A full record table and CSV download for the filtered view.
- A transparent rule-based risk score and risk band for every record.
- An exploratory model comparison for logistic regression, decision trees, random forests, extra trees, gradient boosting, AdaBoost, histogram gradient boosting, K-nearest neighbors, support vector machines, linear discriminant analysis, and Gaussian Naive Bayes.
- Model metrics including accuracy, balanced accuracy, precision, recall/sensitivity, specificity, F1, ROC AUC, PR AUC, Matthews correlation coefficient, and fold-to-fold variability.
- Global permutation importance, native tree importance, logistic coefficient magnitude, out-of-fold predictions, confusion matrices, ROC curves, and local study-level probability explanations.

## Repository structure

```text
.
├── app.py                                  # Streamlit application
├── requirements.txt                         # Python dependencies
├── README.md                                # Project documentation
└── Datasets/
	├── gb_water_quality_2000_2026.csv      # Multi-study source compilation
	├── data_dictionary.csv                 # Column definitions and reference limits
	└── readme.md                           # Dataset-specific notes
```

## Requirements

- Python 3.10 or newer is recommended.
- `pip` for installing Python packages.
- A browser for the local Streamlit interface.

The application depends on Streamlit, pandas, Plotly, and scikit-learn. Exact minimum versions are listed in [requirements.txt](requirements.txt).

## Installation and usage

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start the dashboard:

```bash
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`. If it does not open automatically, copy that address into a browser. To use another port:

```bash
streamlit run app.py --server.port 8502
```

The CSV is loaded relative to `app.py`, so start Streamlit from the repository root or provide the path to the application file:

```bash
streamlit run /path/to/Explainable-Machine-Learning-for-Drinking-Water-Quality-Assessment-in-Gilgit-Baltistan/app.py
```

## Dashboard guide

### Filters and overview

The sidebar filters the records by district or study area, risk band, and publication or sampling year. The headline metrics update to show the records in the current view, high-risk records, reported E. coli failures, and the number of available numeric measurements.

The **Download filtered records** control exports the current filtered data, including calculated `Risk_Score` and `Risk_Band` columns.

### Executive readout

The first section shows the distribution of lower-risk, watch, and high-risk records. The driver chart counts records with reported failures for pH, turbidity, E. coli, and the source study's overall safety judgement.

### Geographic, temporal, and source context

The dashboard compares districts, years, water-source types, and seasons. It also provides descriptive statistics, pairwise correlations, the highest-risk records, WHO compliance counts, and numeric-field completeness.

All counts are record counts unless explicitly labeled otherwise. A record may summarize many individual samples.

### Source records

The source-record section contains district summaries and a full table of the selected records. Blank values are displayed as unreported in the interface and are not treated as passing results.

### Explainable AI lab

The model lab appears when the dataset contains enough rows and at least three high-risk labels. It compares models using the complete dataset and uses the same 3-fold stratified splits for each model. See [Explainable AI lab](#explainable-ai-lab) for interpretation details.

## Data and schema

The main source is `Datasets/gb_water_quality_2000_2026.csv`. It currently contains a multi-study compilation spanning 2000–2026. Some rows represent aggregated study results, ranges, percentages, or narrative summaries rather than individual laboratory samples.

The authoritative column descriptions, units, and reference values are in `Datasets/data_dictionary.csv`. Major field groups are:

| Group | Columns | Purpose |
| --- | --- | --- |
| Study metadata | `Study_ID`, `Authors_Source`, `Year` | Identify and cite the original study or sampling campaign |
| Geography and sampling | `District`, `Location`, `Water_Source_Type`, `Sample_Type`, `Season`, `N_Samples` | Describe where, when, and how samples were collected |
| Physicochemical measures | `Temperature_C`, `pH`, `EC_uS_cm`, `Turbidity_NTU`, `TDS_mg_L`, `DO_mg_L`, `Hardness_mg_L`, `Alkalinity_mg_L`, `Ca_mg_L`, `Mg_mg_L`, `Cl_mg_L`, `Fe_mg_L`, `As_mg_L` | Report measured water properties |
| Microbial measures | `E_coli_CFU_100ml`, `Total_Coliform_CFU_100ml` | Report microbial contamination indicators |
| Compliance and context | `WHO_pH_Pass`, `WHO_Turbidity_Pass`, `WHO_Ecoli_Pass`, `Overall_Safe`, `Notes` | Preserve reported pass/fail judgments and source context |

Selected reference points in the supplied dictionary include pH 6.5–8.5, turbidity below 5 NTU, E. coli equal to zero, total coliform equal to zero, iron below 0.3 mg/L, and arsenic below 0.01 mg/L under the listed WHO reference. Check the dictionary and the original source before applying any limit to a new regulatory decision.

## Risk scoring

The dashboard calculates an interpretable heuristic score from the reported compliance fields. The score starts at zero and adds:

| Condition | Points |
| --- | ---: |
| `WHO_pH_Pass == "No"` | 25 |
| `WHO_Turbidity_Pass == "No"` | 25 |
| `WHO_Ecoli_Pass == "No"` | 40 |
| `Overall_Safe == "No"` | 10 |
| E. coli compliance is unreported and `Overall_Safe == "No"` | 20 |

The final score is capped at 100. Risk bands are:

- **High risk:** score at least 70.
- **Watch:** score from 35 through 69.
- **Lower risk:** score below 35.

This score is an explanation aid, not a validated probability of contamination. Missing values do not add passing points, and an absent compliance field should not be interpreted as evidence that a sample passed.

## Explainable AI lab

The model target is a binary label derived from the heuristic score: records with `Risk_Score >= 70` are labeled high risk. This means the models learn a small, constructed label rather than an independently collected outcome.

### Features and preprocessing

- Numeric columns are eligible when at least five records contain a reported value.
- `Total_Coliform_CFU_100ml` is excluded from modeling to avoid using a sparse microbial field in the current compilation.
- Missing numeric values are median-imputed.
- Missingness indicators are added so the models can distinguish an imputed value from a directly reported value.
- Scaling is applied to models that need comparable feature magnitudes, including logistic regression, K-nearest neighbors, support vector machines, and linear discriminant analysis.

### Validation and metrics

The lab uses shuffled 3-fold stratified cross-validation with a fixed random seed. It reports mean scores and standard deviations across folds. PR AUC is included because the high-risk class is rare. Recall measures detection of high-risk records, while specificity measures how often records outside that class avoid false alarms.

Permutation importance is calculated on the complete small dataset, so it is useful for exploration but can be unstable. Local probabilities are case-study explanations and should not be read as calibrated forecasts.

## Data-quality behavior

The source CSV contains unquoted commas in some free-text fields. The loader in `app.py` detects the year and compliance columns, reconstructs the expected 29-column record, and shows a data-quality notice when rows require repair. This preserves the records and makes the compliance fields usable, but it cannot recover information that was not present or determine the exact meaning of ambiguous narrative text.

When updating the CSV:

1. Keep the header and column order aligned with `data_dictionary.csv`.
2. Quote any field containing commas, especially `Authors_Source` and `Notes`.
3. Preserve blank values as blank rather than replacing them with zero or a passing value.
4. Keep the original source citation in `Authors_Source` and relevant context in `Notes`.
5. Recheck row counts, compliance fields, units, and duplicated study identifiers before publishing results.

## Limitations and responsible use

- The current compilation has 25 aggregated records and only three high-risk labels.
- Study-level records can summarize different sample counts, locations, methods, and reporting conventions.
- Results are not necessarily representative of all water sources or communities in Gilgit-Baltistan.
- Missingness is substantial and may be systematic; imputation in the model lab does not create new measurements.
- Reported ranges and narrative percentages are retained as supplied and are not independently reanalyzed.
- The rule-based score and model target are not substitutes for laboratory confirmation.
- Model comparisons are hypothesis-generating and should not be used to make treatment, clinical, or regulatory decisions.

For publication or operational use, verify each claim against the original study, document sampling and laboratory methods, use individual-sample data where possible, and consult applicable Pakistani and international drinking-water standards.

## Troubleshooting

### `streamlit: command not found`

Activate the virtual environment and install the requirements again:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

You can also invoke Streamlit through Python:

```bash
python -m streamlit run app.py
```

### Dataset not found

Run the command from the repository root and confirm that this file exists:

```text
Datasets/gb_water_quality_2000_2026.csv
```

### The model lab is unavailable

The model lab intentionally requires enough rows and at least three high-risk labels. With the current data, it should be available, but filtering does not change the model-training dataset. Check the source CSV if records or labels have been removed.

### A chart says there are not enough values

Charts require a minimum number of non-missing values. This is expected for sparse study summaries and is preferable to presenting an unstable chart as a firm result.

## Contributing

Contributions should preserve traceability and reproducibility. A useful change generally includes:

1. A short description of the data or code change.
2. The source and citation for newly added records or thresholds.
3. Any impact on the risk-score rules or model interpretation.
4. A local run of `streamlit run app.py` to check the dashboard.
5. Updated documentation when fields, commands, or interpretation change.

Do not commit private laboratory records, credentials, or unverified claims about water safety.

## License and attribution

No license file is currently included in the repository. Confirm the licensing and reuse terms for the original studies before redistributing their data or reproducing their results.

When using this project or dataset, cite the original study listed in `Authors_Source` and identify this repository as the dashboard and compilation layer. The dataset is a research compilation; attribution to the underlying authors remains necessary.
