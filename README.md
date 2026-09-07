# Explainable Machine Learning for Drinking Water Quality Assessment in Gilgit-Baltistan

An explainable water-quality intelligence dashboard for Gilgit-Baltistan, Pakistan. It explores the multi-study compilation from 2000–2026, highlights microbial and physicochemical risk signals, and makes every displayed risk band traceable to reported compliance fields.

## Run the dashboard

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and includes:

- Interactive district, year, and risk-band filters
- Study-level risk overview and comparison charts
- A searchable record table with reported WHO compliance fields
- A transparent rule-based risk score for explainability
- An exploratory Explainable AI lab comparing logistic regression, decision trees, random forests, extra trees, gradient boosting, AdaBoost, histogram gradient boosting, K-nearest neighbors, support vector machines, linear discriminant analysis, and Gaussian Naive Bayes
- Global permutation-importance plots, native tree importance, logistic coefficient magnitude, and local study-level probability explanations
- A data-quality notice for the source CSV's unquoted commas

The source is a multi-study compilation, not a replacement for laboratory testing or regulatory certification. Numeric ranges and narrative findings remain as reported; cite the original study listed in `Authors_Source` when using the data.

The Explainable AI lab is intentionally exploratory because the current compilation has 25 aggregated records and only three high-risk labels. It uses 3-fold stratified validation, median imputation with missingness indicators, and numeric fields reported in at least five records. Model scores and explanations should be treated as hypothesis-generating until more individual-sample data is available.
