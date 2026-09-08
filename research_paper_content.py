from __future__ import annotations

import streamlit as st


def render_research_paper() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#14211f; --muted:#62716d; --teal:#087f73; --aqua:#d8f1ec; --orange:#e77c42; --line:#d9e4e0; --paper:#fbfdfc; }
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
        .stApp { background: var(--paper); }
        h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
        .block-container { max-width: 1120px; padding: 2.5rem 3rem 3rem; }
        .paper-header { border-bottom: 1px solid var(--line); padding-bottom: 1.8rem; margin-bottom: 2rem; }
        .eyebrow { color: var(--teal); font-size: .75rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; }
        .paper-header h1 { font-size: clamp(2.2rem, 4vw, 4rem); line-height: 1.02; margin: .55rem 0 .8rem; max-width: 900px; }
        .subtitle { color: var(--muted); font-size: 1.05rem; max-width: 850px; line-height: 1.65; }
        .abstract { background: var(--aqua); border: 1px solid #b8dfd7; border-left: 4px solid var(--teal); border-radius: 5px; padding: 1.1rem 1.25rem; margin: 1.4rem 0 2.2rem; line-height: 1.7; }
        .key-finding { background: white; border: 1px solid var(--line); border-top: 4px solid var(--orange); border-radius: 6px; padding: 1rem 1.1rem; min-height: 140px; }
        .key-finding strong { display:block; font-family:'Space Grotesk'; font-size:1.7rem; margin:.25rem 0; }
        .key-finding span { color: var(--muted); font-size:.85rem; }
        .paper-section { margin-top: 2.4rem; }
        .paper-section h2 { border-bottom: 1px solid var(--line); padding-bottom: .55rem; }
        .paper-section p, .paper-section li { line-height: 1.7; }
        .method-table { width:100%; border-collapse: collapse; background:white; margin: 1rem 0; }
        .method-table th, .method-table td { border:1px solid var(--line); padding:.7rem .8rem; text-align:left; vertical-align:top; }
        .method-table th { background:#eef7f4; color:#365a54; font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
        .footer { border-top:1px solid var(--line); margin-top:3rem; padding-top:1rem; color:var(--muted); font-size:.8rem; text-align:center; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="paper-header"><div class="eyebrow">Research paper · 2026</div>'
        '<h1>Explainable Machine Learning for Drinking Water Quality Assessment in Gilgit-Baltistan</h1>'
        '<div class="subtitle">An evidence-led framework for exploring physicochemical and microbial water-quality signals across a multi-study compilation from Gilgit-Baltistan, Pakistan.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="abstract"><strong>Abstract.</strong> This work presents a transparent water-quality intelligence dashboard and exploratory machine-learning workflow for drinking-water evidence from Gilgit-Baltistan. The system combines study metadata, reported physicochemical measurements, microbial indicators, and source compliance fields into an interactive analytical interface. A rule-based score makes each risk band traceable to reported pH, turbidity, E. coli, and overall-safety judgments. An exploratory model laboratory compares eleven classifiers using stratified cross-validation, missingness-aware preprocessing, and complementary performance and explanation methods. The compilation currently contains 25 aggregated records and three high-risk labels; therefore, the findings are hypothesis-generating and not a validated safety predictor. The primary contribution is an auditable workflow that keeps uncertainty, missing data, source attribution, and model limitations visible to researchers and practitioners.</div>',
        unsafe_allow_html=True,
    )
    metric_columns = st.columns(4)
    for column, value, label in zip(metric_columns, ["25", "2000-2026", "11", "3-fold"], ["Aggregated records", "Study period", "Compared classifiers", "Stratified validation"]):
        with column:
            st.markdown(f'<div class="key-finding"><span>{label}</span><strong>{value}</strong><span>Current project scope</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>1. Introduction</h2>', unsafe_allow_html=True)
    st.markdown("""
Gilgit-Baltistan contains diverse mountain, river, spring, glacier-fed, and groundwater systems. Water-quality evidence is distributed across studies that differ in location, sampling design, reporting detail, and laboratory context. A useful analytical tool must make those differences visible instead of presenting a single opaque score.

This project addresses that need with an interactive dashboard focused on three principles: preserve the source record, expose the rule behind every risk classification, and treat machine learning as exploratory when the sample is small or aggregated. The interface supports comparison across districts, source types, seasons, years, reported compliance fields, and numeric indicators.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>2. Data and study design</h2>', unsafe_allow_html=True)
    st.markdown("""
The source file, `Datasets/gb_water_quality_2000_2026.csv`, is a curated multi-study compilation. It includes study identifiers, author or institution references, publication or sampling years, district and location fields, water-source and sample types, seasons, sample counts, physicochemical measures, microbial measures, compliance flags, and narrative notes. The accompanying `Datasets/data_dictionary.csv` defines the fields, units, and supplied WHO or Pakistan reference points.

The records are not homogeneous individual-sample observations. Several rows summarize a study area, report ranges or percentages, or combine multiple underlying samples. Missing values are retained as missing. The dashboard reports data completeness and labels unavailable values as unreported rather than inferring compliance.
""")
    st.markdown("""
<table class="method-table">
<tr><th>Evidence group</th><th>Examples</th><th>Analytical role</th></tr>
<tr><td>Context</td><td>District, location, water-source type, season</td><td>Stratify and compare reported evidence</td></tr>
<tr><td>Physicochemical</td><td>pH, turbidity, EC, TDS, hardness, iron, arsenic</td><td>Describe water properties and potential guideline concerns</td></tr>
<tr><td>Microbial</td><td>E. coli and total coliforms</td><td>Identify contamination signals; E. coli has zero-tolerance reference status</td></tr>
<tr><td>Compliance</td><td>WHO pH, turbidity, E. coli, and overall safety flags</td><td>Provide source-reported evidence for the interpretable score</td></tr>
</table>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>3. Transparent risk methodology</h2>', unsafe_allow_html=True)
    st.markdown("""
The dashboard calculates a heuristic `Risk_Score` from reported compliance fields. The score is intentionally simple so a reviewer can reconstruct it without a model explanation library:

- pH failure adds 25 points.
- Turbidity failure adds 25 points.
- E. coli failure adds 40 points.
- An overall unsafe judgment adds 10 points.
- An unreported E. coli result combined with an overall unsafe judgment adds 20 points.

The score is capped at 100. Scores of 70 or more are **High risk**, scores from 35 to 69 are **Watch**, and scores below 35 are **Lower risk**. This is an evidence-organizing heuristic, not a probability, diagnosis, certification, or replacement for laboratory confirmation.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>4. Exploratory machine-learning workflow</h2>', unsafe_allow_html=True)
    st.markdown("""
The model target is derived from the transparent score: a record is labeled high risk when `Risk_Score >= 70`. This target is a constructed label, not an independently observed outcome. The workflow compares explainability patterns and generates hypotheses for future individual-sample studies.

Numeric features are selected when at least five values are reported. Total coliform is excluded from the current model feature set because it is especially sparse in this compilation. Missing numeric values receive median imputation plus missingness indicators. Standardization is applied to scale-sensitive classifiers.

The comparison includes logistic regression, decision tree, random forest, extra trees, gradient boosting, AdaBoost, histogram gradient boosting, K-nearest neighbors, support vector machine, linear discriminant analysis, and Gaussian Naive Bayes. Each model uses the same shuffled 3-fold stratified validation scheme and fixed random seed.
""")
    st.markdown("""
<table class="method-table">
<tr><th>Evaluation area</th><th>Measures or method</th><th>Interpretation</th></tr>
<tr><td>Discrimination</td><td>ROC AUC and PR AUC</td><td>Ranking quality; PR AUC is important for the rare high-risk class</td></tr>
<tr><td>Classification</td><td>Accuracy, balanced accuracy, precision, recall, specificity, F1, MCC</td><td>Different trade-offs between missed high-risk records and false alarms</td></tr>
<tr><td>Global explanation</td><td>Permutation importance, native tree importance, coefficient magnitude</td><td>Candidate drivers of model behavior, not causal effects</td></tr>
<tr><td>Local explanation</td><td>Study-level fitted and out-of-fold probabilities</td><td>Case-study context; not a calibrated forecast</td></tr>
</table>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>5. Findings and interpretation</h2>', unsafe_allow_html=True)
    st.markdown("""
The dashboard keeps microbial and physicochemical evidence distinct. E. coli failures receive the largest rule-based weight because the supplied reference treats E. coli as a zero-tolerance drinking-water indicator. Turbidity and pH failures contribute separate, visible components, while the original study's overall judgment remains available as source context.

District, year, water-source, and season comparisons are descriptive. Correlations and charts are shown only when enough overlapping values exist. The interface also displays numeric completeness because apparent differences between locations may reflect reporting coverage as well as water conditions.

No claim of generalization, causation, or predictive superiority should be drawn from the current model scorecard. With only three high-risk labels, fold-level estimates and feature importances can change substantially when one record changes.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>6. Limitations and responsible use</h2>', unsafe_allow_html=True)
    st.markdown("""
1. The dataset is small, aggregated, and assembled from studies with different designs.
2. Missingness may be systematic and cannot be resolved through imputation alone.
3. The derived high-risk label is based on the project's heuristic score.
4. Model probabilities are not validated or calibrated for operational decisions.
5. Source notes, ranges, and percentages require review against original publications.
6. The dataset is not necessarily representative of every community or water source in Gilgit-Baltistan.

Future work should prioritize individual-sample data, consistent laboratory protocols, fuller geographic coverage, independent labels, external validation, uncertainty quantification, and review by water-quality specialists. Any public-health or treatment decision must use current laboratory results and applicable regulatory guidance.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>7. Conclusion</h2>', unsafe_allow_html=True)
    st.markdown("""
This project demonstrates a practical, auditable pattern for water-quality analysis: preserve heterogeneous source evidence, make heuristic judgments inspectable, quantify missingness, and present machine-learning explanations with their limitations. Its current value is as a research and teaching instrument for asking better questions about water safety in Gilgit-Baltistan, not as an autonomous decision system.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="paper-section"><h2>References and project citation</h2>', unsafe_allow_html=True)
    st.markdown("""
1. World Health Organization, *Guidelines for Drinking-water Quality*, supplied reference points as documented in `Datasets/data_dictionary.csv`.
2. Government of Pakistan, National Drinking Water Quality Standards, reference values as documented in `Datasets/data_dictionary.csv`.
3. Original studies and institutions listed in the dataset's `Authors_Source` field.
4. M. Ahammad, “Explainable Machine Learning for Drinking Water Quality Assessment in Gilgit-Baltistan,” GitHub repository, 2026. [Online]. Available: https://github.com/ahammadmejbah/Explainable-Machine-Learning-for-Drinking-Water-Quality-Assessment-in-Gilgit-Baltistan

For BibTeX, see `CITATION.bib`. Reuse of this page, dashboard, source compilation, figures, tables, or derived results must cite this repository and the original studies. Citation does not replace permission or licensing requirements.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="footer">© 2026 Mejbah Ahammad | Lead AI Instructor &amp; Research Scientist Portfolio</div>', unsafe_allow_html=True)
