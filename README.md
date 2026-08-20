# Data Practicum II: Investigating the Co-occurrence of Supply Chain Gaps driven by Business Closures in the US

## Project Overview

This notebook explores the co-occurrence of supply chain gaps driven by business closures in the United States, focusing on state-level, sector-level, and state-sector unit interactions. The analysis utilizes various metrics related to firm and establishment deaths, labor market dynamics, and economic indicators. Temporal Convolutional Networks (TCNs) are employed to model and predict these co-occurrence patterns, complemented by network graph visualizations to illustrate complex interdependencies.

## Table of Contents

1.  [Data Loading](#data-loading)
2.  [Feature Engineering](#feature-engineering)
3.  [State-State Co-occurrence Modeling](#state-state-co-occurrence-modeling)
    *   [Correlation Analysis](#correlation-analysis-state-state)
    *   [State Embeddings](#state-embeddings)
    *   [Co-occurrence Network](#co-occurrence-network-state-state)
    *   [Data Aggregation for TCN](#data-aggregation-for-tcn-state-state)
    *   [TCN Model Implementation and Evaluation](#tcn-model-implementation-and-evaluation-state-state)
    *   [Feature Importance](#feature-importance-state-state)
4.  [Sector-Sector Co-occurrence Modeling](#sector-sector-co-occurrence-modeling)
    *   [Correlation Analysis](#correlation-analysis-sector-sector)
    *   [Sector Embeddings](#sector-embeddings)
    *   [Co-occurrence Network](#co-occurrence-network-sector-sector)
    *   [Data Aggregation for TCN](#data-aggregation-for-tcn-sector-sector)
    *   [TCN Model Implementation and Evaluation](#tcn-model-implementation-and-evaluation-sector-sector)
    *   [Feature Importance](#feature-importance-sector-sector)
5.  [State-Sector Co-occurrence Modeling](#state-sector-co-occurrence-modeling)
    *   [Correlation Analysis](#correlation-analysis-state-sector)
    *   [State-Sector Unit Embeddings](#state-sector-unit-embeddings)
    *   [Co-occurrence Network](#co-occurrence-network-state-sector)
    *   [Cross-Sectoral Co-occurrence Network (Closure vs. Entry)](#cross-sectoral-co-occurrence-network-closure-vs-entry)

---

## 1. Data Loading

The initial phase involves loading `df_merged.csv` and performing essential data cleaning:

*   Filtering for years greater than 1999.
*   Converting the `year` column to a datetime object.

```
Original shape: (436636, 33), after filtering year > 1999: (227859, 33)
Columns: ['Unnamed: 0', 'year', 'st', 'sector', 'fage', 'firms', 'estabs', 'emp', 'denom', 'estabs_entry', 'estabs_entry_rate', 'estabs_exit', 'estabs_exit_rate', 'job_creation', 'job_creation_births', 'job_creation_continuers', 'job_creation_rate_births', 'job_creation_rate', 'job_destruction', 'job_destruction_deaths', 'job_destruction_continuers', 'job_destruction_rate_deaths', 'job_destruction_rate', 'net_job_creation', 'net_job_creation_rate', 'reallocation_rate', 'firmdeath_firms', 'firmdeath_estabs', 'firmdeath_emp', 'sector_name', 'state_abbr', 'income', 'population']
```

No missing values were found after this initial processing.

## 2. Feature Engineering

This section focuses on creating features that capture sector-level business closures and their drivers. Key features engineered include:

*   **Primary Closure Targets:** `closure_target` (firm deaths) and normalized rates (`closure_rate`, `closure_rate_firms`).
*   **Labor Market Stress and Density Indicators:** `employment_density`, `job_destruction_intensity`.
*   **Temporal Lag Features:** `lag1_closure_rate`, `lag2_closure_rate`, `lag1_job_destruction_rate`, etc., to capture historical dependencies.
*   **Derived Features:** 3-year rolling averages (`ma3_closure_rate`, `ma3_job_destruction_rate`) and year-over-year growth rates (`income_yoy`, `population_yoy`, `employment_yoy`).

All NaN values introduced by lagging are dropped before modeling.

Selected Features for Sector-Level Closure Prediction:

| feature                   | definition                    | supply chain relevance                                    |
| :------------------------ | :---------------------------- | :-------------------------------------------------------- |
| closure_target            | Firm deaths (count)           | Actual business closures in sector                        |
| closure_rate              | Estab. deaths / Estabs        | Normalized closure intensity by sector                    |
| closure_rate_firms        | Firm deaths / Firms           | Firm-level closure intensity by sector                    |
| job_destruction_rate      | Job destruction rate          | Labor market stress leading indicator                     |
| net_job_creation_rate     | Net job creation rate         | Sector labor market momentum                              |
| reallocation_rate         | Reallocation rate             | Structural churn within sector                            |
| lag1_closure_rate         | Previous-year closure rate    | LSTM temporal dependency (t-1)                            |
| lag2_closure_rate         | Prior 2-year closure rate     | LSTM temporal dependency (t-2)                            |
| ma3_closure_rate          | 3-year MA of closure rate     | Smoothed trend in sector closures                         |
| income                    | Median income                 | Proxy for local demand in sector                          |
| population                | Population                    | Market size and labor pool                                |
| income_yoy                | Income growth (YoY)           | Demand momentum by sector                                 |
| population_yoy            | Population growth (YoY)       | Demographic momentum                                      |
| employment_density        | Employment / population       | Sector concentration in local area                        |

## 3. State-State Co-occurrence Modeling

### Correlation Analysis {#correlation-analysis-state-state}

This section examines the correlation of business closure rates (`closure_rate`) among U.S. states.

### Co-occurrence Network {#co-occurrence-network-state-state}

**Network Graph of State Co-occurrence based on Closure Rate Correlation**

This graph connects states where the closure rates are strongly positively correlated (Pearson r >= 0.8). 
Edge thickness indicates correlation strength, node size reflects total state firm deaths. 
States are colored by community detection.

**State-State Co-occurrence Network**

<img width="950" height="605" alt="image" src="https://github.com/user-attachments/assets/2b1d8b9b-8893-4d41-8a5f-580f93397a90" />

### Data Aggregation for TCN {#data-aggregation-for-tcn-state-state}

Data is transformed into a state-pair-year format with difference features for TCN modeling. A binary target `cooccurrence_target_pair` is defined (1 if correlation >= 0.8, else 0).

### TCN Model Implementation and Evaluation {#tcn-model-implementation-and-evaluation-state-state}

A Temporal Convolutional Network (TCN) is built and trained to predict state-state co-occurrence, using a sequence length of 3 years. The model utilizes dilated causal convolutions and residual connections.

**TCN Model Performance (State-State)**

| Metric            | Value    |
| :---------------- | :------- |
| Test Loss         | 0.5659   |
| Test Accuracy     | 0.6806   |
| Test AUC          | 0.7858   |

**Classification Report (State-State)**

```
              precision    recall  f1-score   support

           0       0.92      0.65      0.76      3762
           1       0.38      0.80      0.51      1012

    accuracy                           0.68      4774
   macro avg       0.65      0.72      0.64      4774
weighted avg       0.81      0.68      0.71      4774
```

**Confusion Matrix (State-State)**

<img width="528" height="470" alt="image" src="https://github.com/user-attachments/assets/16013aef-9650-48e3-88d0-924bfc6e8d4b" />


### Feature Importance {#feature-importance-state-state}

Permutation feature importance based on AUC drop:

| Feature                       | Importance (AUC Drop) |
| :---------------------------- | :-------------------- |
| reallocation_rate_diff        | 0.1205                |
| employment_density_diff       | 0.1043                |
| income_diff                   | 0.0817                |
| population_yoy_diff           | 0.0421                |
| population_diff               | 0.0400                |
| closure_rate_firms_diff       | 0.0380                |
| lag1_closure_rate_diff        | 0.0212                |
| net_job_creation_rate_diff    | 0.0157                |
| closure_rate_diff             | 0.0137                |
| income_yoy_diff               | 0.0099                |
| emp_diff                      | 0.0094                |
| job_destruction_rate_diff     | 0.0092                |
| employment_yoy_diff           | 0.0089                |
| ma3_job_destruction_rate_diff | 0.0088                |
| lag2_closure_rate_diff        | 0.0056                |
| ma3_closure_rate_diff         | 0.0054                |

## 4. Sector-Sector Co-occurrence Modeling

### Correlation Analysis {#correlation-analysis-sector-sector}

This section analyzes the correlation of business closure rates (`closure_rate`) among different sectors.

### Co-occurrence Network {#co-occurrence-network-sector-sector}

**Network Graph of Sector Co-occurrence based on Closure Rate Correlation**

This graph connects sectors where the closure rates are strongly positively correlated (Pearson r >= 0.8). 
Edge thickness indicates correlation strength, node size reflects total sector firm deaths. 
Sectors are colored by community detection. Node labels represent 2-digit NAICS codes.

**Sector-Sector Co-occurrence Network** 

<img width="1281" height="1074" alt="image" src="https://github.com/user-attachments/assets/d484616a-5963-4bd9-ad08-b2d7e98187a0" />


### Data Aggregation for TCN {#data-aggregation-for-tcn-sector-sector}

Similar to state-state, data is aggregated to a sector-pair-year level with difference features and a binary `cooccurrence_target_pair_sectors`.

### TCN Model Implementation and Evaluation {#tcn-model-implementation-and-evaluation-sector-sector}

A TCN model is applied to predict sector-sector co-occurrence, with a sequence length of 3 years.

**TCN Model Performance (Sector-Sector)**

| Metric            | Value    |
| :---------------- | :------- |
| Test Loss         | 0.3573   |
| Test Accuracy     | 0.7714   |
| Test AUC          | 0.8078   |

**Classification Report (Sector-Sector)**

```
              precision    recall  f1-score   support

           0       0.92      0.81      0.86       682
           1       0.25      0.49      0.33        88

    accuracy                           0.77       770
   macro avg       0.59      0.65      0.60       770
weighted avg       0.85      0.77      0.80       770
```

**Confusion Matrix (Sector-Sector)**

<img width="528" height="470" alt="image" src="https://github.com/user-attachments/assets/6fb83fb7-4cf9-4961-bb2d-97f7ee813f84" />


### Feature Importance {#feature-importance-sector-sector}

Permutation feature importance based on AUC drop for sector-level model:

| Feature                       | Importance (AUC Drop) |
| :---------------------------- | :-------------------- |
| reallocation_rate_diff        | 0.0310                |
| lag2_closure_rate_diff        | 0.0306                |
| job_destruction_rate_diff     | 0.0267                |
| population_yoy_diff           | 0.0260                |
| emp_diff                      | 0.0167                |
| income_diff                   | 0.0147                |
| closure_rate_firms_diff       | 0.0144                |
| lag1_closure_rate_diff        | 0.0121                |
| ma3_job_destruction_rate_diff | 0.0118                |
| closure_rate_diff             | 0.0117                |
| income_yoy_diff               | 0.0090                |
| ma3_closure_rate_diff         | 0.0086                |
| employment_density_diff       | 0.0075                |
| population_diff               | 0.0060                |
| employment_yoy_diff           | 0.0058                |
| net_job_creation_rate_diff    | 0.0002                |

## 5. State-Sector Co-occurrence Modeling

### Correlation Analysis {#correlation-analysis-state-sector}

This section investigates two types of correlations:

1.  **State vs. Sector Overall Closure Rate Correlation:** Measures the correlation between a state's overall closure rate and a sector's overall closure rate.
2.  **State-Sector Unit Closure Rate Correlation:** Examines correlations between time series of business closure rates for every unique (state, sector) combination. This is used for network graphs and TCNs.


### Co-occurrence Network {#co-occurrence-network-state-sector}

**Network Graph of Top State-Sector Units with Strong Positive Correlation in Closure Rates**

This graph connects top 30 state-sector units where closure rates are strongly positively correlated (Pearson r >= 0.8). 
Edge thickness indicates correlation strength, node size reflects total unit firm deaths. 
Units are colored by community detection. Node labels are in 'STATE_NAICS_CODE' format.

**State-Sector Co-occurrence Network**

<img width="1726" height="1617" alt="image" src="https://github.com/user-attachments/assets/f3d9980a-bc20-4626-867e-d69f67621e79" />


### Cross-Sectoral Co-occurrence Network (Closure vs. Entry) {#cross-sectoral-co-occurrence-network-closure-vs-entry}

**Network Graph of Cross-Sectoral Co-occurrence (Closure vs. Entry)**

This graph connects sectors where a source sector's closure rate is strongly positively correlated with a target sector's establishment entry rate (Pearson r >= 0.3). 
Edge thickness indicates correlation strength, arrows point from source closure to target entry. Node size reflects total sector firm deaths. 
Sectors are colored by community detection based on their overall relatedness. Node labels represent 2-digit NAICS codes.

**Cross-Sectoral Co-occurrence Network**

<img width="1340" height="994" alt="image" src="https://github.com/user-attachments/assets/112426cb-84e1-47f9-a294-05f05d2c233d" />


**2-digit NAICS 2017 Codes:**

*   11 - Agriculture, Forestry, Fishing and Hunting
*   21 - Mining, Quarrying, and Oil and Gas Extraction
*   22 - Utilities
*   23 - Construction
*   31-33 - Manufacturing
*   42 - Wholesale Trade
*   44-45 - Retail Trade
*   48-49 - Transportation and Warehousing
*   51 - Information
*   52 - Finance and Insurance
*   53 - Real Estate and Rental and Leasing
*   54 - Professional, Scientific, and Technical Services
*   55 - Management of Companies and Enterprises
*   56 - Administrative and Support and Waste Management and Remediation Services
*   61 - Educational Services
*   62 - Health Care and Social Assistance
*   71 - Arts, Entertainment, and Recreation
*   72 - Accommodation and Food Services
*   81 - Other Services (except Public Administration)

## Libraries Used

This notebook utilizes the following Python libraries:

*   **pandas (`pd`)**: For data manipulation and analysis, including DataFrame operations, reading CSV files, and aggregation.
*   **numpy (`np`)**: For numerical operations, especially array manipulation and mathematical functions.
*   **pathlib**: For object-oriented filesystem paths.
*   **sklearn.decomposition.TruncatedSVD**: For dimensionality reduction to create embeddings from correlation matrices.
*   **community.community_louvain (`co`)**: For community detection in network graphs.
*   **networkx (`nx`)**: For creating, manipulating, and studying the structure of complex networks.
*   **collections.defaultdict**: For simplifying the creation of dictionaries with default values.
*   **matplotlib.pyplot (`plt`)**: For creating static, interactive, and animated visualizations in Python.
*   **netgraph.Graph**: For advanced and visually appealing network graph plotting.
*   **plotly.express (`px`)**: For creating interactive plots, used here for generating color palettes.
*   **tensorflow (`tf`)**: An open-source machine learning framework for building and training neural networks.
*   **tensorflow.keras**: High-level API for building and training deep learning models, integrated into TensorFlow.
*   **tensorflow.keras.layers**: Provides common neural network layers like `Conv1D`, `BatchNormalization`, `Dropout`, `Add`, `Activation`, `GlobalAveragePooling1D`, and `Dense`.
*   **tensorflow.keras.callbacks.EarlyStopping**: A callback to stop training when a monitored metric has stopped improving.
*   **sklearn.model_selection.train_test_split**: For splitting datasets into training, validation, and test sets.
*   **sklearn.preprocessing.StandardScaler**: For standardizing features by removing the mean and scaling to unit variance.
*   **sklearn.metrics.classification_report**: For generating a text report showing the main classification metrics.
*   **sklearn.metrics.roc_auc_score**: For calculating the Area Under the Receiver Operating Characteristic Curve (AUC-ROC) score.
*   **sklearn.metrics.confusion_matrix**: For computing the confusion matrix to evaluate the accuracy of a classification.
