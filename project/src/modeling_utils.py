"""
modeling_utils.py
==================
Clinical Prioritization Engine — Streamlit section for the
"Data Audit & Exploration" dashboard.

Unlike the other tabs (which receive the df already loaded from the
sidebar and compute PCA/K-Means live), this tab replicates Section 1
of the original "Week 10" notebook: it loads TWO of its own files
and joins them by index:

    df          <- healtcare_processedv2.csv   (already-cleaned dataset, everything encoded as integers)
    clusters_df <- healthcare_pca_clusters.csv (same dataset + 'cluster' column, with NaN
                                                 for rows that didn't make it into clustering)

Usage from app.py:
    from modeling_utils import run_care_prioritization_section
    run_care_prioritization_section()   # no arguments: handles its own file loading
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from pathlib import Path


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from catboost import CatBoostClassifier



BASE_DIR = Path(__file__).resolve().parent.parent

INTERIM_FILE = BASE_DIR / "data" / "interim" / "healtcare_processedv2.csv"
CLUSTERS_FILE = BASE_DIR / "data" / "processed" / "healthcare_pca_clusters.csv"


df = pd.read_csv(INTERIM_FILE)
clusters_df = pd.read_csv(CLUSTERS_FILE)

plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'axes.edgecolor': '#888888',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#444444',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# Data schema (everything encoded as integers, same as the notebook)
# ─────────────────────────────────────────────────────────────

TARGET = "Test Results"

LABEL_ENCODING_MAPS = {
    "Gender": {0: "Female", 1: "Male"},
    "Blood Type": {0: "A+", 1: "A-", 2: "AB+", 3: "AB-", 4: "B+", 5: "B-", 6: "O+", 7: "O-"},
    "Medical Condition": {0: "Arthritis", 1: "Asthma", 2: "Cancer", 3: "Diabetes", 4: "Hypertension", 5: "Obesity"},
    "Insurance Provider": {0: "Aetna", 1: "Blue Cross", 2: "Cigna", 3: "Medicare", 4: "UnitedHealthcare"},
    "Admission Type": {0: "Elective", 1: "Emergency", 2: "Urgent"},
    "Medication": {0: "Aspirin", 1: "Ibuprofen", 2: "Lipitor", 3: "Paracetamol", 4: "Penicillin"},
    "Test Results": {0: "Normal", 1: "Inconclusive", 2: "Abnormal"},
    "Admission Day of Week": {0: "Friday", 1: "Monday", 2: "Saturday", 3: "Sunday", 4: "Thursday", 5: "Tuesday", 6: "Wednesday"},
    "Admission Month": {0: "April", 1: "August", 2: "December", 3: "February", 4: "January", 5: "July", 6: "June", 7: "March", 8: "May", 9: "November", 10: "October", 11: "September"},
}

ABNORMAL_CODE = 2  # 'Abnormal' in Test Results

FEATURES = [
    "Age", "Gender", "Blood Type", "Medical Condition",
    "Insurance Provider", "Billing Amount", "Admission Type",
    "Medication", "Length of Stay", "Admission Day of Week",
    "Admission Month", "Admission Year", "Cluster_Biz", "Cost_Per_Day",
]

# Categorical columns (CatBoost receives them natively, without one-hot)
CATEGORICAL_FEATURES = [
    "Gender", "Blood Type", "Medical Condition", "Insurance Provider",
    "Admission Type", "Medication", "Admission Day of Week", "Admission Month",
    "Cluster_Biz",
]

MODEL_OPTIONS = ["Baseline", "Logistic Regression", "Random Forest", "CatBoost", "Ensemble"]


# ─────────────────────────────────────────────────────────────
# 1. Data loading and alignment (replicates Section 1 of the notebook)
# ─────────────────────────────────────────────────────────────

def load_and_merge_datasets() -> tuple[pd.DataFrame, dict]:
    """
    Automatically loads the repository's CSVs,
    aligns them by index, and adds Cost_Per_Day.
    """

    df = pd.read_csv(INTERIM_FILE)
    clusters_df = pd.read_csv(CLUSTERS_FILE)

    clusters_df = clusters_df.dropna(subset=["cluster"])
    clusters_df["Cluster_Biz"] = clusters_df["cluster"].astype(int)

    df_aligned = df.loc[clusters_df.index].copy()
    df_aligned["Cluster_Biz"] = clusters_df["Cluster_Biz"].values

    df_aligned["Cost_Per_Day"] = (
        df_aligned["Billing Amount"] /
        df_aligned["Length of Stay"].replace(0, np.nan)
    )

    meta = {
        "n_total_rows": len(df),
        "n_with_cluster": len(df_aligned),
        "cluster_counts": (
            clusters_df["Cluster_Biz"]
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    return df_aligned, meta


# ─────────────────────────────────────────────────────────────
# 2. Temporal split + scaling
# ─────────────────────────────────────────────────────────────

def temporal_split(df: pd.DataFrame, features: list, target_col: str, year_col="Admission Year", min_test_rows=20):
    meta = {}
    if year_col in df.columns:
        cutoff_year = df[year_col].quantile(0.80, interpolation="lower")
        train_mask = df[year_col] <= cutoff_year
        test_mask = df[year_col] > cutoff_year
        if test_mask.sum() >= min_test_rows:
            train_df, test_df = df[train_mask], df[test_mask]
            meta["strategy"] = "temporal"
            meta["cutoff_year"] = int(cutoff_year)
        else:
            meta["strategy"] = "sorted_index_fallback"
            df_sorted = df.sort_values(year_col).reset_index(drop=True)
            split_idx = int(len(df_sorted) * 0.80)
            train_df, test_df = df_sorted.iloc[:split_idx], df_sorted.iloc[split_idx:]
    else:
        meta["strategy"] = "row_order_fallback"
        split_idx = int(len(df) * 0.80)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train, y_train = train_df[features].reset_index(drop=True), train_df[target_col].reset_index(drop=True)
    X_test, y_test = test_df[features].reset_index(drop=True), test_df[target_col].reset_index(drop=True)
    meta["n_train"], meta["n_test"] = len(X_train), len(X_test)
    return X_train, X_test, y_train, y_test, meta


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler over ALL features (same as the original notebook:
    since they already come encoded as integers, they are scaled
    directly without one-hot)."""
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_test_sc, scaler


# ─────────────────────────────────────────────────────────────
# 3. Metrics
# ─────────────────────────────────────────────────────────────

def asymmetric_cost(y_true, y_pred, fn_cost=3, fp_cost=1) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    total = 0
    for t, p in zip(y_true, y_pred):
        if t == ABNORMAL_CODE and p == 0:
            total += fn_cost
        elif t == 0 and p == ABNORMAL_CODE:
            total += fp_cost
        elif t != p:
            total += 1
    return total / len(y_true)


def evaluate_model(y_true, y_pred, target_names=None) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "asymmetric_cost": asymmetric_cost(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, target_names=target_names, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def precision_at_k(ranking_df: pd.DataFrame, k_list=(10, 20, 50), real_col="real_class", target_class="Abnormal") -> dict:
    out = {}
    for k in k_list:
        top_k = ranking_df.head(k)
        out[k] = (top_k[real_col] == target_class).mean() if len(top_k) else np.nan
    return out


# ─────────────────────────────────────────────────────────────
# 4. Models
# ─────────────────────────────────────────────────────────────

def train_baseline(X_train_sc, y_train):
    m = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    m.fit(X_train_sc, y_train)
    return m


def train_logistic_regression(X_train_sc, y_train):
    m = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    m.fit(X_train_sc, y_train)
    return m


def train_random_forest(X_train_sc, y_train, n_estimators=200, max_depth=10):
    m = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    m.fit(X_train_sc, y_train)
    return m


def get_cat_features_idx(features: list) -> list:
    return [features.index(c) for c in CATEGORICAL_FEATURES if c in features]


def train_catboost(X_train, y_train, X_test, y_test, cat_features_idx,
                    iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=5,
                    early_stopping_rounds=50, verbose=0):
    m = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        loss_function="MultiClass", class_weights=[1, 1, 1],
        l2_leaf_reg=l2_leaf_reg, cat_features=cat_features_idx,
        random_seed=RANDOM_STATE, verbose=verbose,
    )
    m.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=early_stopping_rounds)
    return m


def ensemble_predict_proba(y_prob_a, y_prob_b):
    return (y_prob_a + y_prob_b) / 2


# ─────────────────────────────────────────────────────────────
# 5. Clinical ranking
# ─────────────────────────────────────────────────────────────

def build_ranking(X_test_raw: pd.DataFrame, y_prob, y_pred, y_test) -> pd.DataFrame:
    label_map = LABEL_ENCODING_MAPS[TARGET]
    ranking_df = X_test_raw.copy()

    class_names = [label_map[i] for i in range(y_prob.shape[1])]
    for i, name in enumerate(class_names):
        ranking_df[f"prob_{name.lower()}"] = y_prob[:, i].round(3)

    ranking_df["predicted_class"] = [label_map[x] for x in np.asarray(y_pred)]
    ranking_df["real_class"] = [label_map[x] for x in np.asarray(y_test)]
    ranking_df["prob_abnormal"] = y_prob[:, ABNORMAL_CODE].round(3)
    ranking_df["priority_rank"] = ranking_df["prob_abnormal"].rank(ascending=False).astype(int)

    # Extra human-readable columns, useful for reviewing the ranking
    for col in ["Medical Condition", "Admission Type", "Gender"]:
        if col in ranking_df.columns:
            ranking_df[f"{col}_label"] = ranking_df[col].map(LABEL_ENCODING_MAPS.get(col, {}))

    return ranking_df.sort_values("priority_rank").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 6. Error analysis
# ─────────────────────────────────────────────────────────────

def error_analysis(X_test_raw: pd.DataFrame, y_test, y_pred) -> dict:
    label_map = LABEL_ENCODING_MAPS[TARGET]
    y_test_arr, y_pred_arr = np.asarray(y_test), np.asarray(y_pred)
    error_mask = y_pred_arr != y_test_arr

    X_errors = X_test_raw[error_mask].copy()
    X_errors["y_real"] = y_test_arr[error_mask]
    X_errors["y_pred"] = y_pred_arr[error_mask]

    error_pairs = pd.crosstab(
        X_errors["y_real"].map(label_map),
        X_errors["y_pred"].map(label_map),
        rownames=["Actual"], colnames=["Predicted"],
    )
    critical_errors = X_errors[(X_errors["y_real"] == ABNORMAL_CODE) & (X_errors["y_pred"] == 0)]

    return {
        "n_errors": int(error_mask.sum()),
        "n_total": len(y_test_arr),
        "error_rate": float(error_mask.mean()) if len(y_test_arr) else 0.0,
        "error_pairs": error_pairs,
        "critical_errors": critical_errors,
    }


# ─────────────────────────────────────────────────────────────
# 7. Full pipeline
# ─────────────────────────────────────────────────────────────

def run_full_pipeline(df_clean: pd.DataFrame, selected_models: list, features: list = None) -> dict:
    features = features or FEATURES
    selected_models = set(selected_models)
    if "Ensemble" in selected_models:
        selected_models.update({"Random Forest", "CatBoost"})

    df_clean = df_clean.dropna(subset=features + [TARGET]).copy()
    df_clean[TARGET] = df_clean[TARGET].astype(int)
    if "Cluster_Biz" in df_clean.columns:
        df_clean["Cluster_Biz"] = df_clean["Cluster_Biz"].astype(int)

    label_map = LABEL_ENCODING_MAPS[TARGET]
    target_names = [label_map[k] for k in sorted(label_map)]

    X_train, X_test, y_train, y_test, split_meta = temporal_split(df_clean, features, TARGET)
    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)
    cat_idx = get_cat_features_idx(features)

    results = {
        "features": features,
        "split_meta": split_meta,
        "metrics": {},
        "y_test": y_test,
        "X_test_raw": X_test,
    }

    y_prob_rf = y_prob_cb = None

    if "Baseline" in selected_models:
        m = train_baseline(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        results["metrics"]["Baseline"] = evaluate_model(y_test, y_pred, target_names)

    if "Logistic Regression" in selected_models:
        m = train_logistic_regression(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        results["metrics"]["Logistic Regression"] = evaluate_model(y_test, y_pred, target_names)

    if "Random Forest" in selected_models:
        m = train_random_forest(X_train_sc, y_train)
        y_pred = m.predict(X_test_sc)
        y_prob_rf = m.predict_proba(X_test_sc)
        results["metrics"]["Random Forest"] = evaluate_model(y_test, y_pred, target_names)
        results["feature_importance"] = pd.Series(m.feature_importances_, index=features).sort_values(ascending=False)

    if "CatBoost" in selected_models:
        m = train_catboost(X_train[features], y_train, X_test[features], y_test, cat_idx)
        y_pred = m.predict(X_test[features]).flatten()
        y_prob_cb = m.predict_proba(X_test[features])
        results["metrics"]["CatBoost"] = evaluate_model(y_test, y_pred, target_names)

    if "Ensemble" in selected_models and y_prob_rf is not None and y_prob_cb is not None:
        y_prob_ens = ensemble_predict_proba(y_prob_rf, y_prob_cb)
        y_pred_ens = y_prob_ens.argmax(axis=1)
        results["metrics"]["Ensemble"] = evaluate_model(y_test, y_pred_ens, target_names)
        results["ranking"] = build_ranking(X_test, y_prob_ens, y_pred_ens, y_test)
        results["error_analysis"] = error_analysis(X_test, y_test, y_pred_ens)
    elif "CatBoost" in selected_models and y_prob_cb is not None:
        y_pred_cb = y_prob_cb.argmax(axis=1)
        results["ranking"] = build_ranking(X_test, y_prob_cb, y_pred_cb, y_test)
        results["error_analysis"] = error_analysis(X_test, y_test, y_pred_cb)
    elif "Random Forest" in selected_models and y_prob_rf is not None:
        y_pred_rf = y_prob_rf.argmax(axis=1)
        results["ranking"] = build_ranking(X_test, y_prob_rf, y_pred_rf, y_test)
        results["error_analysis"] = error_analysis(X_test, y_test, y_pred_rf)

    return results


# ─────────────────────────────────────────────────────────────
# 8. Streamlit section (called from app.py, no arguments)
# ─────────────────────────────────────────────────────────────

def run_care_prioritization_section():

    try:
        df_model, load_meta = load_and_merge_datasets()
    except Exception as e:
        st.error(f"Error loading/merging the files: {e}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Total Rows (processed dataset)", f"{load_meta['n_total_rows']:,}")
    c2.metric("Rows with Cluster Assigned", f"{load_meta['n_with_cluster']:,}")
    st.caption(f"Cluster_Biz distribution: {load_meta['cluster_counts']}")

    with st.expander("Preview of the aligned data"):
        preview_cols = [c for c in FEATURES + [TARGET] if c in df_model.columns]
        st.dataframe(df_model[preview_cols].head(10), width='stretch')

    st.divider()
    st.markdown("##### 2. Model Configuration")
    col1, col2 = st.columns(2)
    with col1:
        selected_features = st.multiselect(
            "Features:", options=[f for f in FEATURES if f in df_model.columns],
            default=[f for f in FEATURES if f in df_model.columns],
            key="prioritization_features",
        )
    with col2:
        selected_models = st.multiselect(
            "Models to Train:", options=MODEL_OPTIONS, default=MODEL_OPTIONS,
            key="prioritization_models",
        )

    train_clicked = st.button("🚀 Train Models", type="primary", key="prioritization_train_btn")

    if train_clicked:
        if not selected_features or not selected_models:
            st.warning("Select at least one feature and one model.")
        else:
            with st.spinner("Training clinical prioritization models..."):
                try:
                    results = run_full_pipeline(df_model, selected_models, selected_features)
                    st.session_state["prioritization_results"] = results
                except Exception as e:
                    st.error(f"Error during training: {e}")

    results = st.session_state.get("prioritization_results")
    if results is None:
        st.info("Configure the options and click **Train Models** to see results.")
        return

    meta = results["split_meta"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Training Rows", meta["n_train"])
    c2.metric("Test Rows", meta["n_test"])
    c3.metric("Split Strategy", meta["strategy"])

    st.divider()
    st.markdown("##### Model Comparison")
    rows = [
        {"Model": name, "Accuracy": round(m["accuracy"], 4),
         "F1 Macro": round(m["f1_macro"], 4), "Asymmetric Cost": round(m["asymmetric_cost"], 4)}
        for name, m in results["metrics"].items()
    ]
    metrics_df = pd.DataFrame(rows).sort_values("F1 Macro", ascending=False)
    st.dataframe(metrics_df, width='stretch', hide_index=True)

    model_for_detail = st.selectbox(
        "View detail for a model:", options=list(results["metrics"].keys()), key="prioritization_detail_model"
    )
    detail = results["metrics"][model_for_detail]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"###### Classification report — {model_for_detail}")
        st.code(detail["classification_report"])
    with col_b:
        st.markdown(f"###### Confusion Matrix — {model_for_detail}")
        labels = [LABEL_ENCODING_MAPS[TARGET][k] for k in sorted(LABEL_ENCODING_MAPS[TARGET])]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(detail["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    if "feature_importance" in results:
        st.divider()
        st.markdown("##### Feature Importance (Random Forest)")
        fi = results["feature_importance"]
        fig, ax = plt.subplots(figsize=(8, 5))
        fi.sort_values().plot(kind="barh", ax=ax, color="#5da5da")
        ax.set_xlabel("Importance (Gini)")
        fig.tight_layout()
        st.pyplot(fig, width='stretch')
        plt.close(fig)

    if "ranking" in results:
        st.divider()
        st.markdown("##### 🚨 Clinical Priority Ranking")
        ranking_df = results["ranking"]
        display_cols = ["priority_rank"] + [c for c in ranking_df.columns if c != "priority_rank"]
        st.dataframe(ranking_df[display_cols].head(20), width='stretch', hide_index=True)

        k_values = [k for k in (10, 20, 50) if k <= len(ranking_df)]
        if k_values:
            st.markdown("###### Precision@K")
            precisions = precision_at_k(ranking_df, k_values)
            col_p1, col_p2 = st.columns([1, 2])
            with col_p1:
                st.dataframe(
                    pd.DataFrame({"K": list(precisions.keys()), "Precision@K": [round(v, 3) for v in precisions.values()]}),
                    hide_index=True, width='stretch',
                )
            with col_p2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar([f"@{k}" for k in precisions], list(precisions.values()), color="#f17c67", edgecolor="black")
                freq_base = (ranking_df["real_class"] == "Abnormal").mean()
                ax.axhline(freq_base, color="#5da5da", linestyle="--", label=f"Base rate ({freq_base:.3f})")
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Precision")
                ax.legend()
                fig.tight_layout()
                st.pyplot(fig, width='stretch')
                plt.close(fig)

        csv_bytes = ranking_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download full ranking (CSV)", csv_bytes, file_name="clinical_priority_ranking.csv")

    if "error_analysis" in results:
        st.divider()
        st.markdown("##### 🔎 Error Analysis")
        ea = results["error_analysis"]
        st.write(f"Total errors: **{ea['n_errors']} / {ea['n_total']}** ({ea['error_rate']*100:.1f}%)")
        st.dataframe(ea["error_pairs"], width='stretch')

        if len(ea["critical_errors"]) > 0:
            st.warning(f"⚠️ Critical errors (Abnormal → Normal): {len(ea['critical_errors'])}")
            st.dataframe(ea["critical_errors"], width='stretch')
        else:
            st.success("No critical errors (Abnormal → Normal) in this run.")