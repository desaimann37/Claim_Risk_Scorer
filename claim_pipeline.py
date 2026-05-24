"""
claim_pipeline.py
------------------
Shared logic for the Auto Insurance Claim-Frequency project.

Business framing
================
Insurers price a policy roughly as:   PURE PREMIUM  =  CLAIM FREQUENCY  x  CLAIM SEVERITY
This project models the FREQUENCY half: how likely a policy is to file >= 1 claim.

Dataset
=======
French Motor Third-Party Liability (freMTPL2freq) -- the canonical public auto-insurance
dataset. Loaded from OpenML. If OpenML is unreachable (e.g. offline), we generate a
SYNTHETIC dataset with the SAME schema so the whole pipeline still runs and is testable.
On your own machine the real data will load automatically.

Key real-world subtlety: EXPOSURE. A policy active for 6 months has Exposure=0.5 and a
fair chance to accumulate fewer claims than one active a full year. We turn the raw claim
COUNT into a binary "had a claim" target for a clean classification demo, but we keep
exposure around so you can talk about it intelligently in the interview.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# Columns we model with, mirroring the real freMTPL2freq schema.
NUMERIC_FEATURES = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
CATEGORICAL_FEATURES = ["VehBrand", "VehGas", "Region", "Area"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data():
    """Return (df, source_str). Tries real OpenML data, falls back to synthetic."""
    try:
        from sklearn.datasets import fetch_openml
        ds = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto")
        df = ds.frame.copy()
        # Normalise dtypes
        for c in NUMERIC_FEATURES + ["Exposure", "ClaimNb"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["ClaimNb", "Exposure"])
        return df, "REAL freMTPL2freq (OpenML)"
    except Exception as e:  # offline / blocked -> synthetic with identical schema
        return _synthetic_data(), f"SYNTHETIC fallback ({type(e).__name__})"


def _synthetic_data(n=60000):
    """Realistic stand-in with the same columns + a sensible risk structure."""
    rng = np.random.default_rng(RANDOM_STATE)
    driv_age = rng.integers(18, 85, n)
    veh_age = rng.integers(0, 25, n)
    veh_power = rng.integers(4, 16, n)
    bonus_malus = np.clip(rng.normal(60, 25, n), 50, 230).round().astype(int)
    density = np.clip(rng.lognormal(4.5, 1.3, n), 1, 30000).round().astype(int)
    exposure = np.clip(rng.beta(2, 2, n), 0.02, 1.0)
    veh_brand = rng.choice([f"B{i}" for i in range(1, 15)], n)
    veh_gas = rng.choice(["Regular", "Diesel"], n)
    region = rng.choice([f"R{i}" for i in range(1, 23)], n)
    area = rng.choice(list("ABCDEF"), n)

    # Latent risk: young + old drivers riskier, high bonus_malus riskier,
    # denser areas riskier, more powerful vehicles slightly riskier.
    age_risk = 0.00045 * (driv_age - 45) ** 2
    lin = (-3.6 + age_risk + 0.012 * (bonus_malus - 60)
           + 0.10 * np.log1p(density) + 0.04 * veh_power
           + 0.015 * veh_age + np.log(np.clip(exposure, 0.05, 1)))
    lam = np.exp(lin)
    claim_nb = rng.poisson(lam)

    return pd.DataFrame({
        "ClaimNb": claim_nb, "Exposure": exposure.round(4),
        "VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age,
        "BonusMalus": bonus_malus, "Density": density,
        "VehBrand": veh_brand, "VehGas": veh_gas, "Region": region, "Area": area,
    })


def prepare(df):
    """Clean, cap outliers, build binary target. Returns (X, y, df_clean)."""
    df = df.copy()
    # Keep only the columns we use (real dataset has a few extras / IDPol).
    keep = [c for c in (["ClaimNb", "Exposure"] + ALL_FEATURES) if c in df.columns]
    df = df[keep]
    # Defensive cleaning: drop impossible values, cap extreme claim counts.
    df = df[(df["Exposure"] > 0) & (df["DrivAge"].between(18, 100))]
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    # Binary target: did this policy have at least one claim?
    y = (df["ClaimNb"] > 0).astype(int)
    X = df[ALL_FEATURES].copy()
    return X, y, df


def build_preprocessor():
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES),
                              ("cat", cat, CATEGORICAL_FEATURES)])


def train_models(X_train, y_train):
    """Train an interpretable GLM (logistic regression) and an XGBoost model."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    # Class imbalance: most policies never claim. We tell the models to care.
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos = neg / max(pos, 1)

    glm = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])
    glm.fit(X_train, y_train)

    xgb = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    xgb.fit(X_train, y_train)
    return {"GLM (Logistic Regression)": glm, "XGBoost": xgb}


def evaluate(model, X_test, y_test):
    """Return a dict of the metrics that MATTER for a rare-event problem."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  precision_score, recall_score, f1_score,
                                  accuracy_score)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "Accuracy (misleading here!)": accuracy_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, proba),
        "PR-AUC (avg precision)": average_precision_score(y_test, proba),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "_proba": proba,
    }


def feature_names(fitted_pipeline):
    """Recover human-readable feature names after one-hot encoding."""
    prep = fitted_pipeline.named_steps["prep"]
    names = list(NUMERIC_FEATURES)
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    names += list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    return names


if __name__ == "__main__":
    # Smoke test the whole pipeline.
    from sklearn.model_selection import train_test_split
    df, src = load_data()
    print("Data source:", src, "| rows:", len(df))
    X, y, _ = prepare(df)
    print("Claim rate (positive class):", round(y.mean(), 4))
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    models = train_models(Xtr, ytr)
    for name, m in models.items():
        res = evaluate(m, Xte, yte)
        print(f"\n{name}")
        for k, v in res.items():
            if not k.startswith("_"):
                print(f"  {k:32s} {v:.4f}")
