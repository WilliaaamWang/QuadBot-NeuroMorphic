"""Regression utilities for analysing half-centre sweep results.

This module loads the CSV files produced by ``hc_sweep.py`` and fits
simple polynomial regression models to relate the sweep parameters to the
bursting features of the neurons.  Results are written to
``NeuronModels/hc_sweep/fitting`` as text files and PNG plots.

The analysis focuses on four features extracted from the voltage traces:

``interburst_freq`` – mean frequency between burst onsets
``intraburst_freq`` – mean frequency of spikes within bursts
``duty_cycle``      – proportion of the period spent bursting
``mean_spikes_per_burst`` – mean number of spikes in a burst

Only rows where **both** neurons were classified as ``"bursting"`` are
used for the regressions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


DATA_DIR = Path(__file__).resolve().parent.parent / "hc_sweep"
SINGLE_DIR = DATA_DIR / "single_param"
MULTI_DIR = DATA_DIR / "multi_param"
OUT_DIR = DATA_DIR / "fitting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features of interest (columns without the ``_A``/``_B`` suffix)
FEATURES = [
    "interburst_freq",
    "intraburst_freq",
    "duty_cycle",
    "mean_spikes_per_burst",
]


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"File not found: {path}")
        return None
    df = pd.read_csv(path)
    # Drop heavy columns if present
    heavy = [c for c in df.columns if c.startswith("V_trace") or c == "time"]
    df.drop(columns=heavy, inplace=True, errors="ignore")
    return df


def _only_bursting(df: pd.DataFrame) -> pd.DataFrame:
    if "regime_A" not in df.columns:
        return df
    mask = (df["regime_A"] == "bursting") & (df["regime_B"] == "bursting")
    return df.loc[mask].reset_index(drop=True)


def _average_features(df: pd.DataFrame) -> pd.DataFrame:
    new = df.copy()
    for feat in FEATURES:
        a = f"{feat}_A"
        b = f"{feat}_B"
        if a in df.columns and b in df.columns:
            new[feat] = df[[a, b]].mean(axis=1)
    return new


def _remove_outliers(
    df: pd.DataFrame, param: str, feature: str, thresh: float = 1.5
) -> tuple[pd.DataFrame, list[int]]:
    """Remove outliers from ``df`` based on the IQR of ``feature``.

    Returns the cleaned dataframe and the indices of detected outliers
    in the original ``df``.
    """
    sub = df[[param, feature]].dropna()
    if sub.empty:
        return sub, []

    q1, q3 = np.percentile(sub[feature], [25, 75])
    iqr = q3 - q1
    lower = q1 - thresh * iqr
    upper = q3 + thresh * iqr

    mask = (sub[feature] >= lower) & (sub[feature] <= upper)
    cleaned = sub.loc[mask].copy()
    out_idx = sub.index[~mask].tolist()
    return cleaned.reset_index(drop=True), out_idx


def _fit_polynomial(
    df: pd.DataFrame,
    params: Iterable[str],
    feature: str,
    degree: int = 2,
) -> tuple[LinearRegression, float] | None:
    mask = df[feature].notna()
    for p in params:
        mask &= df[p].notna()
    if mask.sum() < 2:
        return None

    X = df.loc[mask, list(params)].values
    y = df.loc[mask, feature].values

    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    r2 = model.score(X, y)
    return model, r2


def _format_equation(model, param_names: list[str]) -> str:
    poly = model.named_steps["polynomialfeatures"]
    lin = model.named_steps["linearregression"]
    terms = poly.get_feature_names_out(param_names)
    coeffs = lin.coef_
    parts = [f"{lin.intercept_:.6f}"]
    for c, t in zip(coeffs, terms):
        parts.append(f"({c:.6f}*{t})")
    return " + ".join(parts)


def _plot_fit(
    df: pd.DataFrame,
    param: str,
    feature: str,
    model_before,
    model_after,
    outlier_idx: Iterable[int],
    out_path: Path,
) -> None:
    """Plot scatter with outliers marked and both regression lines."""
    x = df[param].values.reshape(-1, 1)
    order = np.argsort(x[:, 0])
    x_sorted = x[order]
    y_pred_before = model_before.predict(x_sorted)
    y_pred_after = model_after.predict(x_sorted)

    plt.figure(figsize=(5, 4))
    plt.scatter(df[param], df[feature], s=15, color="tab:blue", label="data")
    if outlier_idx:
        plt.scatter(
            df.loc[outlier_idx, param],
            df.loc[outlier_idx, feature],
            color="tab:red",
            marker="x",
            label="outliers",
        )
    plt.plot(x_sorted[:, 0], y_pred_before, "k--", linewidth=1.2, label="before")
    plt.plot(x_sorted[:, 0], y_pred_after, color="tab:orange", linewidth=1.5, label="after")
    plt.xlabel(param)
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_multidim_fit(
    df: pd.DataFrame,
    params: list[str],
    feature: str,
    model,
    out_path: Path,
) -> None:
    """Plot a fit for multiple parameters.

    For two parameters a 3-D surface is drawn.  For higher dimensions a
    simple predicted-vs-actual scatter is saved instead.
    """
    X = df[params].values
    y = df[feature].values

    if X.shape[1] == 2:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        x1 = X[:, 0]
        x2 = X[:, 1]
        grid_x1 = np.linspace(x1.min(), x1.max(), 30)
        grid_x2 = np.linspace(x2.min(), x2.max(), 30)
        gx1, gx2 = np.meshgrid(grid_x1, grid_x2)
        X_grid = np.column_stack([gx1.ravel(), gx2.ravel()])
        y_pred = model.predict(X_grid).reshape(gx1.shape)

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x1, x2, y, color="tab:blue", s=15)
        ax.plot_surface(gx1, gx2, y_pred, alpha=0.3, color="tab:orange")
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        ax.set_zlabel(feature)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    else:
        y_pred = model.predict(X)
        plt.figure(figsize=(4, 4))
        plt.scatter(y, y_pred, s=15, color="tab:blue")
        mn, mx = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def analyse_single_param(param: str, degree: int = 2) -> None:
    path = SINGLE_DIR / f"{param}_sweep.csv"
    df = _load_csv(path)
    if df is None:
        return

    df = _only_bursting(_average_features(df))
    if df.empty:
        print(f"No bursting rows for {param}")
        return

    param_dir = OUT_DIR / param
    param_dir.mkdir(parents=True, exist_ok=True)

    # Correlations for the raw data
    corrs: dict[str, float] = {}
    for feat in FEATURES:
        if feat in df:
            corrs[feat] = df[param].corr(df[feat])

    with open(param_dir / "correlation.json", "w") as f:
        json.dump(corrs, f, indent=2)

    # Fits and plots per feature
    for feat in FEATURES:
        if feat not in df:
            continue

        feature_dir = param_dir / feat
        feature_dir.mkdir(parents=True, exist_ok=True)

        before = _fit_polynomial(df, [param], feat, degree)
        if not before:
            continue
        model_before, r2_before = before

        cleaned, out_idx = _remove_outliers(df, param, feat)
        after = _fit_polynomial(cleaned, [param], feat, degree)
        if not after:
            continue
        model_after, r2_after = after

        eq_before = _format_equation(model_before, [param])
        eq_after = _format_equation(model_after, [param])

        with open(feature_dir / f"fit_{degree}.txt", "w") as f:
            f.write(f"Before: {eq_before} (R2={r2_before:.4f})\n")
            f.write(f"After: {eq_after} (R2={r2_after:.4f})\n")

        _plot_fit(
            df,
            param,
            feat,
            model_before,
            model_after,
            out_idx,
            feature_dir / f"fit_{degree}.png",
        )


def analyse_group(name: str, params: list[str], degree: int = 1) -> None:
    path = MULTI_DIR / f"{name}_sweep.csv"
    df = _load_csv(path)
    if df is None:
        return

    df = _only_bursting(_average_features(df))
    if df.empty:
        print(f"No bursting rows for group {name}")
        return

    group_dir = OUT_DIR / name
    group_dir.mkdir(parents=True, exist_ok=True)

    for feat in FEATURES:
        if feat not in df:
            continue
        feature_dir = group_dir / feat
        feature_dir.mkdir(parents=True, exist_ok=True)

        res = _fit_polynomial(df, params, feat, degree)
        if not res:
            continue
        model, r2 = res
        eq = _format_equation(model, params)
        with open(feature_dir / f"fit_{degree}.txt", "w") as f:
            f.write(f"Equation: {eq}\nR2: {r2:.4f}\n")
        _plot_multidim_fit(df, params, feat, model, feature_dir / f"fit_{degree}.png")



# New helper function to load and combine all single-param and group sweep data
def _load_all_data() -> pd.DataFrame:
    """Load and merge all single-parameter and multi-parameter sweep results (bursting only)."""
    frames = []
    # List of all individual parameters (from single_param sweeps)
    all_params = [
        "Vs0", "Vus0", "g_us", "delta_Vus", "tau_us",
        "g_s", "tau_s", "g_syn_i", "tau_i", "Vi_threshold", "Vi0",
    ]
    # Load each single-parameter sweep CSV
    for param in all_params:
        df = _load_csv(SINGLE_DIR / f"{param}_sweep.csv")
        if df is None:
            continue
        df = _only_bursting(_average_features(df))
        if df.empty:
            continue
        frames.append(df)
    # Load each multi-parameter group sweep CSV
    for name, params in {
        "balance_point": ["Vs0", "Vus0"],
        "ultraslow_dynamics": ["g_us", "delta_Vus", "tau_us"],
        "slow_dynamics": ["g_s", "tau_s"],
        "synaptic": ["g_syn_i", "tau_i", "Vi_threshold", "Vi0"],
    }.items():
        df = _load_csv(MULTI_DIR / f"{name}_sweep.csv")
        if df is None:
            continue
        df = _only_bursting(_average_features(df))
        if df.empty:
            continue
        frames.append(df)
    # Concatenate all data and drop duplicate parameter sets (if any)
    if not frames:
        return pd.DataFrame()  # no data
    combined = pd.concat(frames, ignore_index=True)
    # Drop exact duplicate rows (e.g., identical param combinations from overlapping sweeps)
    param_cols = [c for c in combined.columns if c in all_params]
    combined = combined.drop_duplicates(subset=param_cols).reset_index(drop=True)
    return combined

# New function to analyze one feature using all parameters
def analyse_all_params_for_feature(feature: str, df: pd.DataFrame, degree: int = 1) -> None:
    if feature not in df.columns:
        return
    feature_dir = OUT_DIR / feature
    feature_dir.mkdir(parents=True, exist_ok=True)
    # Fit polynomial model with all parameters for this feature
    params = [
        "Vs0", "Vus0", "g_us", "delta_Vus", "tau_us",
        "g_s", "tau_s", "g_syn_i", "tau_i", "Vi_threshold", "Vi0",
    ]
    res = _fit_polynomial(df, params, feature, degree)
    if not res:
        return
    model, r2 = res
    # Save equation and R²
    eq = _format_equation(model, params)
    with open(feature_dir / f"fit_{degree}.txt", "w") as f:
        f.write(f"Equation: {eq}\nR2: {r2:.4f}\n")
    # Save regression coefficients (including intercept) as JSON
    poly = model.named_steps["polynomialfeatures"]
    lin = model.named_steps["linearregression"]
    terms = poly.get_feature_names_out(params)
    coeffs = lin.coef_
    intercept = lin.intercept_
    # If a bias term '1' is present, incorporate it into the intercept for clarity
    if terms.size > 0 and terms[0] == "1":
        intercept += coeffs[0]
        terms = terms[1:]
        coeffs = coeffs[1:]
    coeff_dict = {"Intercept": round(float(intercept), 6)}
    for term, coef in zip(terms, coeffs):
        coeff_dict[str(term)] = round(float(coef), 6)
    with open(feature_dir / "coefficients.json", "w") as f:
        json.dump(coeff_dict, f, indent=2)
    # Plot predicted vs actual for this feature
    _plot_multidim_fit(df, params, feature, model, feature_dir / f"fit_{degree}.png")

# --- Inside main(), after existing analyses of single_params and param_groups ---
    # Analyze each feature using all parameters (multi-parameter regression)
    combined_df = _load_all_data()
    if combined_df.empty:
        print("No data available for multi-parameter analysis.")
    else:
        for feat in FEATURES:
            analyse_all_params_for_feature(feat, combined_df, degree=1)



def main() -> None:
    single_params = [
        "Vs0",
        "Vus0",
        "g_us",
        "delta_Vus",
        "tau_us",
        "g_s",
        "tau_s",
        "g_syn_i",
        "tau_i",
        "Vi_threshold",
        "Vi0",
    ]

    param_groups = {
        "balance_point": ["Vs0", "Vus0"],
        "ultraslow_dynamics": ["g_us", "delta_Vus", "tau_us"],
        "slow_dynamics": ["g_s", "tau_s"],
        "synaptic": ["g_syn_i", "tau_i", "Vi_threshold", "Vi0"],
    }

    for p in single_params:
        analyse_single_param(p, degree=2)

    for name, params in param_groups.items():
        analyse_group(name, params, degree=1)


if __name__ == "__main__":
    main()