"""Improved regression analysis for half-centre sweep results.

This module implements regularized regression with cross-validation to build
robust models relating parameters to bursting features. It addresses overfitting
and provides quantitative recipes for parameter adjustment.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

from regression_models import ImprovedRegressor


DATA_DIR = Path(__file__).resolve().parent / "hc_sweep"
SINGLE_DIR = DATA_DIR / "single_param"
MULTI_DIR = DATA_DIR / "multi_param"
OUT_DIR = DATA_DIR / "fitting"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Features of interest
FEATURES = [
    "interburst_freq",
    "intraburst_freq", 
    "duty_cycle",
    "mean_spikes_per_burst",
]

# All parameters from the model
ALL_PARAMS = [
    "Vs0", "Vus0", "g_us", "delta_Vus", "tau_us",
    "g_s", "tau_s", "g_syn_i", "tau_i", "Vi_threshold", "Vi0",
]
DEFAULT_PARAMS = {
    'cap': 0.82, 'k': 250.0,
    'V0': -52.0, 'Vs0': -50.0, 'Vus0': -52.0,
    'g_f': 1.0, 'g_s': 0.5, 'g_us': 0.015,
    'tau_s': 4.3, 'tau_us': 278.0,
    'V_threshold': 20.0, 'V_peak': 20.0,
    'V_reset': -45.0, 'Vs_reset': 7.5, 'delta_Vus': 1.7,
    'Ve0': 0.0, 'Vi0': -90.0,
    'Ve_threshold': -20.0, 'Vi_threshold': -20.0,
    'tau_e': 50.0, 'tau_i': 50.0,
    'g_syn_e': 0.5, 'g_syn_i': 30.0
}


# class ImprovedRegressor:
#     """Regularized polynomial regression with cross-validation."""
    
#     def __init__(self, degree: int = 2, alpha: float = 1.0, 
#                  model_type: str = "ridge", cv_folds: int = 5):
#         """Initialize the regressor.
        
#         Parameters
#         ----------
#         degree : int
#             Polynomial degree for feature expansion
#         alpha : float
#             Regularization strength
#         model_type : str
#             Type of regularization: "ridge", "lasso", or "elastic"
#         cv_folds : int
#             Number of cross-validation folds
#         """
#         self.degree = degree
#         self.alpha = alpha
#         self.model_type = model_type
#         self.cv_folds = cv_folds
#         self.pipeline = None
#         self.feature_names = None
#         self.param_names = None
#         self.scaler_params = None  # Store scaling parameters
        
#     def _create_pipeline(self):
#         """Create the sklearn pipeline with scaling and regularization."""
#         # Choose regularization model
#         if self.model_type == "ridge":
#             model = Ridge(alpha=self.alpha)
#         elif self.model_type == "lasso":
#             model = Lasso(alpha=self.alpha, max_iter=5000)
#         elif self.model_type == "elastic":
#             model = ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=5000)
#         else:
#             raise ValueError(f"Unknown model type: {self.model_type}")
            
#         # Build pipeline with scaling
#         self.pipeline = Pipeline([
#             ('scaler', StandardScaler()),
#             ('poly', PolynomialFeatures(self.degree, include_bias=True)),
#             ('model', model)
#         ])
        
#     def fit(self, X: np.ndarray, y: np.ndarray, 
#             param_names: List[str]) -> Tuple[float, float]:
#         """Fit the model with cross-validation.
        
#         Returns
#         -------
#         train_r2 : float
#             R² score on training data
#         cv_r2 : float
#             Mean R² score from cross-validation
#         """
#         self.param_names = param_names
#         self._create_pipeline()
        
#         # Fit the model
#         self.pipeline.fit(X, y)
        
#         # Store feature names after polynomial expansion
#         poly = self.pipeline.named_steps['poly']
#         self.feature_names = poly.get_feature_names_out(param_names)
        
#         # Store scaling parameters for later use
#         scaler = self.pipeline.named_steps['scaler']
#         self.scaler_params = {
#             'mean': scaler.mean_,
#             'scale': scaler.scale_
#         }
        
#         # Calculate scores
#         train_pred = self.pipeline.predict(X)
#         train_r2 = r2_score(y, train_pred)
        
#         # Cross-validation
#         cv_scores = cross_val_score(self.pipeline, X, y, 
#                                    cv=KFold(self.cv_folds, shuffle=True, random_state=42),
#                                    scoring='r2')
#         cv_r2 = cv_scores.mean()
        
#         return train_r2, cv_r2
    
#     def get_coefficients(self) -> Dict[str, float]:
#         """Extract coefficients including intercept."""
#         model = self.pipeline.named_steps['model']
#         coeffs = {'intercept': float(model.intercept_)}
        
#         for name, coef in zip(self.feature_names, model.coef_):
#             if abs(coef) > 1e-10:  # Only include non-zero coefficients
#                 coeffs[name] = float(coef)
                
#         return coeffs
    
#     def predict_change(self, param_changes: Dict[str, float], 
#                       current_params: Dict[str, float]) -> float:
#         """Predict the change in feature value given parameter changes.
        
#         Parameters
#         ----------
#         param_changes : dict
#             Changes to apply to parameters (deltas)
#         current_params : dict
#             Current parameter values
        
#         Returns
#         -------
#         float
#             Predicted change in feature value
#         """
#         # Create current and new parameter arrays
#         current_x = np.array([current_params[p] for p in self.param_names])
#         new_params = current_params.copy()
#         new_params.update({k: current_params[k] + v 
#                           for k, v in param_changes.items()})
#         new_x = np.array([new_params[p] for p in self.param_names])
        
#         # Predict both values
#         current_pred = self.pipeline.predict(current_x.reshape(1, -1))[0]
#         new_pred = self.pipeline.predict(new_x.reshape(1, -1))[0]
        
#         return new_pred - current_pred
    
#     def compute_sensitivities(self, X: np.ndarray) -> Dict[str, float]:
#         """Compute parameter sensitivities at the mean point.
        
#         Returns a dictionary mapping each parameter to its sensitivity
#         (derivative of the model with respect to that parameter).
#         """
#         # Use mean values as reference point
#         x_mean = X.mean(axis=0)
        
#         # Transform to polynomial features
#         scaler = self.pipeline.named_steps['scaler']
#         poly = self.pipeline.named_steps['poly']
#         model = self.pipeline.named_steps['model']
        
#         x_scaled = scaler.transform(x_mean.reshape(1, -1))
#         x_poly = poly.transform(x_scaled)
        
#         # Compute sensitivities
#         sensitivities = {}
#         for i, param in enumerate(self.param_names):
#             # Derivative with respect to scaled parameter
#             deriv = 0
#             for j, feat_name in enumerate(self.feature_names):
#                 coef = model.coef_[j]
#                 # Check if this feature contains our parameter
#                 if param in feat_name:
#                     # For linear terms
#                     if feat_name == param:
#                         deriv += coef
#                     # For quadratic terms
#                     elif feat_name == f"{param}^2":
#                         deriv += 2 * coef * x_scaled[0, i]
#                     # For interaction terms
#                     elif ' ' in feat_name and param in feat_name.split(' '):
#                         other_params = [p for p in feat_name.split(' ') if p != param]
#                         if len(other_params) == 1:
#                             other_idx = self.param_names.index(other_params[0])
#                             deriv += coef * x_scaled[0, other_idx]
            
#             # Adjust for scaling
#             sensitivities[param] = deriv / scaler.scale_[i]
            
#         return sensitivities


def load_and_prepare_data() -> pd.DataFrame:
    """Load all sweep data and prepare for analysis.
    
    This function handles the fact that single-parameter sweeps only record
    the varied parameter, while other parameters are at default values.
    It fills in these defaults and reorganizes columns for clarity.
    """
    frames = []
    
    # Load single-parameter sweeps
    for param in ALL_PARAMS:
        path = SINGLE_DIR / f"{param}_sweep.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        # Remove heavy columns
        df = df.drop(columns=[c for c in df.columns 
                             if c.startswith('V_trace') or c == 'time'], 
                    errors='ignore')
        
        # Fill in default values for parameters not in this sweep
        # This is crucial: single-parameter sweeps don't record unchanging parameters
        for p in ALL_PARAMS:
            if p not in df.columns and p in DEFAULT_PARAMS:
                df[p] = DEFAULT_PARAMS[p]
        
        frames.append(df)
    
    # Load multi-parameter sweeps
    groups = {
        "balance_point": ["Vs0", "Vus0"],
        "ultraslow_dynamics": ["g_us", "delta_Vus", "tau_us"],
        "slow_dynamics": ["g_s", "tau_s"],
        "synaptic": ["g_syn_i", "tau_i", "Vi_threshold", "Vi0"],
    }
    
    for name, params in groups.items():
        # Handle pairwise combinations if more than 2 parameters
        if len(params) > 2:
            from itertools import combinations
            for p1, p2 in combinations(params, 2):
                path = MULTI_DIR / f"{name}_{p1}_{p2}_sweep.csv"
                if path.exists():
                    df = pd.read_csv(path)
                    df = df.drop(columns=[c for c in df.columns 
                                         if c.startswith('V_trace') or c == 'time'], 
                                errors='ignore')
                    
                    # Fill defaults for non-swept parameters
                    for p in ALL_PARAMS:
                        if p not in df.columns and p in DEFAULT_PARAMS:
                            df[p] = DEFAULT_PARAMS[p]
                    
                    frames.append(df)
        else:
            path = MULTI_DIR / f"{name}_sweep.csv"
            if path.exists():
                df = pd.read_csv(path)
                df = df.drop(columns=[c for c in df.columns 
                                     if c.startswith('V_trace') or c == 'time'], 
                            errors='ignore')
                
                # Fill defaults for non-swept parameters
                for p in ALL_PARAMS:
                    if p not in df.columns and p in DEFAULT_PARAMS:
                        df[p] = DEFAULT_PARAMS[p]
                
                frames.append(df)
    
    # Combine all data
    if not frames:
        raise ValueError("No data files found!")
        
    combined = pd.concat(frames, ignore_index=True)
    
    print(f"Total rows before filtering: {len(combined)}")
    
    # Filter for bursting neurons only
    mask = (combined['regime_A'] == 'bursting') & (combined['regime_B'] == 'bursting')
    combined = combined[mask].reset_index(drop=True)
    
    print(f"Rows after bursting filter: {len(combined)}")
    
    # Average features between neurons A and B
    # We need to handle NaN values carefully
    print("\nAveraging features between neurons A and B...")
    for feat in FEATURES:
        a, b = f"{feat}_A", f"{feat}_B"
        if a in combined.columns and b in combined.columns:
            # Count NaN values before averaging
            nan_a = combined[a].isna().sum()
            nan_b = combined[b].isna().sum()
            
            # Use nanmean to handle cases where one neuron has valid data
            # This computes mean ignoring NaN values
            combined[feat] = combined[[a, b]].apply(
                lambda row: np.nanmean([row[a], row[b]]) 
                if not (pd.isna(row[a]) and pd.isna(row[b]))
                else np.nan, 
                axis=1
            )
            
            nan_combined = combined[feat].isna().sum()
            print(f"  {feat}: {nan_a} NaN in A, {nan_b} NaN in B, "
                  f"{nan_combined} NaN after averaging")
    
    # Now let's understand why we have NaN values
    print("\nAnalyzing NaN patterns...")
    
    # Check which features have the most NaN values
    nan_counts = {}
    for feat in FEATURES:
        if feat in combined.columns:
            nan_counts[feat] = combined[feat].isna().sum()
    
    print("NaN counts by feature:")
    for feat, count in sorted(nan_counts.items(), key=lambda x: x[1], reverse=True):
        percent = 100 * count / len(combined)
        print(f"  {feat}: {count} ({percent:.1f}%)")
    
    # Analyze the relationship between features and NaN values
    # Often, if interburst_freq is NaN, other burst features will be too
    if all(f in combined.columns for f in FEATURES):
        # Create a mask for rows where all features are valid
        all_valid_mask = ~combined[FEATURES].isna().any(axis=1)
        n_all_valid = all_valid_mask.sum()
        print(f"\nRows with all features valid: {n_all_valid} ({100*n_all_valid/len(combined):.1f}%)")
        
        # Check if certain parameter ranges tend to produce NaN features
        print("\nParameter statistics for rows with all valid features:")
        if n_all_valid > 0:
            valid_data = combined[all_valid_mask]
            for param in ALL_PARAMS[:5]:  # Show first 5 parameters
                if param in valid_data.columns:
                    mean_all = combined[param].mean()
                    mean_valid = valid_data[param].mean()
                    print(f"  {param}: mean_all={mean_all:.3f}, "
                          f"mean_valid={mean_valid:.3f}, "
                          f"diff={mean_valid-mean_all:.3f}")
    
    # Now check for any remaining missing values in parameters
    missing_params = []
    for p in ALL_PARAMS:
        if p in combined.columns:
            n_missing = combined[p].isna().sum()
            if n_missing > 0:
                missing_params.append((p, n_missing))
    
    if missing_params:
        print("Warning: Missing values found in parameters:")
        for param, count in missing_params:
            print(f"  {param}: {count} missing values")
    
    # Remove rows with any missing parameter values
    # This should now preserve most data since we filled in defaults
    before_dropna = len(combined)
    combined = combined.dropna(subset=ALL_PARAMS)
    after_dropna = len(combined)
    
    if before_dropna != after_dropna:
        print(f"Dropped {before_dropna - after_dropna} rows due to missing parameter values")
    
    # Remove duplicate parameter combinations
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=ALL_PARAMS).reset_index(drop=True)
    after_dedup = len(combined)
    
    if before_dedup != after_dedup:
        print(f"Removed {before_dedup - after_dedup} duplicate parameter combinations")
    
    # Reorganize columns: parameters first, then features
    # This makes the dataframe more intuitive to work with
    param_cols = [p for p in ALL_PARAMS if p in combined.columns]
    feature_cols = FEATURES + [f"{feat}_A" for feat in FEATURES] + [f"{feat}_B" for feat in FEATURES]
    feature_cols = [f for f in feature_cols if f in combined.columns]
    other_cols = [c for c in combined.columns if c not in param_cols + feature_cols]
    
    # Reorder columns
    ordered_columns = param_cols + feature_cols + other_cols
    combined = combined[ordered_columns]
    
    print(f"Final dataset: {len(combined)} rows × {len(combined.columns)} columns")
    print(f"Parameter columns: {param_cols}")
    print(f"Feature columns: {[f for f in FEATURES if f in combined.columns]}")
    
    return combined


def analyze_feature(feature: str, df: pd.DataFrame, 
                   output_dir: Path) -> Dict[str, Any]:
    """Analyze one feature with regularized regression.
    
    Returns dictionary with model, scores, and analysis results.
    """
    feature_dir = output_dir / feature
    feature_dir.mkdir(exist_ok=True)
    
    # First, remove rows where this feature is NaN
    valid_mask = ~df[feature].isna()
    df_valid = df[valid_mask].copy()
    
    print(f"\n{'='*60}")
    print(f"Feature: {feature}")
    print(f"Total samples: {len(df)}, Valid samples: {len(df_valid)} "
          f"({100*len(df_valid)/len(df):.1f}%)")
    
    if len(df_valid) < 10:
        print(f"ERROR: Too few valid samples for {feature} (need at least 10)")
        return None
    
    # Prepare data
    X = df_valid[ALL_PARAMS].values
    y = df_valid[feature].values
    
    # Diagnostic information about the feature distribution
    print(f"\nAnalyzing {feature}")
    print(f"  Original samples: {len(y)}")
    print(f"  Feature statistics:")
    print(f"    Mean: {np.mean(y):.6f}")
    print(f"    Std: {np.std(y):.6f}")
    print(f"    Min: {np.min(y):.6f}")
    print(f"    Max: {np.max(y):.6f}")
    
    # Check for constant or near-constant values
    unique_values = np.unique(y)
    print(f"    Unique values: {len(unique_values)}")
    if len(unique_values) < 10:
        print(f"    Values: {unique_values}")
    
    # Remove outliers using IQR - but let's be more careful
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    
    print(f"  IQR analysis:")
    print(f"    Q1 (25th percentile): {q1:.6f}")
    print(f"    Q3 (75th percentile): {q3:.6f}")
    print(f"    IQR: {iqr:.6f}")
    
    # If IQR is too small, use a different approach
    if iqr < 1e-6 or iqr < 0.01 * np.std(y):
        print(f"  WARNING: IQR too small ({iqr:.6f}), using std-based outlier removal")
        mean_y = np.mean(y)
        std_y = np.std(y)
        if std_y > 0:
            # Use 3-sigma rule instead
            mask = np.abs(y - mean_y) <= 3 * std_y
        else:
            # All values are identical, keep all
            print(f"  WARNING: All values are identical, keeping all samples")
            mask = np.ones(len(y), dtype=bool)
    else:
        # Normal IQR-based removal
        lower_bound = q1 - 1.5*iqr
        upper_bound = q3 + 1.5*iqr
        print(f"    Lower bound: {lower_bound:.6f}")
        print(f"    Upper bound: {upper_bound:.6f}")
        mask = (y >= lower_bound) & (y <= upper_bound)
    
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"  Cleaned: {len(y_clean)} samples ({len(y) - len(y_clean)} removed)")
    
    # Try different regularization strengths
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_model = None
    best_cv_r2 = -np.inf
    best_alpha = None
    
    results = []
    for alpha in alphas:
        model = ImprovedRegressor(degree=2, alpha=alpha, model_type="ridge")
        train_r2, cv_r2 = model.fit(X_clean, y_clean, ALL_PARAMS)
        results.append({
            'alpha': alpha,
            'train_r2': train_r2,
            'cv_r2': cv_r2
        })
        
        if cv_r2 > best_cv_r2:
            best_cv_r2 = cv_r2
            best_model = model
            best_alpha = alpha
    
    print(f"  Best alpha: {best_alpha}, CV R²: {best_cv_r2:.4f}")
    
    # Save the best model
    with open(feature_dir / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    # Save coefficients
    coeffs = best_model.get_coefficients()
    with open(feature_dir / "coefficients.json", "w") as f:
        json.dump(coeffs, f, indent=2)
    
    # Compute and save sensitivities
    sensitivities = best_model.compute_sensitivities(X_clean)
    with open(feature_dir / "sensitivities.json", "w") as f:
        json.dump(sensitivities, f, indent=2)
    
    # Plot regularization path
    plt.figure(figsize=(6, 4))
    alphas_plot = [r['alpha'] for r in results]
    train_scores = [r['train_r2'] for r in results]
    cv_scores = [r['cv_r2'] for r in results]
    
    plt.semilogx(alphas_plot, train_scores, 'o-', label='Training R²')
    plt.semilogx(alphas_plot, cv_scores, 's-', label='CV R²')
    plt.xlabel('Regularization strength (α)')
    plt.ylabel('R² Score')
    plt.title(f'{feature} - Regularization Path')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(feature_dir / "regularization_path.png", dpi=150)
    plt.close()
    
    # Plot predicted vs actual
    y_pred = best_model.pipeline.predict(X_clean)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_clean, y_pred, alpha=0.5, s=20)
    min_val = min(y_clean.min(), y_pred.min())
    max_val = max(y_clean.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel(f'Actual {feature}')
    plt.ylabel(f'Predicted {feature}')
    plt.title(f'{feature} - Predicted vs Actual (R² = {best_cv_r2:.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(feature_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()
    
    return {
        'model': best_model,
        'alpha': best_alpha,
        'train_r2': results[alphas.index(best_alpha)]['train_r2'],
        'cv_r2': best_cv_r2,
        'sensitivities': sensitivities,
        'n_samples': len(y_clean)
    }


def create_sensitivity_heatmap(all_results: Dict[str, Dict], output_dir: Path):
    """Create a heatmap showing parameter sensitivities across all features."""
    # Collect sensitivities into a matrix
    sens_data = []
    for feat in FEATURES:
        if feat in all_results:
            sensitivities = all_results[feat]['sensitivities']
            sens_data.append([sensitivities.get(p, 0) for p in ALL_PARAMS])
    
    sens_matrix = np.array(sens_data)
    
    # Normalize each row (feature) to show relative importance
    sens_normalized = np.abs(sens_matrix)
    row_sums = sens_normalized.sum(axis=1, keepdims=True)
    sens_normalized = sens_normalized / row_sums
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(sens_normalized, 
                xticklabels=ALL_PARAMS,
                yticklabels=FEATURES,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Relative Sensitivity'})
    plt.title('Parameter Sensitivities (Normalized by Feature)')
    plt.xlabel('Parameters')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap.png", dpi=150)
    plt.close()
    
    # Also save raw sensitivities
    plt.figure(figsize=(12, 6))
    sns.heatmap(sens_matrix,
                xticklabels=ALL_PARAMS,
                yticklabels=FEATURES,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Sensitivity'})
    plt.title('Parameter Sensitivities (Raw Values)')
    plt.xlabel('Parameters')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap_raw.png", dpi=150)
    plt.close()


def visualize_feature_distributions(df: pd.DataFrame, all_results, output_dir: Path):
    """Create visualizations of feature distributions to diagnose data issues."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES):
        if feature not in df.columns:
            continue
            
        ax = axes[i]
        data = df[feature].dropna()
        
        # Histogram with KDE
        ax.hist(data, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        q1, q3 = np.percentile(data, [25, 75])
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        ax.axvline(q1, color='orange', linestyle=':', linewidth=1, label=f'Q1: {q1:.3f}')
        ax.axvline(q3, color='orange', linestyle=':', linewidth=1, label=f'Q3: {q3:.3f}')
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(f'{feature} Distribution (n={len(data)})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_distributions.png", dpi=150)
    plt.close()
    
    # Also create box plots for outlier visualization
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    features_data = []
    features_labels = []
    
    for feature in FEATURES:
        if feature in df.columns:
            data = df[feature].dropna()
            features_data.append(data)
            features_labels.append(feature)
    
    bp = ax.boxplot(features_data, labels=features_labels, patch_artist=True)
    
    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax.set_ylabel('Feature Value')
    ax.set_title('Feature Distributions - Box Plot View')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_boxplots.png", dpi=150)
    plt.close()
    """Create a heatmap showing parameter sensitivities across all features."""
    # Collect sensitivities into a matrix
    sens_data = []
    for feat in FEATURES:
        if feat in all_results:
            sensitivities = all_results[feat]['sensitivities']
            sens_data.append([sensitivities.get(p, 0) for p in ALL_PARAMS])
    
    sens_matrix = np.array(sens_data)
    
    # Normalize each row (feature) to show relative importance
    sens_normalized = np.abs(sens_matrix)
    row_sums = sens_normalized.sum(axis=1, keepdims=True)
    sens_normalized = sens_normalized / row_sums
    
    # Create heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(sens_normalized, 
                xticklabels=ALL_PARAMS,
                yticklabels=FEATURES,
                cmap='YlOrRd',
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Relative Sensitivity'})
    plt.title('Parameter Sensitivities (Normalized by Feature)')
    plt.xlabel('Parameters')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap.png", dpi=150)
    plt.close()
    
    # Also save raw sensitivities
    plt.figure(figsize=(12, 6))
    sns.heatmap(sens_matrix,
                xticklabels=ALL_PARAMS,
                yticklabels=FEATURES,
                cmap='RdBu_r',
                center=0,
                annot=True,
                fmt='.3f',
                cbar_kws={'label': 'Sensitivity'})
    plt.title('Parameter Sensitivities (Raw Values)')
    plt.xlabel('Parameters')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_heatmap_raw.png", dpi=150)
    plt.close()


def main():
    """Run the complete analysis pipeline."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"Loaded {len(df)} bursting samples with complete parameter sets")
    
    
    # Analyze each feature
    all_results = {}
    for feature in FEATURES:
        if feature in df.columns:
            results = analyze_feature(feature, df, OUT_DIR)
            all_results[feature] = results
    
    # Create sensitivity heatmap
    create_sensitivity_heatmap(all_results, OUT_DIR)
    
    # Visualize feature distributions to diagnose any data issues
    print("\nCreating feature distribution plots...")
    visualize_feature_distributions(df, all_results, OUT_DIR)
    
    # Save summary statistics
    summary = {
        'n_samples': len(df),
        'features': {}
    }
    for feat, res in all_results.items():
        summary['features'][feat] = {
            'cv_r2': res['cv_r2'],
            'train_r2': res['train_r2'],
            'alpha': res['alpha'],
            'n_samples_clean': res['n_samples']
        }
    
    with open(OUT_DIR / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\nAnalysis complete! Results saved to:", OUT_DIR)

if __name__ == "__main__":
    main()