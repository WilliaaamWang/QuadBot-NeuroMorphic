# model_fitting.py

import pandas as pd
import numpy as np

# Optional: Import libraries for symbolic regression and ML models
try:
    from gplearn.genetic import SymbolicRegressor
except ImportError:
    SymbolicRegressor = None
try:
    import pysr
except ImportError:
    pysr = None
from sklearn.ensemble import RandomForestRegressor

# Define which groups to analyze (matching the sweep CSV files produced)
groups_to_fit = {
    'group1_Vs0_Vus0': ['Vs0', 'Vus0'],
    'group2_gus_deltaTau': ['g_us', 'delta_Vus', 'tau_us'],
    'group3_gs_tau': ['g_s', 'tau_s'],
    'group4_synaptic': ['g_syn_i', 'tau_i', 'Vi_threshold', 'Vi0']
}

# Iterate through each group's results
for group, param_names in groups_to_fit.items():
    filename = f"sweep_{group}.csv"
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Results file {filename} not found. Skipping.")
        continue
    print(f"\nAnalyzing {group} (parameters: {param_names})")
    # Features in the data
    features = ['interburst_freq', 'intraburst_freq', 'spikes_per_burst', 'duty_cycle']
    X = df[param_names].values
    # For each feature, attempt to find a relationship
    for feature in features:
        y = df[feature].values
        print(f"  Feature: {feature}")
        # Determine modeling approach based on dimensionality of input
        dim = X.shape[1]
        if dim <= 2 and SymbolicRegressor is not None:
            # Use genetic programming for symbolic regression (gplearn)
            model = SymbolicRegressor(population_size=5000, generations=20, stopping_criteria=0.01,
                                      p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05, 
                                      p_point_mutation=0.1, max_samples=0.9, verbose=0, random_state=42)
            model.fit(X, y)
            equation = model._program
            print(f"    Discovered formula: {equation}")
        elif dim <= 3 and pysr is not None:
            # Use PySR for symbolic regression if available (for up to 3-parameter relations)
            # PySR may require installation of Julia; this is an optional approach.
            try:
                equations = pysr.pysr(X, y, 
                                       weights=None,
                                       procs=4,
                                       niterations=100,
                                       binary_operators=["+", "-", "*", "/", "pow"],
                                       unary_operators=["sqrt", "log", "exp"],
                                       maxsize=10)
                print("    PySR best equation:", equations.iloc[0]['equation'])
            except Exception as e:
                print(f"    PySR failed: {e}")
        else:
            # For higher dimensions, use a Random Forest as an interpretable model
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            # Rank features by importance
            sorted_idx = np.argsort(importances)[::-1]
            importance_info = ", ".join([f"{param_names[i]}: {importances[i]:.3f}" for i in sorted_idx])
            print(f"    Feature importances (RF): {importance_info}")
            # Additional insight: we can output partial dependence or correlation as needed
            # Here, we print correlation sign for a rough idea of influence direction
            corr = np.corrcoef(X.T, y)[-1, :-1]  # correlation of each param with y
            sign_info = ", ".join([f"{param_names[i]}({'+' if corr[i]>0 else '-'})" for i in range(len(param_names))])
            print(f"    (Correlation signs: {sign_info})")
    print("  Model analysis for group complete.")
