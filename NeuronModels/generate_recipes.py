"""Generate quantitative recipes for achieving desired feature changes.

This module uses the trained regression models to provide specific parameter
adjustment recommendations for achieving target changes in bursting features.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import pandas as pd

# Import the shared regression model
from regression_models import ImprovedRegressor


FITTING_DIR = Path(__file__).parent / "hc_sweep/fitting"
RECIPE_DIR = FITTING_DIR / "recipes"
RECIPE_DIR.mkdir(exist_ok=True)

# Default parameters from the model
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

# Parameters we can modify
MODIFIABLE_PARAMS = [
    "Vs0", "Vus0", "g_us", "delta_Vus", "tau_us",
    "g_s", "tau_s", "g_syn_i", "tau_i", "Vi_threshold", "Vi0",
]

# Reasonable bounds for parameter changes (as fraction of default)
PARAM_BOUNDS = {
    "Vs0": (-0.2, 0.2),  # ±20% of default
    "Vus0": (-0.2, 0.2),
    "g_us": (-0.5, 2.0),  # Can vary more
    "delta_Vus": (-0.5, 2.0),
    "tau_us": (-0.5, 2.0),
    "g_s": (-0.5, 2.0),
    "tau_s": (-0.5, 2.0),
    "g_syn_i": (-0.5, 2.0),
    "tau_i": (-0.5, 2.0),
    "Vi_threshold": (-0.3, 0.3),
    "Vi0": (-0.2, 0.2),
}


class RecipeGenerator:
    """Generate parameter adjustment recipes for desired feature changes."""
    
    def __init__(self, feature: str):
        """Load the model for the specified feature."""
        self.feature = feature
        self.model_path = FITTING_DIR / feature / "model.pkl"
        self.sensitivity_path = FITTING_DIR / feature / "sensitivities.json"
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"No model found for {feature}")
            
        # Load model and sensitivities
        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        
        with open(self.sensitivity_path, "r") as f:
            self.sensitivities = json.load(f)
    
    def rank_parameters(self) -> List[Tuple[str, float]]:
        """Rank parameters by their influence on this feature.
        
        Returns list of (param_name, sensitivity) tuples sorted by absolute sensitivity.
        """
        ranked = [(p, abs(s)) for p, s in self.sensitivities.items()]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked
    
    def simple_recipe(self, target_change: float, 
                     max_params: int = 3) -> Dict[str, float]:
        """Generate a simple recipe using the most influential parameters.
        
        Parameters
        ----------
        target_change : float
            Desired change in the feature value
        max_params : int
            Maximum number of parameters to adjust
            
        Returns
        -------
        dict
            Recommended parameter changes (as fractions of default values)
        """
        # Get the most influential parameters
        ranked = self.rank_parameters()
        top_params = [p for p, _ in ranked[:max_params] if p in MODIFIABLE_PARAMS]
        
        if not top_params:
            return {}
        
        # Use sensitivities to distribute the change
        total_sensitivity = sum(abs(self.sensitivities[p]) for p in top_params)
        
        recipe = {}
        for param in top_params:
            sensitivity = self.sensitivities[param]
            if abs(sensitivity) < 1e-10:
                continue
                
            # Required change in this parameter to achieve target
            # (distributed proportionally to sensitivity)
            weight = abs(sensitivity) / total_sensitivity
            param_change = (target_change * weight) / sensitivity
            
            # Convert to fraction of default value
            default_val = DEFAULT_PARAMS[param]
            fraction_change = param_change / abs(default_val)
            
            # Apply bounds
            min_frac, max_frac = PARAM_BOUNDS.get(param, (-1, 1))
            fraction_change = np.clip(fraction_change, min_frac, max_frac)
            
            recipe[param] = fraction_change
        
        return recipe
    
    def optimized_recipe(self, target_change: float,
                        preference: str = "minimal",
                        max_params: Optional[int] = None) -> Dict[str, float]:
        """Generate an optimized recipe using constrained optimization.
        
        Parameters
        ----------
        target_change : float
            Desired change in the feature value
        preference : str
            Optimization preference:
            - "minimal": Minimize total parameter changes
            - "balanced": Balance changes across parameters
            - "conservative": Prefer smaller changes to more parameters
        max_params : int, optional
            Maximum number of parameters to adjust
            
        Returns
        -------
        dict
            Optimized parameter changes (as fractions of default values)
        """
        # Set up optimization problem
        n_params = len(MODIFIABLE_PARAMS)
        param_indices = {p: i for i, p in enumerate(MODIFIABLE_PARAMS)}
        
        # Initial guess (no changes)
        x0 = np.zeros(n_params)
        
        # Bounds for each parameter
        bounds = []
        for param in MODIFIABLE_PARAMS:
            if param in DEFAULT_PARAMS:
                min_frac, max_frac = PARAM_BOUNDS.get(param, (-1, 1))
                bounds.append((min_frac, max_frac))
            else:
                bounds.append((0, 0))  # Don't modify if not in defaults
        
        # Objective function based on preference
        if preference == "minimal":
            # Minimize sum of absolute changes
            def objective(x):
                return np.sum(np.abs(x))
        elif preference == "balanced":
            # Minimize sum of squared changes
            def objective(x):
                return np.sum(x**2)
        elif preference == "conservative":
            # Minimize maximum change
            def objective(x):
                return np.max(np.abs(x))
        else:
            raise ValueError(f"Unknown preference: {preference}")
        
        # Constraint: achieve target change
        def constraint_func(x):
            # Convert fractions to actual parameter values
            current_params = DEFAULT_PARAMS.copy()
            param_changes = {}
            for i, param in enumerate(MODIFIABLE_PARAMS):
                if param in DEFAULT_PARAMS:
                    param_changes[param] = x[i] * abs(DEFAULT_PARAMS[param])
            
            # Predict the change
            predicted_change = self.model.predict_change(param_changes, current_params)
            return predicted_change
        
        # Set up constraint
        constraint = {
            'type': 'eq',
            'fun': lambda x: constraint_func(x) - target_change
        }
        
        # Add sparsity constraint if max_params specified
        if max_params is not None:
            # This is a bit tricky with scipy.optimize
            # We'll use a penalty approach instead
            original_objective = objective
            penalty_weight = 1000.0
            
            def penalized_objective(x):
                n_nonzero = np.sum(np.abs(x) > 0.01)  # Count "active" parameters
                penalty = penalty_weight * max(0, n_nonzero - max_params)
                return original_objective(x) + penalty
            
            objective = penalized_objective
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[constraint],
            options={'ftol': 1e-6, 'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Optimization failed: {result.message}")
            # Fall back to simple recipe
            return self.simple_recipe(target_change)
        
        # Extract non-zero changes
        recipe = {}
        for i, param in enumerate(MODIFIABLE_PARAMS):
            if abs(result.x[i]) > 0.01:  # Threshold for considering it non-zero
                recipe[param] = result.x[i]
        
        return recipe
    
    def evaluate_recipe(self, recipe: Dict[str, float],
                       current_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Evaluate a recipe and predict the resulting changes.
        
        Parameters
        ----------
        recipe : dict
            Parameter changes as fractions of default values
        current_params : dict, optional
            Current parameter values (defaults to DEFAULT_PARAMS)
            
        Returns
        -------
        dict
            Evaluation results including predicted change and confidence
        """
        if current_params is None:
            current_params = DEFAULT_PARAMS.copy()
        
        # Convert fractions to actual changes
        param_changes = {}
        for param, fraction in recipe.items():
            if param in current_params:
                param_changes[param] = fraction * abs(current_params[param])
        
        # Predict the change
        predicted_change = self.model.predict_change(param_changes, current_params)
        
        # Estimate confidence based on the magnitude of changes
        # (larger changes -> less confidence due to extrapolation)
        avg_fraction = np.mean(list(np.abs(list(recipe.values()))))
        confidence = max(0, 1 - avg_fraction)  # Simple linear decrease
        
        return {
            'predicted_change': predicted_change,
            'confidence': confidence,
            'n_params_changed': len(recipe),
            'avg_param_change': avg_fraction
        }


def generate_recipe_report(feature: str, target_changes: List[float]):
    """Generate a comprehensive recipe report for a feature.
    
    Parameters
    ----------
    feature : str
        The feature to generate recipes for
    target_changes : list
        List of target changes to consider (e.g., [-20%, -10%, +10%, +20%])
    """
    generator = RecipeGenerator(feature)
    
    report = {
        'feature': feature,
        'parameter_ranking': generator.rank_parameters(),
        'recipes': {}
    }
    
    for target in target_changes:
        recipes = {
            'simple': generator.simple_recipe(target),
            'minimal': generator.optimized_recipe(target, preference='minimal'),
            'balanced': generator.optimized_recipe(target, preference='balanced'),
            'conservative': generator.optimized_recipe(target, preference='conservative')
        }
        
        # Evaluate each recipe
        evaluations = {}
        for name, recipe in recipes.items():
            eval_result = generator.evaluate_recipe(recipe)
            evaluations[name] = {
                'recipe': recipe,
                'evaluation': eval_result,
                'readable_changes': {
                    param: f"{fraction:+.1%} ({fraction * abs(DEFAULT_PARAMS[param]):+.3f})"
                    for param, fraction in recipe.items()
                }
            }
        
        report['recipes'][f"target_{target:+g}"] = evaluations
    
    # Save report
    output_path = RECIPE_DIR / f"{feature}_recipes.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Also create a human-readable summary
    summary_path = RECIPE_DIR / f"{feature}_recipes_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Recipe Report for {feature}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Parameter Importance Ranking:\n")
        for i, (param, sens) in enumerate(report['parameter_ranking'][:5], 1):
            f.write(f"{i}. {param}: sensitivity = {sens:.4f}\n")
        
        f.write("\n" + "-" * 50 + "\n\n")
        
        for target_key, recipes_eval in report['recipes'].items():
            target_val = float(target_key.replace('target_', ''))
            f.write(f"Target Change: {target_val:+g}\n")
            f.write("-" * 30 + "\n")
            
            for recipe_type, data in recipes_eval.items():
                f.write(f"\n{recipe_type.upper()} Recipe:\n")
                
                recipe = data['recipe']
                if not recipe:
                    f.write("  No changes recommended\n")
                    continue
                
                for param, fraction in recipe.items():
                    actual_change = fraction * abs(DEFAULT_PARAMS[param])
                    f.write(f"  {param}: {fraction:+.1%} "
                           f"({DEFAULT_PARAMS[param]:.3f} → "
                           f"{DEFAULT_PARAMS[param] + actual_change:.3f})\n")
                
                eval_data = data['evaluation']
                f.write(f"  Predicted change: {eval_data['predicted_change']:.4f}\n")
                f.write(f"  Confidence: {eval_data['confidence']:.1%}\n")
            
            f.write("\n")


def main():
    """Generate recipes for all features."""
    # Define target changes as percentages of typical feature ranges
    typical_ranges = {
        'interburst_freq': 2.0,      # Hz
        'intraburst_freq': 50.0,     # Hz
        'duty_cycle': 0.5,           # fraction
        'mean_spikes_per_burst': 10  # count
    }
    
    for feature in ['interburst_freq', 'intraburst_freq', 'duty_cycle', 'mean_spikes_per_burst']:
        try:
            # Generate recipes for ±10%, ±20%, ±50% changes
            typical_range = typical_ranges.get(feature, 1.0)
            target_changes = [
                -0.5 * typical_range,
                -0.2 * typical_range,
                -0.1 * typical_range,
                0.1 * typical_range,
                0.2 * typical_range,
                0.5 * typical_range
            ]
            
            generate_recipe_report(feature, target_changes)
            print(f"Generated recipes for {feature}")
            
        except Exception as e:
            print(f"Failed to generate recipes for {feature}: {e}")
    
    print(f"\nRecipes saved to: {RECIPE_DIR}")


if __name__ == "__main__":
    main()