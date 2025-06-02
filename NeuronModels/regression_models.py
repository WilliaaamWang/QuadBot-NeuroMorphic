"""Shared regression models for half-centre analysis.

This module contains the regression classes used by both model_fitting.py
and generate_recipes.py. By keeping these in a separate module, we ensure
that pickle can properly serialize and deserialize the models.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score


class ImprovedRegressor:
    """Regularized polynomial regression with cross-validation.
    
    This class wraps sklearn models with additional functionality for
    sensitivity analysis and parameter change prediction. It's designed
    to be pickle-able for saving trained models.
    """
    
    def __init__(self, degree: int = 2, alpha: float = 1.0, 
                 model_type: str = "ridge", cv_folds: int = 5):
        """Initialize the regressor.
        
        Parameters
        ----------
        degree : int
            Polynomial degree for feature expansion
        alpha : float
            Regularization strength (higher = more regularization)
        model_type : str
            Type of regularization: "ridge", "lasso", or "elastic"
        cv_folds : int
            Number of cross-validation folds
        """
        self.degree = degree
        self.alpha = alpha
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.pipeline = None
        self.feature_names = None
        self.param_names = None
        self.scaler_params = None  # Store scaling parameters
        
    def _create_pipeline(self):
        """Create the sklearn pipeline with scaling and regularization."""
        # Choose regularization model based on type
        if self.model_type == "ridge":
            model = Ridge(alpha=self.alpha)
        elif self.model_type == "lasso":
            model = Lasso(alpha=self.alpha, max_iter=5000)
        elif self.model_type == "elastic":
            model = ElasticNet(alpha=self.alpha, l1_ratio=0.5, max_iter=5000)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        # Build pipeline: Scale → Polynomial Features → Regularized Linear Model
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('poly', PolynomialFeatures(self.degree, include_bias=True)),
            ('model', model)
        ])
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            param_names: List[str]) -> Tuple[float, float]:
        """Fit the model with cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features)
        y : np.ndarray
            Target values (n_samples,)
        param_names : List[str]
            Names of the parameters (features)
        
        Returns
        -------
        train_r2 : float
            R² score on training data
        cv_r2 : float
            Mean R² score from cross-validation
        """
        self.param_names = param_names
        self._create_pipeline()
        
        # Fit the model
        self.pipeline.fit(X, y)
        
        # Store feature names after polynomial expansion
        poly = self.pipeline.named_steps['poly']
        self.feature_names = poly.get_feature_names_out(param_names)
        
        # Store scaling parameters for interpretability
        scaler = self.pipeline.named_steps['scaler']
        self.scaler_params = {
            'mean': scaler.mean_,
            'scale': scaler.scale_
        }
        
        # Calculate training score
        train_pred = self.pipeline.predict(X)
        train_r2 = r2_score(y, train_pred)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            self.pipeline, X, y, 
            cv=KFold(self.cv_folds, shuffle=True, random_state=42),
            scoring='r2'
        )
        cv_r2 = cv_scores.mean()
        
        return train_r2, cv_r2
    
    def get_coefficients(self) -> Dict[str, float]:
        """Extract coefficients including intercept.
        
        Returns a dictionary mapping feature names to their coefficients.
        Only includes non-zero coefficients (above threshold).
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before getting coefficients")
            
        model = self.pipeline.named_steps['model']
        coeffs = {'intercept': float(model.intercept_)}
        
        # Add non-zero coefficients
        for name, coef in zip(self.feature_names, model.coef_):
            if abs(coef) > 1e-10:  # Only include meaningful coefficients
                coeffs[name] = float(coef)
                
        return coeffs
    
    def predict_change(self, param_changes: Dict[str, float], 
                      current_params: Dict[str, float]) -> float:
        """Predict the change in feature value given parameter changes.
        
        This is useful for understanding how adjusting parameters will
        affect the output feature.
        
        Parameters
        ----------
        param_changes : dict
            Changes to apply to parameters (deltas, not fractions)
        current_params : dict
            Current parameter values
        
        Returns
        -------
        float
            Predicted change in feature value
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before prediction")
            
        # Create arrays for current and new parameter values
        current_x = np.array([current_params[p] for p in self.param_names])
        
        # Apply changes to get new parameter values
        new_params = current_params.copy()
        new_params.update({
            k: current_params.get(k, 0) + v 
            for k, v in param_changes.items()
        })
        new_x = np.array([new_params[p] for p in self.param_names])
        
        # Predict both values and return the difference
        current_pred = self.pipeline.predict(current_x.reshape(1, -1))[0]
        new_pred = self.pipeline.predict(new_x.reshape(1, -1))[0]
        
        return new_pred - current_pred
    
    def compute_sensitivities(self, X: np.ndarray) -> Dict[str, float]:
        """Compute parameter sensitivities at the mean point.
        
        Sensitivities tell us how much the output changes for a unit
        change in each input parameter. This is essentially the partial
        derivative of the model with respect to each parameter.
        
        Parameters
        ----------
        X : np.ndarray
            Training data used to determine the mean operating point
        
        Returns
        -------
        Dict[str, float]
            Sensitivity (derivative) for each parameter
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before computing sensitivities")
            
        # Use mean values as reference point
        x_mean = X.mean(axis=0)
        
        # Get pipeline components
        scaler = self.pipeline.named_steps['scaler']
        poly = self.pipeline.named_steps['poly']
        model = self.pipeline.named_steps['model']
        
        # Transform the mean point
        x_scaled = scaler.transform(x_mean.reshape(1, -1))
        x_poly = poly.transform(x_scaled)
        
        # Compute sensitivities for each parameter
        sensitivities = {}
        
        for i, param in enumerate(self.param_names):
            # Calculate derivative with respect to this parameter
            deriv = 0
            
            for j, feat_name in enumerate(self.feature_names):
                coef = model.coef_[j]
                
                # Parse the feature name to understand its structure
                # Linear terms: just the parameter name
                if feat_name == param:
                    deriv += coef
                
                # Quadratic terms: parameter^2
                elif feat_name == f"{param}^2":
                    deriv += 2 * coef * x_scaled[0, i]
                
                # Interaction terms: parameter1 parameter2
                elif ' ' in feat_name and param in feat_name.split(' '):
                    # This is an interaction term involving our parameter
                    other_params = [p for p in feat_name.split(' ') if p != param]
                    if len(other_params) == 1:
                        # Two-way interaction
                        other_idx = self.param_names.index(other_params[0])
                        deriv += coef * x_scaled[0, other_idx]
            
            # Adjust for scaling (chain rule)
            # The derivative with respect to the original (unscaled) parameter
            # needs to account for the scaling transformation
            sensitivities[param] = deriv / scaler.scale_[i]
            
        return sensitivities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Predicted values (n_samples,)
        """
        if self.pipeline is None:
            raise ValueError("Model must be fitted before prediction")
        return self.pipeline.predict(X)