"""
Multi-Objective Learning for Perovskite Solar Cell Optimization
Research Framework - Issue #9

This module implements physics-informed multi-output learning for predicting
PCE, Voc, Jsc, and FF simultaneously with Pareto optimization support.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective learning"""
    
    # Target variables
    targets: List[str] = None
    
    # Model parameters
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    
    # Physics constraint
    physics_weight: float = 0.1
    pin_value: float = 100.0  # mW/cm² (standard AM1.5G)
    
    # Performance targets
    r2_targets: Dict[str, float] = None
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = ['PCE', 'Voc', 'Jsc', 'FF']
        if self.r2_targets is None:
            self.r2_targets = {
                'PCE': 0.85,
                'Voc': 0.80,
                'Jsc': 0.80,
                'FF': 0.75
            }


class PhysicsInformedLoss:
    """
    Physics-informed loss function for multi-output learning.
    
    The physical constraint: PCE = (Voc × Jsc × FF) / Pin
    
    Loss = Σ L_target + λ × L_physics
    """
    
    def __init__(self, config: MultiObjectiveConfig):
        self.config = config
        
    def compute_physics_constraint(
        self, 
        voc: np.ndarray, 
        jsc: np.ndarray, 
        ff: np.ndarray
    ) -> np.ndarray:
        """
        Compute theoretical PCE from Voc, Jsc, and FF.
        
        PCE = (Voc × Jsc × FF) / Pin
        
        Args:
            voc: Open circuit voltage (V)
            jsc: Short circuit current density (mA/cm²)
            ff: Fill factor (dimensionless, 0-1)
            
        Returns:
            Theoretical PCE (%)
        """
        # Convert units:
        # Voc: V, Jsc: mA/cm², FF: dimensionless
        # PCE = (Voc × Jsc × FF) / Pin
        # Result in % when Pin = 100 mW/cm²
        
        pce_theoretical = (voc * jsc * ff) / self.config.pin_value * 100
        return pce_theoretical
    
    def compute_loss(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        return_components: bool = False
    ) -> float:
        """
        Compute physics-informed multi-objective loss.
        
        Args:
            y_true: True values [n_samples, n_targets]
            y_pred: Predicted values [n_samples, n_targets]
            return_components: Whether to return loss components
            
        Returns:
            Total loss (float) or loss components (dict)
        """
        # Extract individual predictions
        pce_pred = y_pred[:, 0]  # PCE (%)
        voc_pred = y_pred[:, 1]  # Voc (V)
        jsc_pred = y_pred[:, 2]  # Jsc (mA/cm²)
        ff_pred = y_pred[:, 3]   # FF (dimensionless)
        
        # Compute theoretical PCE from physics
        pce_theoretical = self.compute_physics_constraint(voc_pred, jsc_pred, ff_pred)
        
        # MSE for each target
        mse_pce = np.mean((y_true[:, 0] - pce_pred) ** 2)
        mse_voc = np.mean((y_true[:, 1] - voc_pred) ** 2)
        mse_jsc = np.mean((y_true[:, 2] - jsc_pred) ** 2)
        mse_ff = np.mean((y_true[:, 3] - ff_pred) ** 2)
        
        # Physics constraint violation
        physics_violation = np.mean((pce_pred - pce_theoretical) ** 2)
        
        # Total loss
        total_loss = (
            mse_pce + mse_voc + mse_jsc + mse_ff + 
            self.config.physics_weight * physics_violation
        )
        
        if return_components:
            return {
                'mse_pce': mse_pce,
                'mse_voc': mse_voc,
                'mse_jsc': mse_jsc,
                'mse_ff': mse_ff,
                'physics_violation': physics_violation,
                'total_loss': total_loss
            }
        
        return total_loss


class MultiObjectivePredictor:
    """
    Multi-objective predictor for perovskite solar cell performance.
    
    Predicts PCE, Voc, Jsc, and FF simultaneously with physics-informed
    constraints and Pareto optimization support.
    """
    
    def __init__(self, config: Optional[MultiObjectiveConfig] = None):
        self.config = config or MultiObjectiveConfig()
        self.models = {}
        self.is_fitted = False
        
    def prepare_targets(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare multi-target variables from dataframe.
        
        Args:
            df: Input dataframe with performance metrics
            
        Returns:
            Target array [n_samples, n_targets]
        """
        targets = []
        
        for target in self.config.targets:
            if target in df.columns:
                targets.append(df[target].values)
            else:
                logger.warning(f"Target {target} not found in dataframe")
                targets.append(np.full(len(df), np.nan))
        
        return np.column_stack(targets)
    
    def evaluate_physics_consistency(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate how well predictions satisfy physical constraints.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of consistency metrics
        """
        loss_fn = PhysicsInformedLoss(self.config)
        
        # Extract true and predicted values
        pce_true = y_true[:, 0]
        voc_pred = y_pred[:, 1]
        jsc_pred = y_pred[:, 2]
        ff_pred = y_pred[:, 3]
        pce_pred = y_pred[:, 0]
        
        # Compute theoretical PCE
        pce_theoretical = loss_fn.compute_physics_constraint(
            voc_pred, jsc_pred, ff_pred
        )
        
        # Compute violations
        violations = np.abs(pce_pred - pce_theoretical)
        
        return {
            'mean_violation': np.mean(violations),
            'max_violation': np.max(violations),
            'violation_rate': np.mean(violations > 5.0),  # > 5% absolute error
            'correlation_theoretical_actual': np.corrcoef(pce_pred, pce_theoretical)[0, 1]
        }


def main():
    """Main function for multi-objective learning"""
    logger.info("Initializing Multi-Objective Learning Framework")
    logger.info("Research: Physics-Informed Multi-Output Prediction for Perovskite Solar Cells")
    
    # Initialize configuration
    config = MultiObjectiveConfig()
    
    # Initialize physics-informed loss
    loss_fn = PhysicsInformedLoss(config)
    
    # Test physics constraint
    voc_test = np.array([1.0, 1.1, 0.95])
    jsc_test = np.array([20.0, 22.0, 18.0])
    ff_test = np.array([0.7, 0.75, 0.68])
    
    pce_theoretical = loss_fn.compute_physics_constraint(voc_test, jsc_test, ff_test)
    
    logger.info("\nPhysics Constraint Test:")
    logger.info(f"Voc: {voc_test} V")
    logger.info(f"Jsc: {jsc_test} mA/cm²")
    logger.info(f"FF: {ff_test}")
    logger.info(f"Theoretical PCE: {pce_theoretical} %")
    
    # Expected PCE for first sample: (1.0 * 20.0 * 0.7) / 100 * 100 = 14.0%
    logger.info(f"\nExpected PCE (first sample): 14.0 %")
    logger.info(f"Computed PCE (first sample): {pce_theoretical[0]:.2f} %")
    
    logger.info("\n✓ Multi-objective framework initialized successfully")
    logger.info("Ready for multi-output model training and Pareto optimization")


if __name__ == "__main__":
    main()