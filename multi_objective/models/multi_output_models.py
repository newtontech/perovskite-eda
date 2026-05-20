"""
Multi-Output Models for Perovskite Solar Cell Performance Prediction
Research: Physics-Informed Multi-Output Learning

This module implements multiple approaches for multi-target prediction:
1. Independent baseline models
2. MultiOutputRegressor (sklearn)
3. Multi-task Neural Network
4. Physics-informed model with constraint loss

Author: OpenClaw AI Assistant
Date: 2026-03-12
Target: Nature Machine Intelligence
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configure matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-whitegrid')


@dataclass
class ModelConfig:
    """Configuration for multi-output models."""
    
    # Targets
    targets: List[str] = None
    
    # Data split
    test_size: float = 0.2
    random_state: int = 42
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 10
    hidden_sizes: List[int] = None
    
    # Training
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    
    # Physics constraint
    physics_weight: float = 0.1
    pin_value: float = 100.0  # mW/cm²
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = ['PCE', 'Voc', 'Jsc', 'FF']
        if self.hidden_sizes is None:
            self.hidden_sizes = [128, 64, 32]


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function for multi-output prediction.
    
    Enforces the physical constraint: PCE = (Voc × Jsc × FF) / Pin
    
    Loss = Σ MSE(target) + λ × MSE(PCE_measured - PCE_theoretical)
    """
    
    def __init__(self, physics_weight: float = 0.1, pin_value: float = 100.0):
        super().__init__()
        self.physics_weight = physics_weight
        self.pin_value = pin_value
        self.mse = nn.MSELoss()
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics-informed loss.
        
        Args:
            y_pred: Predicted values [batch_size, 4] (PCE, Voc, Jsc, FF)
            y_true: True values [batch_size, 4]
            
        Returns:
            Total loss and loss components
        """
        # Extract predictions
        pce_pred = y_pred[:, 0]
        voc_pred = y_pred[:, 1]
        jsc_pred = y_pred[:, 2]
        ff_pred = y_pred[:, 3]
        
        # Compute theoretical PCE from physics
        pce_theoretical = (voc_pred * jsc_pred * ff_pred) / self.pin_value * 100
        
        # MSE for each target
        mse_pce = self.mse(pce_pred, y_true[:, 0])
        mse_voc = self.mse(voc_pred, y_true[:, 1])
        mse_jsc = self.mse(jsc_pred, y_true[:, 2])
        mse_ff = self.mse(ff_pred, y_true[:, 3])
        
        # Physics constraint violation
        physics_violation = self.mse(pce_pred, pce_theoretical)
        
        # Total loss
        total_loss = (
            mse_pce + mse_voc + mse_jsc + mse_ff +
            self.physics_weight * physics_violation
        )
        
        return total_loss, {
            'mse_pce': mse_pce,
            'mse_voc': mse_voc,
            'mse_jsc': mse_jsc,
            'mse_ff': mse_ff,
            'physics_violation': physics_violation
        }


class MultiTaskNeuralNetwork(nn.Module):
    """
    Multi-task Neural Network for multi-output prediction.
    
    Architecture:
    - Shared feature extraction layers
    - Task-specific output heads
    
    Benefits:
    - Learns shared representations
    - Captures correlations between outputs
    """
    
    def __init__(
        self, 
        input_size: int, 
        hidden_sizes: List[int] = [128, 64, 32],
        output_size: int = 4
    ):
        super().__init__()
        
        # Shared layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific heads
        self.pce_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.voc_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.jsc_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.ff_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        shared_features = self.shared_layers(x)
        
        # Get predictions from each head
        pce = self.pce_head(shared_features)
        voc = self.voc_head(shared_features)
        jsc = self.jsc_head(shared_features)
        ff = self.ff_head(shared_features)
        
        # Concatenate outputs
        return torch.cat([pce, voc, jsc, ff], dim=1)


class MultiOutputPredictor:
    """
    Complete multi-output prediction system.
    
    Implements multiple approaches:
    1. Independent models (baseline)
    2. MultiOutputRegressor
    3. Multi-task Neural Network
    4. Physics-informed Neural Network
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.scaler_X = StandardScaler()
        self.scalers_y = {}
        self.models = {}
        self.results = {}
        
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Extract features
        X = df[feature_cols].values
        y = df[self.config.targets].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Scale features
        X_train = self.scaler_X.fit_transform(X_train)
        X_test = self.scaler_X.transform(X_test)
        
        # Scale targets separately
        for i, target in enumerate(self.config.targets):
            scaler = StandardScaler()
            y_train[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).ravel()
            y_test[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).ravel()
            self.scalers_y[target] = scaler
        
        return X_train, X_test, y_train, y_test
    
    def train_independent_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Dict[str, object]:
        """
        Train independent models for each target (baseline).
        
        Returns:
            Dictionary of trained models
        """
        print("\n" + "=" * 60)
        print("Training Independent Models (Baseline)")
        print("=" * 60)
        
        models = {}
        
        for i, target in enumerate(self.config.targets):
            print(f"\nTraining model for {target}...")
            
            model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train[:, i])
            models[target] = model
            
            print(f"  ✓ {target} model trained")
        
        self.models['independent'] = models
        return models
    
    def train_multioutput_regressor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> MultiOutputRegressor:
        """
        Train MultiOutputRegressor.
        
        Returns:
            Trained MultiOutputRegressor
        """
        print("\n" + "=" * 60)
        print("Training MultiOutputRegressor")
        print("=" * 60)
        
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                random_state=self.config.random_state,
                n_jobs=-1
            )
        )
        
        model.fit(X_train, y_train)
        self.models['multioutput'] = model
        
        print("  ✓ MultiOutputRegressor trained")
        return model
    
    def train_multitask_nn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        y_test: np.ndarray = None,
        use_physics_loss: bool = False
    ) -> MultiTaskNeuralNetwork:
        """
        Train multi-task neural network.
        
        Args:
            X_train, y_train: Training data
            X_test, y_test: Validation data
            use_physics_loss: Whether to use physics-informed loss
            
        Returns:
            Trained neural network
        """
        model_name = 'physics_informed_nn' if use_physics_loss else 'multitask_nn'
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.replace('_', ' ').title()}")
        print("=" * 60)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Initialize model
        model = MultiTaskNeuralNetwork(
            input_size=X_train.shape[1],
            hidden_sizes=self.config.hidden_sizes,
            output_size=len(self.config.targets)
        )
        
        # Loss and optimizer
        if use_physics_loss:
            criterion = PhysicsInformedLoss(
                physics_weight=self.config.physics_weight,
                pin_value=self.config.pin_value
            )
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                outputs = model(batch_X)
                
                if use_physics_loss:
                    loss, _ = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.4f}")
        
        self.models[model_name] = model
        print(f"  ✓ {model_name.replace('_', ' ').title()} trained")
        
        return model
    
    def evaluate_models(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Dict]:
        """
        Evaluate all trained models.
        
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "=" * 60)
        print("Model Evaluation")
        print("=" * 60)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.replace('_', ' ').title()}:")
            
            # Get predictions
            if 'nn' in model_name:
                model.eval()
                with torch.no_grad():
                    X_test_t = torch.FloatTensor(X_test)
                    y_pred = model(X_test_t).numpy()
            elif model_name == 'independent':
                y_pred = np.column_stack([
                    model[target].predict(X_test) 
                    for target in self.config.targets
                ])
            else:
                y_pred = model.predict(X_test)
            
            # Inverse transform predictions
            y_pred_original = np.zeros_like(y_pred)
            y_test_original = np.zeros_like(y_test)
            
            for i, target in enumerate(self.config.targets):
                y_pred_original[:, i] = self.scalers_y[target].inverse_transform(
                    y_pred[:, i].reshape(-1, 1)
                ).ravel()
                y_test_original[:, i] = self.scalers_y[target].inverse_transform(
                    y_test[:, i].reshape(-1, 1)
                ).ravel()
            
            # Compute metrics for each target
            target_metrics = {}
            
            for i, target in enumerate(self.config.targets):
                r2 = r2_score(y_test_original[:, i], y_pred_original[:, i])
                mae = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
                rmse = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred_original[:, i]))
                
                target_metrics[target] = {
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse
                }
                
                print(f"  {target}: R² = {r2:.4f}, MAE = {mae:.4f}, RMSE = {rmse:.4f}")
            
            # Compute average metrics
            avg_r2 = np.mean([m['R²'] for m in target_metrics.values()])
            avg_mae = np.mean([m['MAE'] for m in target_metrics.values()])
            
            results[model_name] = {
                'target_metrics': target_metrics,
                'avg_r2': avg_r2,
                'avg_mae': avg_mae
            }
            
            print(f"  Average: R² = {avg_r2:.4f}, MAE = {avg_mae:.4f}")
        
        self.results = results
        return results
    
    def get_best_model(self) -> Tuple[str, object]:
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model)
        """
        if not self.results:
            raise ValueError("No models evaluated yet")
        
        best_model_name = max(
            self.results.keys(),
            key=lambda x: self.results[x]['avg_r2']
        )
        
        return best_model_name, self.models[best_model_name]
    
    def save_results(self, output_path: str = 'reports/multi_objective/model_comparison.md'):
        """Save model comparison results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comparison table
        comparison = []
        
        for model_name, result in self.results.items():
            row = {
                'Model': model_name.replace('_', ' ').title(),
                'Avg R²': f"{result['avg_r2']:.4f}",
                'Avg MAE': f"{result['avg_mae']:.4f}"
            }
            
            for target in self.config.targets:
                if target in result['target_metrics']:
                    row[f'{target} R²'] = f"{result['target_metrics'][target]['R²']:.4f}"
            
            comparison.append(row)
        
        df_comparison = pd.DataFrame(comparison)
        
        # Generate report
        report = f"""# Multi-Output Model Comparison

**Generated**: 2026-03-12
**Research**: Physics-Informed Multi-Output Learning for Perovskite Solar Cells

---

## 📊 Model Performance Comparison

{df_comparison.to_markdown(index=False)}

---

## 🔬 Key Findings

1. **Best Model**: {self.get_best_model()[0]}
2. **Physics-Informed Learning**: Incorporates physical constraints for better generalization
3. **Multi-Task Learning**: Captures correlations between targets

---

## 📈 Performance Targets

| Target | R² Goal | Achieved |
|--------|---------|----------|
"""
        
        best_result = self.results[self.get_best_model()[0]]
        
        for target in self.config.targets:
            achieved = '✅' if best_result['target_metrics'][target]['R²'] >= 0.75 else '❌'
            report += f"| {target} | ≥ 0.75 | {achieved} {best_result['target_metrics'][target]['R²']:.4f} |\n"
        
        report += """

---

**Next Steps**: Pareto optimization and material recommendation.
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n📝 Saved model comparison to: {output_path}")


def main():
    """Main execution for multi-output modeling."""
    print("=" * 60)
    print("Multi-Output Prediction for Perovskite Solar Cells")
    print("=" * 60)
    
    # Load data
    data_path = Path('/home/yhm/desktop/code/perovskite-eda-research/.worktrees/issue-9/multi_objective/data/multi_target_data.csv')
    
    if not data_path.exists():
        print(f"\n⚠️ Data not found at {data_path}")
        print("Please run data_preparation.py first.")
        return
    
    df = pd.read_csv(data_path)
    print(f"\n📊 Loaded data: {df.shape}")
    
    # Feature columns (use available columns)
    feature_cols = ['Voc', 'Jsc', 'FF']  # Will be modified based on actual data
    
    # Initialize predictor
    config = ModelConfig()
    predictor = MultiOutputPredictor(config)
    
    # Prepare data
    print("\n📊 Preparing data...")
    # Note: This is a simplified version - in practice, you'd have proper feature columns
    # For now, we'll use the target columns themselves as features for demonstration
    
    print("\n✅ Multi-output model framework ready")
    print("Run with actual feature data for full training")


if __name__ == "__main__":
    main()