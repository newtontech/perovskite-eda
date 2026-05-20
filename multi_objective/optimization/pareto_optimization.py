"""
Pareto Optimization for Perovskite Solar Cell Material Design
Research: Physics-Informed Multi-Output Learning

This module implements multi-objective optimization using NSGA-II
for discovering optimal material combinations that balance PCE, Voc, Jsc, and FF.

Author: OpenClaw AI Assistant
Date: 2026-03-12
Target: Nature Machine Intelligence
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# NSGA-II implementation
try:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    PYMOO_AVAILABLE = True
except ImportError:
    PYMOO_AVAILABLE = False
    print("Warning: pymoo not available. Install with: pip install pymoo")


@dataclass
class OptimizationConfig:
    """Configuration for Pareto optimization."""
    
    # Objectives
    objectives: List[str] = None
    
    # Optimization bounds (normalized)
    bounds: Dict[str, Tuple[float, float]] = None
    
    # NSGA-II parameters
    population_size: int = 100
    n_generations: int = 200
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    
    # Decision variables (material features)
    n_variables: int = 10
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ['PCE', 'Voc', 'Jsc', 'FF']
        
        if self.bounds is None:
            self.bounds = {
                'PCE': (0, 30),      # %
                'Voc': (0.5, 1.5),   # V
                'Jsc': (5, 30),      # mA/cm²
                'FF': (0.3, 0.9)     # dimensionless
            }


class PerovskiteOptimizationProblem(Problem):
    """
    Multi-objective optimization problem for perovskite solar cells.
    
    Objectives:
    - Maximize PCE (Power Conversion Efficiency)
    - Maximize Voc (Open Circuit Voltage)
    - Maximize Jsc (Short Circuit Current Density)
    - Maximize FF (Fill Factor)
    
    Note: pymoo minimizes objectives, so we negate them.
    """
    
    def __init__(
        self,
        prediction_model: Callable,
        n_variables: int = 10,
        bounds: Dict[str, Tuple[float, float]] = None
    ):
        """
        Initialize optimization problem.
        
        Args:
            prediction_model: Model that predicts targets from features
            n_variables: Number of decision variables (features)
            bounds: Bounds for each objective
        """
        self.prediction_model = prediction_model
        self.bounds = bounds or OptimizationConfig().bounds
        
        # Define problem bounds
        # Assuming normalized features [0, 1]
        xl = np.zeros(n_variables)
        xu = np.ones(n_variables)
        
        super().__init__(
            n_var=n_variables,
            n_obj=4,  # PCE, Voc, Jsc, FF
            n_constr=0,
            xl=xl,
            xu=xu
        )
    
    def _evaluate(self, X: np.ndarray, out: Dict, *args, **kwargs):
        """
        Evaluate objectives for population.
        
        Args:
            X: Population [n_individuals, n_variables]
            out: Output dictionary
        """
        n_individuals = X.shape[0]
        
        # Predict performance metrics
        predictions = self.prediction_model(X)
        
        # Extract predictions (assuming order: PCE, Voc, Jsc, FF)
        pce = predictions[:, 0]
        voc = predictions[:, 1]
        jsc = predictions[:, 2]
        ff = predictions[:, 3]
        
        # Negate for minimization (we want to maximize)
        out["F"] = np.column_stack([
            -pce,  # Maximize PCE
            -voc,  # Maximize Voc
            -jsc,  # Maximize Jsc
            -ff    # Maximize FF
        ])


class ParetoOptimizer:
    """
    Complete Pareto optimization system for perovskite solar cells.
    
    Features:
    - NSGA-II algorithm for multi-objective optimization
    - Pareto front identification
    - Knee point detection
    - Decision support for material selection
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.pareto_front = None
        self.pareto_set = None
        self.knee_point = None
        
    def run_optimization(
        self,
        prediction_model: Callable,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run NSGA-II optimization.
        
        Args:
            prediction_model: Model that predicts targets from features
            verbose: Whether to print progress
            
        Returns:
            Tuple of (pareto_front, pareto_set)
        """
        if not PYMOO_AVAILABLE:
            raise ImportError("pymoo is required. Install with: pip install pymoo")
        
        print("\n" + "=" * 60)
        print("NSGA-II Pareto Optimization")
        print("=" * 60)
        
        # Define problem
        problem = PerovskiteOptimizationProblem(
            prediction_model=prediction_model,
            n_variables=self.config.n_variables,
            bounds=self.config.bounds
        )
        
        # Configure algorithm
        algorithm = NSGA2(
            pop_size=self.config.population_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=self.config.crossover_prob, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True
        )
        
        # Set termination
        termination = get_termination(
            "n_gen",
            self.config.n_generations
        )
        
        # Run optimization
        if verbose:
            print(f"\nPopulation size: {self.config.population_size}")
            print(f"Generations: {self.config.n_generations}")
            print("\nRunning optimization...")
        
        res = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=verbose
        )
        
        # Extract Pareto front (negate back to positive values)
        self.pareto_front = -res.F
        self.pareto_set = res.X
        
        print(f"\n✓ Found {len(self.pareto_front)} Pareto-optimal solutions")
        
        return self.pareto_front, self.pareto_set
    
    def find_knee_point(
        self,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Find knee point on Pareto front.
        
        The knee point represents the best compromise solution.
        
        Args:
            weights: Weights for each objective (default: equal weights)
            
        Returns:
            Tuple of (knee_point_values, knee_point_index)
        """
        if self.pareto_front is None:
            raise ValueError("Run optimization first")
        
        # Default weights
        if weights is None:
            weights = {obj: 1.0 for obj in self.config.objectives}
        
        # Normalize objectives to [0, 1]
        pareto_normalized = np.zeros_like(self.pareto_front)
        
        for i, obj in enumerate(self.config.objectives):
            min_val = self.pareto_front[:, i].min()
            max_val = self.pareto_front[:, i].max()
            
            if max_val > min_val:
                pareto_normalized[:, i] = (self.pareto_front[:, i] - min_val) / (max_val - min_val)
            else:
                pareto_normalized[:, i] = 1.0
        
        # Compute weighted sum
        weight_vector = np.array([weights[obj] for obj in self.config.objectives])
        weighted_sums = pareto_normalized @ weight_vector
        
        # Knee point: maximum weighted sum
        knee_idx = np.argmax(weighted_sums)
        self.knee_point = self.pareto_front[knee_idx]
        
        print(f"\n🎯 Knee Point Found:")
        for i, obj in enumerate(self.config.objectives):
            print(f"   {obj}: {self.knee_point[i]:.4f}")
        
        return self.knee_point, knee_idx
    
    def compute_hypervolume(
        self,
        reference_point: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute hypervolume indicator.
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if self.pareto_front is None:
            raise ValueError("Run optimization first")
        
        if reference_point is None:
            # Use bounds as reference
            reference_point = np.array([
                self.config.bounds[obj][1] 
                for obj in self.config.objectives
            ])
        
        # Compute hypervolume
        try:
            from pymoo.indicators.hv import HV
            hv_indicator = HV(ref_point=reference_point)
            hypervolume = hv_indicator(self.pareto_front)
        except:
            # Fallback: simple volume calculation
            hypervolume = np.prod(
                reference_point - self.pareto_front.min(axis=0)
            )
        
        return hypervolume
    
    def visualize_pareto_front(
        self,
        output_path: Optional[str] = None,
        show_knee: bool = True
    ):
        """
        Visualize Pareto front in 2D and 3D.
        
        Args:
            output_path: Path to save figure
            show_knee: Whether to highlight knee point
        """
        if self.pareto_front is None:
            raise ValueError("Run optimization first")
        
        fig = plt.figure(figsize=(16, 12))
        
        # 2D projections
        pairs = [
            ('PCE', 'Voc', 0, 1),
            ('PCE', 'Jsc', 0, 2),
            ('PCE', 'FF', 0, 3),
            ('Voc', 'Jsc', 1, 2)
        ]
        
        for idx, (x_name, y_name, x_idx, y_idx) in enumerate(pairs):
            ax = fig.add_subplot(2, 2, idx + 1)
            
            # Plot Pareto front
            ax.scatter(
                self.pareto_front[:, x_idx],
                self.pareto_front[:, y_idx],
                c='blue',
                alpha=0.6,
                s=30,
                label='Pareto Front'
            )
            
            # Highlight knee point
            if show_knee and self.knee_point is not None:
                ax.scatter(
                    self.knee_point[x_idx],
                    self.knee_point[y_idx],
                    c='red',
                    s=100,
                    marker='*',
                    label='Knee Point',
                    zorder=5
                )
            
            ax.set_xlabel(x_name, fontsize=11)
            ax.set_ylabel(y_name, fontsize=11)
            ax.set_title(f'{x_name} vs {y_name} Trade-off', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Saved Pareto front visualization to: {output_path}")
        
        plt.close()
    
    def visualize_3d_pareto(
        self,
        output_path: Optional[str] = None
    ):
        """
        Create 3D visualization of Pareto front.
        
        Args:
            output_path: Path to save figure
        """
        if self.pareto_front is None:
            raise ValueError("Run optimization first")
        
        fig = plt.figure(figsize=(14, 6))
        
        # 3D plot: PCE vs Voc vs Jsc
        ax1 = fig.add_subplot(121, projection='3d')
        
        scatter = ax1.scatter(
            self.pareto_front[:, 0],  # PCE
            self.pareto_front[:, 1],  # Voc
            self.pareto_front[:, 2],  # Jsc
            c=self.pareto_front[:, 3],  # FF (color)
            cmap='viridis',
            alpha=0.6,
            s=30
        )
        
        ax1.set_xlabel('PCE (%)', fontsize=10)
        ax1.set_ylabel('Voc (V)', fontsize=10)
        ax1.set_zlabel('Jsc (mA/cm²)', fontsize=10)
        ax1.set_title('Pareto Front: PCE-Voc-Jsc', fontsize=11, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax1, pad=0.1)
        cbar.set_label('FF', fontsize=10)
        
        # 3D plot: Voc vs Jsc vs FF
        ax2 = fig.add_subplot(122, projection='3d')
        
        scatter2 = ax2.scatter(
            self.pareto_front[:, 1],  # Voc
            self.pareto_front[:, 2],  # Jsc
            self.pareto_front[:, 3],  # FF
            c=self.pareto_front[:, 0],  # PCE (color)
            cmap='plasma',
            alpha=0.6,
            s=30
        )
        
        ax2.set_xlabel('Voc (V)', fontsize=10)
        ax2.set_ylabel('Jsc (mA/cm²)', fontsize=10)
        ax2.set_zlabel('FF', fontsize=10)
        ax2.set_title('Pareto Front: Voc-Jsc-FF', fontsize=11, fontweight='bold')
        
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1)
        cbar2.set_label('PCE (%)', fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Saved 3D Pareto visualization to: {output_path}")
        
        plt.close()
    
    def generate_report(
        self,
        output_path: str = 'reports/multi_objective/pareto_analysis.md'
    ):
        """Generate comprehensive Pareto analysis report."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = f"""# Pareto Optimization Analysis

**Generated**: 2026-03-12
**Research**: Physics-Informed Multi-Output Learning for Perovskite Solar Cells

---

## 🎯 Multi-Objective Optimization

### Objectives

1. **Maximize PCE** (Power Conversion Efficiency)
2. **Maximize Voc** (Open Circuit Voltage)
3. **Maximize Jsc** (Short Circuit Current Density)
4. **Maximize FF** (Fill Factor)

### Algorithm

- **Method**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- **Population Size**: {self.config.population_size}
- **Generations**: {self.config.n_generations}

---

## 📊 Results

### Pareto Front Statistics

| Metric | Value |
|--------|-------|
| Number of Pareto-optimal solutions | {len(self.pareto_front)} |
| Hypervolume | {self.compute_hypervolume():.4f} |

### Objective Ranges on Pareto Front

"""
        
        for i, obj in enumerate(self.config.objectives):
            min_val = self.pareto_front[:, i].min()
            max_val = self.pareto_front[:, i].max()
            mean_val = self.pareto_front[:, i].mean()
            
            report += f"**{obj}**: {min_val:.2f} - {max_val:.2f} (mean: {mean_val:.2f})\n\n"
        
        if self.knee_point is not None:
            report += """---

## 🎯 Knee Point (Best Compromise)

The knee point represents the optimal trade-off solution.

| Objective | Value |
|-----------|-------|
"""
            
            for i, obj in enumerate(self.config.objectives):
                report += f"| {obj} | {self.knee_point[i]:.4f} |\n"
        
        report += """

---

## 📈 Trade-off Analysis

### Key Trade-offs

1. **Voc-Jsc Trade-off**: Higher bandgap increases Voc but decreases Jsc
2. **FF-PCE Relationship**: FF directly impacts achievable PCE
3. **Multi-objective Balance**: Knee point provides balanced performance

### Design Implications

- **High PCE focus**: Optimize for high Voc and FF
- **High Jsc focus**: Use narrower bandgap materials
- **Balanced approach**: Knee point offers best overall performance

---

## 📊 Figures

- `pareto_front_2d.png` - 2D Pareto front projections
- `pareto_front_3d.png` - 3D Pareto front visualization

---

**Next Steps**: Material recommendation based on Pareto-optimal solutions.
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        print(f"\n📝 Saved Pareto analysis report to: {output_path}")


def main():
    """Main execution for Pareto optimization."""
    print("=" * 60)
    print("Pareto Optimization Framework")
    print("=" * 60)
    
    # Initialize optimizer
    config = OptimizationConfig()
    optimizer = ParetoOptimizer(config)
    
    # Mock prediction model for demonstration
    def mock_prediction_model(X: np.ndarray) -> np.ndarray:
        """Mock model for testing."""
        # Simple linear relationship for demonstration
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, 4))
        
        # Generate synthetic predictions
        predictions[:, 0] = 10 + 15 * X[:, 0] + np.random.randn(n_samples) * 2  # PCE
        predictions[:, 1] = 0.8 + 0.4 * X[:, 1] + np.random.randn(n_samples) * 0.1  # Voc
        predictions[:, 2] = 15 + 10 * X[:, 2] + np.random.randn(n_samples) * 2  # Jsc
        predictions[:, 3] = 0.5 + 0.3 * X[:, 3] + np.random.randn(n_samples) * 0.05  # FF
        
        return predictions
    
    print("\n✅ Pareto optimization framework ready")
    print("Run with actual prediction model for full optimization")


if __name__ == "__main__":
    main()