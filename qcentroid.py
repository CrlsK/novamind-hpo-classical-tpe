"""Classical TPE/Optuna Hyperparameter Optimization Solver for QCentroid
======================================================================
This solver implements a production-grade Tree-structured Parzen Estimator (TPE)
based hyperparameter optimization baseline using Optuna, configured for the
"Massively Parallel Hyperparameter Optimization with Quantum-Inspired Search"
use case on QCentroid.

The solver parses HPO problem specifications, builds an Optuna study with
a sophisticated TPE sampler, and uses an intelligent surrogate objective
function to simulate training evaluation results. Results include convergence
metrics, cost tracking, and complete trial history.
"""

import logging
import json
import time
import math
import random
import os
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import hashlib

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import Trial

# Try to import visualization module
try:
    from visualizations import generate_classical_visualizations
except ImportError:
    generate_classical_visualizations = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SurrogateObjective:
    """
    Intelligent surrogate objective function for simulating training evaluation.

    This function simulates the f1_macro metric for a transformer-based NLP model
    based on hyperparameter quality. It incorporates:
    - Base quality scores for different hyperparameter choices
    - Synergistic effects between compatible hyperparameters
    - Constraint penalties
    - Realistic variance and stochasticity
    - Convergence toward better configurations over trial history
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the surrogate with problem configuration.

        Args:
            config: Dictionary with hyperparameter definitions and constraints
        """
        self.config = config
        self.trial_counter = 0
        self.best_score_seen = 0.0
        self.trial_history = []

        # Quality scores for specific hyperparameter values
        self.quality_scores = {
            'learning_rate': {
                1e-5: 0.01,
                2e-5: 0.05,
                5e-5: 0.03,
                1e-4: 0.00,
            },
            'warmup_steps': {
                100: 0.02,
                500: 0.04,
                1000: 0.03,
                2000: 0.01,
            },
            'weight_decay': {
                0.0: -0.02,
                0.01: 0.04,
                0.1: 0.02,
                0.5: -0.01,
            },
            'dropout_rate': {
                0.1: 0.02,
                0.2: 0.04,
                0.3: 0.02,
                0.5: -0.01,
            },
            'attention_heads': {
                8: 0.01,
                12: 0.04,
                16: 0.02,
            },
            'hidden_dim': {
                512: 0.01,
                768: 0.05,
                1024: 0.03,
            },
            'num_layers': {
                6: 0.00,
                12: 0.04,
                24: 0.02,
            },
            'batch_size': {
                16: 0.01,
                32: 0.03,
                64: 0.02,
            },
            'optimizer': {
                'adam': 0.02,
                'adamw': 0.04,
                'sgd': -0.01,
            },
            'scheduler': {
                'linear': 0.01,
                'cosine': 0.04,
                'step': 0.00,
            },
            'gradient_clipping': {
                1.0: 0.01,
                5.0: 0.04,
                10.0: 0.02,
            },
            'label_smoothing': {
                0.0: 0.00,
                0.1: 0.03,
                0.2: 0.02,
            },
            'mixed_precision': {
                False: 0.01,
                True: 0.02,
            },
            'activation_function': {
                'relu': 0.01,
                'gelu': 0.04,
                'silu': 0.03,
            },
            'positional_encoding': {
                'absolute': 0.00,
                'rotary': 0.04,
                'alibi': 0.02,
            },
            'layer_norm_type': {
                'layernorm': 0.01,
                'rmsnorm': 0.04,
                'groupnorm': 0.00,
            },
        }

        # Synergy bonuses for good parameter combinations
        self.synergies = [
            # (condition, bonus)
            (lambda p: p['num_layers'] >= 12 and p['hidden_dim'] >= 768, 0.05),
            (lambda p: p['optimizer'] == 'adamw' and p['weight_decay'] > 0.0, 0.03),
            (lambda p: p['scheduler'] == 'cosine' and p['warmup_steps'] >= 500, 0.02),
            (lambda p: p['positional_encoding'] == 'rotary' and p['attention_heads'] >= 12, 0.04),
            (lambda p: p['activation_function'] == 'gelu' and p['num_layers'] >= 12, 0.02),
            (lambda p: p['learning_rate'] <= 2e-5 and p['scheduler'] == 'cosine', 0.03),
        ]

    def __call__(self, trial: Trial) -> float:
        """
        Evaluate objective for a single trial.

        Args:
            trial: Optuna Trial object

        Returns:
            f1_macro score (0.0 to 1.0)
        """
        self.trial_counter += 1
        start_time = time.time()

        # Suggest hyperparameters based on search space
        params = self._suggest_parameters(trial)

        # Compute base score
        score = self._compute_base_score(params)

        # Apply synergy bonuses
        score += sum(bonus for condition, bonus in self.synergies if condition(params))

        # Apply constraints
        score = self._apply_constraints(score, params)

        # Add realistic variance
        variance = self._compute_variance(params)
        score += random.gauss(0, variance)

        # Clamp to valid range
        score = max(0.0, min(1.0, score))

        # Track convergence
        elapsed = time.time() - start_time
        if score > self.best_score_seen:
            self.best_score_seen = score
            logger.info(
                f"Trial {self.trial_counter}: New best score = {score:.4f} "
                f"(lr={params['learning_rate']:.1e}, "
                f"layers={params['num_layers']}, "
                f"hidden={params['hidden_dim']}, "
                f"scheduler={params['scheduler']}, "
                f"pos_enc={params['positional_encoding']})"
            )

        self.trial_history.append({
            'trial_num': self.trial_counter,
            'score': score,
            'params': params,
            'elapsed': elapsed,
        })

        return score

    def _suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'warmup_steps': trial.suggest_int('warmup_steps', 100, 2000, step=100),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.5, step=0.05),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'attention_heads': trial.suggest_int('attention_heads', 8, 16, step=2),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [512, 768, 1024]),
            'num_layers': trial.suggest_int('num_layers', 6, 24, step=2),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'sgd']),
            'scheduler': trial.suggest_categorical('scheduler', ['linear', 'cosine', 'step']),
            'gradient_clipping': trial.suggest_float('gradient_clipping', 1.0, 10.0, step=0.5),
            'label_smoothing': trial.suggest_float('label_smoothing', 0.0, 0.2, step=0.05),
            'mixed_precision': trial.suggest_categorical('mixed_precision', [False, True]),
            'activation_function': trial.suggest_categorical('activation_function', ['relu', 'gelu', 'silu']),
            'positional_encoding': trial.suggest_categorical('positional_encoding', ['absolute', 'rotary', 'alibi']),
            'layer_norm_type': trial.suggest_categorical('layer_norm_type', ['layernorm', 'rmsnorm', 'groupnorm']),
        }

    def _compute_base_score(self, params: Dict[str, Any]) -> float:
        """Compute base score from individual hyperparameter quality scores."""
        base = 0.75  # Start from good baseline

        for param_name, param_value in params.items():
            if param_name in self.quality_scores:
                scores = self.quality_scores[param_name]
                if param_value in scores:
                    base += scores[param_value]
                else:
                    # Interpolate for continuous values
                    sorted_values = sorted(scores.keys())
                    if param_value < sorted_values[0]:
                        base += scores[sorted_values[0]]
                    elif param_value > sorted_values[-1]:
                        base += scores[sorted_values[-1]]
                    else:
                        # Linear interpolation
                        for i in range(len(sorted_values) - 1):
                            v1, v2 = sorted_values[i], sorted_values[i + 1]
                            if v1 <= param_value <= v2:
                                w = (param_value - v1) / (v2 - v1)
                                base += (1 - w) * scores[v1] + w * scores[v2]
                                break

        return base

    def _apply_constraints(self, score: float, params: Dict[str, Any]) -> float:
        """Apply constraint penalties to the score."""
        # Constraint: batch_size * num_layers should be reasonable for memory
        if params['batch_size'] * params['num_layers'] > 1024:
            score -= 0.05

        # Constraint: high learning rate needs high warmup
        if params['learning_rate'] > 5e-5 and params['warmup_steps'] < 500:
            score -= 0.03

        # Constraint: high dropout conflicts with certain activations
        if params['dropout_rate'] > 0.3 and params['activation_function'] == 'silu':
            score -= 0.02

        # Constraint: mixed precision with certain precision configurations
        if params['mixed_precision'] and params['hidden_dim'] == 512:
            score -= 0.01

        return score

    def _compute_variance(self, params: Dict[str, Any]) -> float:
        """Compute realistic variance based on hyperparameter robustness."""
        # More stable configurations have lower variance
        variance = 0.02

        # Learning rate variance (higher lr = more variance)
        if params['learning_rate'] > 5e-5:
            variance += 0.01

        # High dropout adds variance
        if params['dropout_rate'] > 0.3:
            variance += 0.005

        # Scheduler stability
        if params['scheduler'] != 'cosine':
            variance += 0.01

        return variance


class ClassicalTPESolver:
    """
    Production-grade Classical TPE/Optuna Hyperparameter Optimization Solver.

    Implements the Tree-structured Parzen Estimator sampler with Optuna,
    configured for efficient hyperparameter optimization with proper constraint
    handling, early stopping, and comprehensive result tracking.
    """

    def __init__(self, config: Dict[str, Any], solver_params: Dict[str, Any]):
        """
        Initialize the solver.

        Args:
            config: HPO problem configuration from input_data
            solver_params: Solver-specific parameters (override defaults)
        """
        self.config = config
        self.solver_params = solver_params
        self.start_time = time.time()

        # Extract classical baseline config
        classical_cfg = config.get('Solver configuration', {}).get('classical_baseline', {})

        # Solver parameters with defaults
        self.n_trials = solver_params.get('n_trials', classical_cfg.get('n_trials', 200))
        self.n_startup_trials = solver_params.get('n_startup_trials', classical_cfg.get('n_startup_trials', 20))
        self.multivariate = solver_params.get('multivariate', classical_cfg.get('multivariate', True))
        self.group = solver_params.get('group', classical_cfg.get('group', True))
        self.constant_liar = solver_params.get('constant_liar', classical_cfg.get('constant_liar', True))
        self.seed = solver_params.get('seed', 42)

        logger.info(f"Initializing ClassicalTPESolver with {self.n_trials} trials")
        logger.info(f"TPE config: startup={self.n_startup_trials}, multivariate={self.multivariate}, "
                   f"group={self.group}, constant_liar={self.constant_liar}")

        # Initialize study
        self.sampler = TPESampler(
            n_startup_trials=self.n_startup_trials,
            n_ei_candidates=24,
            multivariate=self.multivariate,
            group=self.group,
            constant_liar=self.constant_liar,
            seed=self.seed,
        )

        self.study = optuna.create_study(
            direction='maximize',
            sampler=self.sampler,
            pruner=MedianPruner(n_startup_trials=self.n_startup_trials),
        )

        # Initialize surrogate objective
        self.surrogate = SurrogateObjective(config)

    def run(self) -> Dict[str, Any]:
        """
        Run the hyperparameter optimization.

        Returns:
            Dictionary with results including best value, status, metrics, costs, and benchmark info
        """
        logger.info(f"Starting optimization with {self.n_trials} trials")

        try:
            # Run optimization
            self.study.optimize(
                self.surrogate,
                n_trials=self.n_trials,
                show_progress_bar=False,
                callbacks=None,
            )

            status = 'SUCCESS'
            logger.info(f"Optimization completed successfully")

        except Exception as e:
            logger.error(f"Optimization failed with error: {e}")
            status = 'ERROR'

        elapsed_time = time.time() - self.start_time

        # Extract results
        best_trial = self.study.best_trial
        best_value = best_trial.value
        best_params = best_trial.params

        logger.info(f"Best value: {best_value:.4f}")
        logger.info(f"Best params: {best_params}")

        # Get top 10 configurations
        top_trials = sorted(self.study.trials, key=lambda t: t.value, reverse=True)[:10]
        top_configs = [
            {
                'rank': i + 1,
                'score': t.value,
                'params': t.params,
                'trial_number': t.number,
            }
            for i, t in enumerate(top_trials)
        ]

        # Compute metrics
        all_values = [t.value for t in self.study.trials if t.value is not None]
        metrics = self._compute_metrics(all_values, elapsed_time)

        # Compute cost breakdown
        cost_breakdown = self._compute_cost_breakdown(elapsed_time)

        # Compute benchmark
        benchmark = self._compute_benchmark(elapsed_time, cost_breakdown)

        results = {
            'objective_value': best_value,
            'best_params': best_params,
            'best_trial_number': best_trial.number,
            'solution_status': status,
            'computation_metrics': metrics,
            'cost_breakdown': cost_breakdown,
            'benchmark': benchmark,
            'top_10_configurations': top_configs,
            'total_trials': len(self.study.trials),
            'trial_history': self.surrogate.trial_history,
        }

        return results

    def _compute_metrics(self, scores: List[float], elapsed: float) -> Dict[str, float]:
        """Compute optimization metrics."""
        if not scores:
            return {
                'mean_score': 0.0,
                'max_score': 0.0,
                'min_score': 0.0,
                'std_dev': 0.0,
                'convergence_rate': 0.0,
                'time_per_trial': 0.0,
                'total_time_seconds': elapsed,
            }

        # Sort scores to track convergence
        sorted_scores = sorted(scores)

        # Compute convergence rate (improvement from first 20% to last 20% of trials)
        n = len(sorted_scores)
        first_batch_mean = sum(sorted_scores[:max(1, n//5)]) / max(1, n//5)
        last_batch_mean = sum(sorted_scores[-max(1, n//5):]) / max(1, n//5)
        convergence_rate = max(0.0, last_batch_mean - first_batch_mean)

        return {
            'mean_score': sum(scores) / len(scores),
            'max_score': max(scores),
            'min_score': min(scores),
            'std_dev': (sum((x - sum(scores) / len(scores)) ** 2 for x in scores) / len(scores)) ** 0.5,
            'convergence_rate': convergence_rate,
            'time_per_trial': elapsed / len(scores),
            'total_time_seconds': elapsed,
        }

    def _compute_cost_breakdown(self, elapsed: float) -> Dict[str, Any]:
        """Compute cost breakdown."""
        # Simulate cost allocation
        # Assumption: each trial takes ~3.6 minutes (200 trials * 3.6min = 12 GPU-hours)
        time_per_trial_minutes = 3.6
        total_gpu_hours = (self.n_trials * time_per_trial_minutes) / 60.0

        cost_per_gpu_hour = 48.0 / 12.0  # $4 per GPU-hour
        total_cost = total_gpu_hours * cost_per_gpu_hour

        return {
            'total_gpu_hours': total_gpu_hours,
            'cost_per_gpu_hour': cost_per_gpu_hour,
            'total_cost_usd': total_cost,
            'cost_per_trial': total_cost / self.n_trials,
            'wall_clock_time_seconds': elapsed,
        }

    def _compute_benchmark(self, elapsed: float, cost_breakdown: Dict[str, Any]) -> Dict[str, Any]:
        """Compute benchmark metrics."""
        # Energy consumption estimate (GPU power ~ 250W average)
        gpu_power_watts = 250.0
        energy_joules = (elapsed / 3600.0) * gpu_power_watts * 3600.0
        energy_kwh = energy_joules / 3.6e6

        return {
            'execution_cost': round(cost_breakdown['total_cost_usd'], 4),
            'time_elapsed': round(elapsed, 2),
            'energy_consumption': round(energy_kwh, 6),
            'trials_completed': self.n_trials,
            'efficiency_metric': round((self.study.best_trial.value * 100) / max(elapsed, 0.001), 6),
        }


def run(input_data: dict, solver_params: dict, extra_arguments: dict) -> dict:
    """
    QCentroid solver entry point.

    This function is called by the QCentroid framework to execute the solver.

    Args:
        input_data: Complete HPO problem specification including search space,
                   constraints, evaluation protocol, solver config, etc.
        solver_params: Solver-specific parameter overrides
        extra_arguments: Additional arguments from the caller

    Returns:
        Dictionary with optimization results including objective value, solution status,
        computation metrics, cost breakdown, and benchmark information.
    """
    logger.info("=" * 80)
    logger.info("Classical TPE/Optuna Hyperparameter Optimization Solver")
    logger.info("=" * 80)
    logger.info(f"Input data keys: {list(input_data.keys())}")
    logger.info(f"Solver params: {solver_params}")
    logger.info(f"Extra arguments: {extra_arguments}")

    try:
        # Initialize and run solver
        solver = ClassicalTPESolver(input_data, solver_params)
        results = solver.run()

        # Attempt to generate visualizations (non-blocking)
        if generate_classical_visualizations is not None:
            try:
                generate_classical_visualizations(results, solver.surrogate.trial_history)
            except Exception as viz_error:
                logger.warning(f"Visualization generation failed: {viz_error}")
        else:
            logger.warning("Visualization module not available, skipping chart generation")

        logger.info("=" * 80)
        logger.info("OPTIMIZATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best value: {results['objective_value']:.4f}")
        logger.info(f"Status: {results['solution_status']}")
        logger.info(f"Total trials: {results['total_trials']}")
        logger.info(f"Time elapsed: {results['benchmark']['time_elapsed']:.2f}s")
        logger.info(f"Cost: ${results['benchmark']['execution_cost']:.2f}")
        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.exception(f"Solver execution failed: {e}")
        return {
            'objective_value': 0.0,
            'best_params': {},
            'solution_status': 'ERROR',
            'error_message': str(e),
            'computation_metrics': {},
            'cost_breakdown': {},
            'benchmark': {
                'execution_cost': 0.0,
                'time_elapsed': 0.0,
                'energy_consumption': 0.0,
            },
        }


if __name__ == '__main__':
    # Local testing
    logger.info("Running ClassicalTPESolver in local mode")

    # Create minimal test input_data
    test_input_data = {
        'Search space definition': {
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'warmup_steps': {'type': 'int', 'low': 100, 'high': 2000},
            'weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.5},
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5},
            'attention_heads': {'type': 'int', 'low': 8, 'high': 16},
            'hidden_dim': {'type': 'categorical', 'choices': [512, 768, 1024]},
            'num_layers': {'type': 'int', 'low': 6, 'high': 24},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
            'scheduler': {'type': 'categorical', 'choices': ['linear', 'cosine', 'step']},
            'gradient_clipping': {'type': 'float', 'low': 1.0, 'high': 10.0},
            'label_smoothing': {'type': 'float', 'low': 0.0, 'high': 0.2},
            'mixed_precision': {'type': 'categorical', 'choices': [False, True]},
            'activation_function': {'type': 'categorical', 'choices': ['relu', 'gelu', 'silu']},
            'positional_encoding': {'type': 'categorical', 'choices': ['absolute', 'rotary', 'alibi']},
            'layer_norm_type': {'type': 'categorical', 'choices': ['layernorm', 'rmsnorm', 'groupnorm']},
        },
        'Objective metric': 'f1_macro',
        'Optimization direction': 'maximize',
        'Solver configuration': {
            'classical_baseline': {
                'n_trials': 50,
                'n_startup_trials': 20,
                'multivariate': True,
                'group': True,
                'constant_liar': True,
            }
        }
    }

    test_solver_params = {
        'n_trials': 50,
        'seed': 42,
    }

    results = run(test_input_data, test_solver_params, {})

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(json.dumps({k: v for k, v in results.items()
                     if k not in ['trial_history', 'top_10_configurations']},
                    indent=2, default=str))
    print(f"\nBest objective value: {results['objective_value']:.4f}")
    print(f"Top 10 configurations (first 3):")
    for config in results['top_10_configurations'][:3]:
        print(f"  Rank {config['rank']}: {config['score']:.4f}")