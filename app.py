"""
QCentroid Local Test Runner for Classical TPE/Optuna Solver
===========================================================

This application provides a standard test environment for running the
ClassicalTPE solver locally before deployment to the QCentroid platform.

Usage:
    python app.py [options]

Options:
    --n-trials N          Number of trials (default: 50 for testing, 200 for production)
    --n-startup N         Number of startup trials (default: 20)
    --seed SEED           Random seed (default: 42)
    --mode {test,full}    'test' for quick validation, 'full' for production run
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from qcentroid import run

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_input_data(n_trials: int = 50) -> dict:
    """
    Create a complete HPO problem specification for testing.

    Args:
        n_trials: Number of trials to run

    Returns:
        Input data dictionary matching QCentroid specification
    """
    return {
        'problem_name': 'Transformer Hyperparameter Optimization',
        'problem_description': 'NKP transformer model HPO with 16 hyperparameters',
        'dataset_reference': 'glue-mrpc',
        'Search space definition': {
            'learning_rate': {'type': 'float', 'low': 1e-5, 'high': 1e-3, 'log': True},
            'warmup_steps': {'type': 'int', 'low': 100, 'high': 2000, 'step': 100},
            'weight_decay': {'type': 'float', 'low': 0.0, 'high': 0.5, 'step': 0.05},
            'dropout_rate': {'type': 'float', 'low': 0.1, 'high': 0.5, 'step': 0.1},
            'attention_heads': {'type': 'int', 'low': 8, 'high': 16, 'step': 2},
            'hidden_dim': {'type': 'categorical', 'choices': [512, 768, 1024]},
            'num_layers': {'type': 'int', 'low': 6, 'high': 24, 'step': 2},
            'batch_size': {'type': 'categorical', 'choices': [16, 32, 64]},
            'optimizer': {'type': 'categorical', 'choices': ['adam', 'adamw', 'sgd']},
            'scheduler': {'type': 'categorical', 'choices': ['linear', 'cosine', 'step']},
            'gradient_clipping': {'type': 'float', 'low': 1.0, 'high': 10.0, 'step': 0.5},
            'label_smoothing': {'type': 'float', 'low': 0.0, 'high': 0.2, 'step': 0.05},
            'mixed_precision': {'type': 'categorical', 'choices': [False, True]},
            'activation_function': {'type': 'categorical', 'choices': ['relu', 'gelu', 'silu']},
            'positional_encoding': {'type': 'categorical', 'choices': ['absolute', 'rotary', 'alibi']},
            'layer_norm_type': {'type': 'categorical', 'choices': ['layernorm', 'rmsnorm', 'groupnorm']},
        },
        'Constraints specification': {
            'memory_constraint': {'batch_size * num_layers': 'max 1024'},
            'time_constraint': {'max_trial_time_minutes': 3.6},
            'learning_rate_constraint': {'high_lr_requires_warmup': True},
        },
        'Objective metric': 'f1_macro',
        'Optimization direction': 'maximize',
        'Baseline objective value': 0.786,
        'Target objective value': 0.82,
        'Evaluation protocol': {
            'metric_type': 'classification',
            'metric_name': 'f1_macro',
            'dataset': 'glue-mrpc',
            'train_examples': 3668,
            'eval_examples': 408,
            'test_examples': 1725,
        },
        'Resource constraints': {
            'max_trials': n_trials,
            'max_gpu_hours': 12.0,
            'max_budget_usd': 48.0,
        },
        'Early stopping policy': {
            'enabled': True,
            'patience': 20,
            'min_delta': 0.001,
        },
        'Solver configuration': {
            'classical_baseline': {
                'name': 'TPE_Optuna',
                'n_trials': n_trials,
                'n_startup_trials': 20,
                'multivariate': True,
                'group': True,
                'constant_liar': True,
                'seed': 42,
            }
        },
        'Initialization / warm start': {
            'enabled': False,
            'num_warm_start_configs': 0,
        },
    }


def print_results_summary(results: dict) -> None:
    """
    Print a formatted summary of optimization results.

    Args:
        results: Results dictionary from solver
    """
    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("=" * 80)

    # Basic metrics
    print(f"\nObjective Value (Best F1_macro): {results['objective_value']:.4f}")
    print(f"Solution Status: {results['solution_status']}")
    print(f"Total Trials Completed: {results['total_trials']}")

    # Computation metrics
    if 'computation_metrics' in results:
        metrics = results['computation_metrics']
        print(f"\nComputation Metrics:")
        print(f"  Mean Score: {metrics.get('mean_score', 0):.4f}")
        print(f"  Max Score: {metrics.get('max_score', 0):.4f}")
        print(f"  Min Score: {metrics.get('min_score', 0):.4f}")
        print(f"  Std Dev: {metrics.get('std_dev', 0):.4f}")
        print(f"  Convergence Rate: {metrics.get('convergence_rate', 0):.4f}")
        print(f"  Time per Trial: {metrics.get('time_per_trial', 0):.2f}s")
        print(f"  Total Time: {metrics.get('total_time_seconds', 0):.2f}s")

    # Cost breakdown
    if 'cost_breakdown' in results:
        costs = results['cost_breakdown']
        print(f"\nCost Breakdown:")
        print(f"  Total GPU Hours: {costs.get('total_gpu_hours', 0):.2f}")
        print(f"  Cost per GPU Hour: ${costs.get('cost_per_gpu_hour', 0):.2f}")
        print(f"  Total Cost: ${costs.get('total_cost_usd', 0):.2f}")
        print(f"  Cost per Trial: ${costs.get('cost_per_trial', 0):.4f}")

    # Benchmark metrics
    if 'benchmark' in results:
        bench = results['benchmark']
        print(f"\nBenchmark Metrics:")
        print(f"  Execution Cost: ${bench.get('execution_cost', 0):.2f}")
        print(f"  Time Elapsed: {bench.get('time_elapsed', 0):.2f}s")
        print(f"  Energy Consumption: {bench.get('energy_consumption_kwh', 0):.4f} kWh")
        print(f"  Trials Completed: {bench.get('trials_completed', 0)}")
        print(f"  Efficiency Metric: {bench.get('efficiency_metric', 0):.4f} (score/sec)")

    # Best parameters
    if 'best_params' in results:
        print(f"\nBest Hyperparameters:")
        for param, value in sorted(results['best_params'].items()):
            if isinstance(value, float):
                print(f"  {param}: {value:.6f}")
            else:
                print(f"  {param}: {value}")

    # Top 10 configurations
    if 'top_10_configurations' in results:
        print(f"\nTop 10 Configurations:")
        for config in results['top_10_configurations']:
            print(f"  Rank {config['rank']}: f1_macro={config['score']:.4f} "
                  f"(trial {config['trial_number']})")

    print("\n" + "=" * 80)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description='QCentroid Classical TPE Solver Test Runner'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of trials to run (default: 50 for testing)'
    )
    parser.add_argument(
        '--n-startup',
        type=int,
        default=20,
        help='Number of startup trials for TPE (default: 20)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--mode',
        choices=['test', 'full'],
        default='test',
        help="Mode: 'test' for quick validation, 'full' for production run"
    )
    parser.add_argument(
        '--save-results',
        type=str,
        default=None,
        help='Path to save results as JSON (optional)'
    )

    args = parser.parse_args()

    # Adjust trials based on mode
    n_trials = args.n_trials
    if args.mode == 'full':
        n_trials = 200
        logger.info("Running in FULL mode (200 trials)")
    else:
        logger.info(f"Running in TEST mode ({n_trials} trials)")

    # Create input data
    logger.info("Creating HPO problem specification...")
    input_data = create_test_input_data(n_trials=n_trials)

    # Prepare solver params
    solver_params = {
        'n_trials': n_trials,
        'n_startup_trials': args.n_startup,
        'seed': args.seed,
    }

    # Run solver
    logger.info("Starting Classical TPE Solver...")
    start_time = time.time()

    results = run(input_data, solver_params, {})

    elapsed = time.time() - start_time
    logger.info(f"Solver execution completed in {elapsed:.2f}s")

    # Print results
    print_results_summary(results)

    # Save results if requested
    if args.save_results:
        output_path = Path(args.save_results)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare JSON-serializable results
        json_results = {}
        for key, value in results.items():
            if key not in ['trial_history']:  # Skip large arrays
                try:
                    json.dumps(value, default=str)  # Test serializability
                    json_results[key] = value
                except (TypeError, ValueError):
                    json_results[key] = str(value)

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)

        logger.info(f"Results saved to {output_path}")

    return 0 if results['solution_status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    sys.exit(main())

