"""
Visualization module for Classical TPE solver.

This module provides standalone functions to generate HTML visualization files
for hyperparameter optimization results. It creates:
- Convergence plots showing trial scores and running best
- Hyperparameter importance charts
- Score distribution histograms
- Top configurations tables
- Search summary dashboards
"""

import logging
import os
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def generate_classical_visualizations(results: Dict[str, Any], trial_history: List[Dict]) -> None:
    """
    Generate all visualization HTML files in additional_output/ directory.

    Main entry point for visualization generation. Creates comprehensive charts
    and tables from optimization results.

    Args:
        results: Optimization results dictionary with objective_value, benchmark, etc.
        trial_history: List of trial history dictionaries with score and params
    """
    try:
        os.makedirs('additional_output', exist_ok=True)

        # Extract data for visualizations
        all_values = [t['score'] for t in trial_history]
        best_value = results['objective_value']
        top_configs = results['top_10_configurations']

        # Generate each visualization
        _create_convergence_plot(trial_history, all_values)
        _create_hyperparameter_importance(trial_history)
        _create_score_distribution(all_values, best_value)
        _create_top_configurations_table(top_configs)
        _create_search_summary(results)

        logger.info("Visualizations generated successfully in additional_output/")

    except Exception as e:
        logger.warning(f"Failed to generate visualizations: {e}")


def _create_convergence_plot(trial_history: List[Dict], all_values: List[float]) -> None:
    """Create convergence plot showing trial scores and running best."""
    if not trial_history:
        return

    # Calculate running best
    running_best = []
    best_so_far = float('-inf')
    for item in trial_history:
        best_so_far = max(best_so_far, item['score'])
        running_best.append(best_so_far)

    trial_nums = [item['trial_num'] for item in trial_history]
    scores = [item['score'] for item in trial_history]

    # Calculate SVG dimensions and scaling
    width, height = 800, 400
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    min_val = min(min(scores), min(running_best)) * 0.95
    max_val = max(max(scores), max(running_best)) * 1.05
    val_range = max_val - min_val if max_val > min_val else 1

    min_trial = min(trial_nums)
    max_trial = max(trial_nums)
    trial_range = max_trial - min_trial if max_trial > min_trial else 1

    # Generate SVG
    svg_lines = ['<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">']
    svg_lines.append('<style>')
    svg_lines.append('  .grid { stroke: #e5e7eb; stroke-width: 1; }')
    svg_lines.append('  .axis { stroke: #374151; stroke-width: 2; }')
    svg_lines.append('  .axis-label { font-size: 12px; fill: #374151; }')
    svg_lines.append('  .title { font-size: 16px; font-weight: bold; fill: #1f2937; }')
    svg_lines.append('</style>')

    # Grid lines and axes
    svg_lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" class="axis"/>')
    svg_lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" class="axis"/>')

    # Grid
    for i in range(5):
        y = margin + (i * plot_height / 4)
        svg_lines.append(f'<line x1="{margin}" y1="{y}" x2="{width - margin}" y2="{y}" class="grid"/>')

    # Axis labels
    for i in range(5):
        val = min_val + (i * val_range / 4)
        y = height - margin - (i * plot_height / 4)
        svg_lines.append(f'<text x="{margin - 45}" y="{y + 4}" class="axis-label">{val:.3f}</text>')

    for i in range(5):
        trial = min_trial + int(i * trial_range / 4)
        x = margin + (i * plot_width / 4)
        svg_lines.append(f'<text x="{x}" y="{height - margin + 20}" class="axis-label">{trial}</text>')

    # Plot trial scores (light blue)
    for i, (trial, score) in enumerate(zip(trial_nums, scores)):
        x = margin + ((trial - min_trial) / trial_range * plot_width) if trial_range > 0 else margin
        y = height - margin - ((score - min_val) / val_range * plot_height) if val_range > 0 else height - margin
        svg_lines.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#93c5fd" opacity="0.6"/>')

    # Plot running best line (blue)
    best_points = []
    for i, (trial, best) in enumerate(zip(trial_nums, running_best)):
        x = margin + ((trial - min_trial) / trial_range * plot_width) if trial_range > 0 else margin
        y = height - margin - ((best - min_val) / val_range * plot_height) if val_range > 0 else height - margin
        best_points.append((x, y))

    if best_points:
        path_data = ' '.join([f"{'M' if i == 0 else 'L'} {x} {y}" for i, (x, y) in enumerate(best_points)])
        svg_lines.append(f'<path d="{path_data}" stroke="#2563eb" stroke-width="2" fill="none"/>')

    # Labels
    svg_lines.append(f'<text x="400" y="25" class="title" text-anchor="middle">Optimization Convergence</text>')
    svg_lines.append(f'<text x="{margin - 45}" y="35" class="axis-label" text-anchor="end">Score</text>')
    svg_lines.append(f'<text x="{width // 2}" y="{height - 10}" class="axis-label" text-anchor="middle">Trial Number</text>')

    # Legend
    svg_lines.append(f'<circle cx="650" cy="50" r="3" fill="#93c5fd"/>')
    svg_lines.append(f'<text x="660" y="55" class="axis-label">Trial Score</text>')
    svg_lines.append(f'<line x1="650" y1="70" x2="670" y2="70" stroke="#2563eb" stroke-width="2"/>')
    svg_lines.append(f'<text x="680" y="75" class="axis-label">Running Best</text>')

    svg_lines.append('</svg>')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Optimization Convergence</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #1f2937; margin-bottom: 20px; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        svg {{ display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Convergence</h1>
        <div class="chart-container">
            {"".join(svg_lines)}
        </div>
    </div>
</body>
</html>'''

    with open('additional_output/convergence_plot.html', 'w') as f:
        f.write(html_content)


def _create_hyperparameter_importance(trial_history: List[Dict]) -> None:
    """Create hyperparameter importance chart based on correlation with score."""
    if not trial_history:
        return

    # Extract all parameters and scores
    scores = [t['score'] for t in trial_history]
    all_params = {}

    for trial in trial_history:
        for param_name, param_value in trial['params'].items():
            if param_name not in all_params:
                all_params[param_name] = []
            all_params[param_name].append(param_value)

    # Calculate correlation for each parameter
    correlations = {}
    for param_name, values in all_params.items():
        # Check if numeric or categorical
        is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

        if is_numeric:
            # Pearson correlation
            mean_score = sum(scores) / len(scores)
            mean_param = sum(values) / len(values)
            numerator = sum((s - mean_score) * (v - mean_param) for s, v in zip(scores, values))
            denom_s = (sum((s - mean_score) ** 2 for s in scores)) ** 0.5
            denom_v = (sum((v - mean_param) ** 2 for v in values)) ** 0.5
            if denom_s > 0 and denom_v > 0:
                corr = numerator / (denom_s * denom_v)
                correlations[param_name] = abs(corr)
            else:
                correlations[param_name] = 0.0
        else:
            # Mean score per category for categorical
            categories = {}
            for val, score in zip(values, scores):
                if val not in categories:
                    categories[val] = []
                categories[val].append(score)

            cat_means = {cat: sum(s) / len(s) for cat, s in categories.items()}
            overall_mean = sum(scores) / len(scores)
            variance_between = sum(len(cat_scores) * (mean_score - overall_mean) ** 2
                                 for cat, cat_scores in categories.items()
                                 for mean_score in [sum(cat_scores) / len(cat_scores)])
            variance_total = sum((s - overall_mean) ** 2 for s in scores)
            if variance_total > 0:
                eta = (variance_between / variance_total) ** 0.5
                correlations[param_name] = eta
            else:
                correlations[param_name] = 0.0

    # Sort by importance
    sorted_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]

    # Create SVG bar chart
    width, height = 800, 400
    margin = 80
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    max_corr = max([v for _, v in sorted_params]) if sorted_params else 0.1
    bar_width = plot_width / (len(sorted_params) * 1.5) if sorted_params else 50

    svg_lines = ['<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">']
    svg_lines.append('<style>')
    svg_lines.append('  .bar { fill: #2563eb; }')
    svg_lines.append('  .axis { stroke: #374151; stroke-width: 2; }')
    svg_lines.append('  .axis-label { font-size: 11px; fill: #374151; }')
    svg_lines.append('  .title { font-size: 16px; font-weight: bold; fill: #1f2937; }')
    svg_lines.append('</style>')

    # Axes
    svg_lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" class="axis"/>')
    svg_lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" class="axis"/>')

    # Draw bars
    for i, (param_name, corr_value) in enumerate(sorted_params):
        bar_height = (corr_value / max_corr) * plot_height if max_corr > 0 else 0
        x = margin + i * (plot_width / len(sorted_params)) + (plot_width / (len(sorted_params) * 2) - bar_width / 2)
        y = height - margin - bar_height

        svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" class="bar"/>')

        # Label
        label_x = x + bar_width / 2
        label_y = height - margin + 35
        svg_lines.append(f'<text x="{label_x}" y="{label_y}" class="axis-label" text-anchor="middle">{param_name}</text>')
        svg_lines.append(f'<text x="{label_x}" y="{y - 5}" class="axis-label" text-anchor="middle">{corr_value:.2f}</text>')

    # Y-axis labels
    for i in range(5):
        val = (i / 4) * max_corr
        y = height - margin - (i * plot_height / 4)
        svg_lines.append(f'<text x="{margin - 10}" y="{y + 4}" class="axis-label" text-anchor="end">{val:.2f}</text>')

    svg_lines.append(f'<text x="400" y="25" class="title" text-anchor="middle">Hyperparameter Importance</text>')
    svg_lines.append(f'<text x="{margin - 50}" y="35" class="axis-label" text-anchor="end">Correlation</text>')

    svg_lines.append('</svg>')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hyperparameter Importance</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #1f2937; margin-bottom: 20px; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        svg {{ display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Hyperparameter Importance</h1>
        <div class="chart-container">
            {"".join(svg_lines)}
        </div>
    </div>
</body>
</html>'''

    with open('additional_output/hyperparameter_importance.html', 'w') as f:
        f.write(html_content)


def _create_score_distribution(all_values: List[float], best_value: float) -> None:
    """Create histogram of score distribution."""
    if not all_values:
        return

    min_val = min(all_values)
    max_val = max(all_values)
    val_range = max_val - min_val if max_val > min_val else 0.1

    # Create bins
    n_bins = min(20, len(all_values))
    bins = [min_val + (i / n_bins) * val_range for i in range(n_bins + 1)]
    bin_counts = [0] * n_bins

    for val in all_values:
        bin_idx = min(int((val - min_val) / val_range * n_bins), n_bins - 1) if val_range > 0 else 0
        bin_counts[bin_idx] += 1

    mean_val = sum(all_values) / len(all_values)
    max_count = max(bin_counts)

    # Create SVG
    width, height = 800, 400
    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    svg_lines = ['<svg width="800" height="400" xmlns="http://www.w3.org/2000/svg">']
    svg_lines.append('<style>')
    svg_lines.append('  .bar { fill: #10b981; }')
    svg_lines.append('  .axis { stroke: #374151; stroke-width: 2; }')
    svg_lines.append('  .axis-label { font-size: 12px; fill: #374151; }')
    svg_lines.append('  .title { font-size: 16px; font-weight: bold; fill: #1f2937; }')
    svg_lines.append('  .mean-line { stroke: #f59e0b; stroke-width: 2; }')
    svg_lines.append('  .best-line { stroke: #ef4444; stroke-width: 2; }')
    svg_lines.append('</style>')

    # Axes
    svg_lines.append(f'<line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" class="axis"/>')
    svg_lines.append(f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" class="axis"/>')

    # Draw bins
    bar_width = plot_width / n_bins
    for i, count in enumerate(bin_counts):
        bar_height = (count / max_count) * plot_height if max_count > 0 else 0
        x = margin + i * bar_width
        y = height - margin - bar_height

        svg_lines.append(f'<rect x="{x}" y="{y}" width="{bar_width - 2}" height="{bar_height}" class="bar"/>')

    # Mean line
    mean_x = margin + ((mean_val - min_val) / val_range * plot_width) if val_range > 0 else margin
    svg_lines.append(f'<line x1="{mean_x}" y1="{margin}" x2="{mean_x}" y2="{height - margin}" class="mean-line" stroke-dasharray="5,5"/>')

    # Best value line
    best_x = margin + ((best_value - min_val) / val_range * plot_width) if val_range > 0 else margin
    svg_lines.append(f'<line x1="{best_x}" y1="{margin}" x2="{best_x}" y2="{height - margin}" class="best-line" stroke-dasharray="5,5"/>')

    # Labels
    svg_lines.append(f'<text x="400" y="25" class="title" text-anchor="middle">Score Distribution</text>')
    svg_lines.append(f'<text x="{margin - 45}" y="35" class="axis-label" text-anchor="end">Count</text>')
    svg_lines.append(f'<text x="{width // 2}" y="{height - 10}" class="axis-label" text-anchor="middle">Score Value</text>')

    # Legend
    svg_lines.append(f'<line x1="650" y1="50" x2="670" y2="50" class="mean-line" stroke-dasharray="5,5"/>')
    svg_lines.append(f'<text x="680" y="55" class="axis-label">Mean: {mean_val:.3f}</text>')
    svg_lines.append(f'<line x1="650" y1="70" x2="670" y2="70" class="best-line" stroke-dasharray="5,5"/>')
    svg_lines.append(f'<text x="680" y="75" class="axis-label">Best: {best_value:.3f}</text>')

    svg_lines.append('</svg>')

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Score Distribution</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #1f2937; margin-bottom: 20px; }}
        .chart-container {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        svg {{ display: block; margin: 0 auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Score Distribution</h1>
        <div class="chart-container">
            {"".join(svg_lines)}
        </div>
    </div>
</body>
</html>'''

    with open('additional_output/score_distribution.html', 'w') as f:
        f.write(html_content)


def _create_top_configurations_table(top_configs: List[Dict]) -> None:
    """Create HTML table of top 10 configurations."""
    if not top_configs:
        return

    # Get all parameter names from first config
    param_names = sorted(list(top_configs[0]['params'].keys())) if top_configs else []

    table_rows = []
    for config in top_configs:
        rank = config['rank']
        score = config['score']
        trial = config['trial_number']
        params = config['params']

        row = f'<tr><td class="rank">{rank}</td><td class="score">{score:.4f}</td><td class="trial">{trial}</td>'
        for param in param_names:
            val = params.get(param, '')
            if isinstance(val, float):
                row += f'<td class="param">{val:.4e}</td>'
            else:
                row += f'<td class="param">{val}</td>'
        row += '</tr>'
        table_rows.append(row)

    param_headers = ''.join([f'<th>{p}</th>' for p in param_names])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Top Configurations</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 20px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #1f2937; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th {{ background: #2563eb; color: white; padding: 12px; text-align: left; font-weight: 600; border: none; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }}
        tr:hover {{ background: #f9fafb; }}
        tr:last-child td {{ border-bottom: none; }}
        .rank {{ color: #2563eb; font-weight: 600; }}
        .score {{ color: #10b981; font-weight: 600; }}
        .trial {{ color: #6b7280; }}
        .param {{ font-family: monospace; font-size: 12px; color: #374151; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Top 10 Configurations</h1>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Score</th>
                    <th>Trial</th>
                    {param_headers}
                </tr>
            </thead>
            <tbody>
                {"".join(table_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>'''

    with open('additional_output/top_configurations.html', 'w') as f:
        f.write(html_content)


def _create_search_summary(results: Dict[str, Any]) -> None:
    """Create dashboard with search space definition and results summary."""
    best_value = results['objective_value']
    best_trial = results['best_trial_number']
    total_trials = results['total_trials']
    metrics = results['computation_metrics']
    benchmark = results['benchmark']
    best_params = results['best_params']

    # Format best parameters
    params_html = ''.join([f'<tr><td>{k}</td><td><code>{v if not isinstance(v, float) else f"{v:.4e}"}</code></td></tr>'
                          for k, v in sorted(best_params.items())])

    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Summary</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f9fafb; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        h1 {{ color: #1f2937; margin-bottom: 30px; }}
        h2 {{ color: #374151; margin-top: 30px; margin-bottom: 15px; border-bottom: 2px solid #2563eb; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .metric-label {{ color: #6b7280; font-size: 12px; text-transform: uppercase; font-weight: 600; }}
        .metric-value {{ color: #1f2937; font-size: 24px; font-weight: bold; margin-top: 5px; }}
        .metric-value.best {{ color: #10b981; }}
        .metric-value.trial {{ color: #2563eb; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        th {{ background: #f3f4f6; color: #374151; padding: 12px; text-align: left; font-weight: 600; border-bottom: 1px solid #e5e7eb; }}
        td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; }}
        tr:last-child td {{ border-bottom: none; }}
        code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Optimization Search Summary</h1>

        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Best Score</div>
                <div class="metric-value best">{best_value:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Best Trial</div>
                <div class="metric-value trial">{best_trial}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total Trials</div>
                <div class="metric-value">{total_trials}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Score</div>
                <div class="metric-value">{metrics.get('mean_score', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Std Dev</div>
                <div class="metric-value">{metrics.get('std_dev', 0):.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Time (seconds)</div>
                <div class="metric-value">{benchmark.get('time_elapsed', 0):.1f}</div>
            </div>
        </div>

        <h2>Best Configuration</h2>
        <table>
            <thead>
                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                {params_html}
            </tbody>
        </table>

        <h2>Optimization Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Mean Score</td><td>{metrics.get('mean_score', 0):.4f}</td></tr>
                <tr><td>Max Score</td><td>{metrics.get('max_score', 0):.4f}</td></tr>
                <tr><td>Min Score</td><td>{metrics.get('min_score', 0):.4f}</td></tr>
                <tr><td>Std Dev</td><td>{metrics.get('std_dev', 0):.4f}</td></tr>
                <tr><td>Convergence Rate</td><td>{metrics.get('convergence_rate', 0):.4f}</td></tr>
                <tr><td>Time per Trial (s)</td><td>{metrics.get('time_per_trial', 0):.4f}</td></tr>
            </tbody>
        </table>

        <h2>Benchmark Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Execution Cost</td><td>${{benchmark.get('execution_cost', 0):.2f}}</td></tr>
                <tr><td>Time Elapsed (s)</td><td>{benchmark.get('time_elapsed', 0):.2f}</td></tr>
                <tr><td>Energy (kWh)</td><td>{benchmark.get('energy_consumption', 0):.4f}</td></tr>
                <tr><td>Trials Completed</td><td>{benchmark.get('trials_completed', 0)}</td></tr>
                <tr><td>Efficiency (score/s)</td><td>{benchmark.get('efficiency_metric', 0):.6f}</td></tr>
            </tbody>
        </table>

    </div>
</body>
</html>'''

    with open('additional_output/search_summary.html', 'w') as f:
        f.write(html_content)