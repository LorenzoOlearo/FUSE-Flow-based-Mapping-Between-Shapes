#!/usr/bin/env python3
# filepath: /home/ubuntu/GeomDist/script_shrec.py
import argparse
import glob
import os
import re
import subprocess
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns


def parse_metric_file(filepath, metric_name):
    """Parse a metric file and return epochs and metric values."""
    epochs = []
    values = []

    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                # Extract epoch and metric value using regex
                pattern = f"Epoch (\\d+): {metric_name}: ([\\d\\.]+)"
                match = re.search(pattern, line)
                if match:
                    epoch = int(match.group(1))
                    value = float(match.group(2))
                    epochs.append(epoch)
                    values.append(value)

    return np.array(epochs), np.array(values)


def create_comparison_plots(
    base_output_dir,
    experiment_configs,
    datasets,
    metrics=["chamfer_distance", "hausdorff_distance"],
):
    """Create comparison plots for all experiment configurations."""
    # Create plots directory
    plots_dir = os.path.join(base_output_dir, "ablation_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set the seaborn style for better looking plots
    sns.set(style="whitegrid")

    # Setup dataframe to collect final results
    results_data = []

    # Map file names to display labels
    metric_display_names = {
        "chamfer_distance": "Chamfer distance",
        "hausdorff_distance": "Hausdorff distance",
    }

    for dataset in datasets:
        base_name = os.path.splitext(os.path.basename(dataset))[0]

        # For each metric (chamfer and hausdorff)
        for metric_name in metrics:
            # Get the actual string pattern to search for in the file
            search_pattern = metric_display_names.get(
                metric_name, metric_name.replace("_", " ")
            )
            metric_label = (
                "Chamfer Distance"
                if metric_name == "chamfer_distance"
                else "Hausdorff Distance"
            )

            # Create plots for different ablation dimensions

            # 1. Distribution comparison (grouped by network and method)
            for network in ["Network", "MLP", "MLP_tiny"]:
                for method in ["FM", "Geomdist"]:
                    plt.figure(figsize=(12, 6))

                    for dist in ["Gaussian", "Sphere", "Gaussian_Optimized"]:
                        # Find the matching config
                        for config in experiment_configs:
                            if (
                                config["network"] == network
                                and config["method"] == method
                                and config["distribution"] == dist
                            ):

                                output_dir = (
                                    f"{base_output_dir}/{base_name}_{config['id']}"
                                )
                                metric_file = os.path.join(
                                    output_dir, f"{metric_name}.txt"
                                )

                                if os.path.exists(metric_file):
                                    # Pass the actual pattern to search for
                                    epochs, values = parse_metric_file(
                                        metric_file, search_pattern
                                    )
                                    if len(epochs) > 0:
                                        plt.plot(
                                            epochs,
                                            values,
                                            label=f"Distribution: {dist}",
                                            marker="o",
                                            markersize=3,
                                            linestyle="-",
                                        )

                                        # Save final value to results
                                        results_data.append(
                                            {
                                                "Dataset": base_name,
                                                "Method": method,
                                                "Network": network,
                                                "Distribution": dist,
                                                "Metric": metric_name,
                                                "Final Value": (
                                                    values[-1]
                                                    if len(values) > 0
                                                    else None
                                                ),
                                                "Best Value": (
                                                    min(values)
                                                    if len(values) > 0
                                                    else None
                                                ),
                                            }
                                        )

                    plt.title(
                        f"{metric_label} - {base_name}\nMethod: {method}, Network: {network}"
                    )
                    plt.xlabel("Epochs")
                    plt.ylabel(metric_label)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.legend()
                    plt.yscale("log")

                    # Save the plot
                    output_file = os.path.join(
                        plots_dir,
                        f"{base_name}_{metric_name}_dist_comparison_{method}_{network}.png",
                    )
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

            # 2. Network comparison (grouped by distribution and method)
            for dist in ["Gaussian", "Sphere", "Gaussian_Optimized"]:
                for method in ["FM", "Geomdist"]:
                    plt.figure(figsize=(12, 6))

                    for network in ["Network", "MLP", "MLP_tiny"]:
                        # Find the matching config
                        for config in experiment_configs:
                            if (
                                config["distribution"] == dist
                                and config["method"] == method
                                and config["network"] == network
                            ):

                                output_dir = (
                                    f"{base_output_dir}/{base_name}_{config['id']}"
                                )
                                metric_file = os.path.join(
                                    output_dir, f"{metric_name}.txt"
                                )

                                if os.path.exists(metric_file):
                                    # Pass the actual pattern to search for
                                    epochs, values = parse_metric_file(
                                        metric_file, search_pattern
                                    )
                                    if len(epochs) > 0:
                                        plt.plot(
                                            epochs,
                                            values,
                                            label=f"Network: {network}",
                                            marker="o",
                                            markersize=3,
                                            linestyle="-",
                                        )

                    plt.title(
                        f"{metric_label} - {base_name}\nMethod: {method}, Distribution: {dist}"
                    )
                    plt.xlabel("Epochs")
                    plt.ylabel(metric_label)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.legend()
                    plt.yscale("log")

                    # Save the plot
                    output_file = os.path.join(
                        plots_dir,
                        f"{base_name}_{metric_name}_network_comparison_{method}_{dist}.png",
                    )
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

            # 3. Method comparison (grouped by distribution and network)
            for dist in ["Gaussian", "Sphere", "Gaussian_Optimized"]:
                for network in ["Network", "MLP", "MLP_tiny"]:
                    plt.figure(figsize=(12, 6))

                    for method in ["FM", "Geomdist"]:
                        # Find the matching config
                        for config in experiment_configs:
                            if (
                                config["distribution"] == dist
                                and config["network"] == network
                                and config["method"] == method
                            ):

                                output_dir = (
                                    f"{base_output_dir}/{base_name}_{config['id']}"
                                )
                                metric_file = os.path.join(
                                    output_dir, f"{metric_name}.txt"
                                )

                                if os.path.exists(metric_file):
                                    # Pass the actual pattern to search for
                                    epochs, values = parse_metric_file(
                                        metric_file, search_pattern
                                    )
                                    if len(epochs) > 0:
                                        plt.plot(
                                            epochs,
                                            values,
                                            label=f"Method: {method}",
                                            marker="o",
                                            markersize=3,
                                            linestyle="-",
                                        )

                    plt.title(
                        f"{metric_label} - {base_name}\nNetwork: {network}, Distribution: {dist}"
                    )
                    plt.xlabel("Epochs")
                    plt.ylabel(metric_label)
                    plt.grid(True, linestyle="--", alpha=0.7)
                    plt.legend()
                    plt.yscale("log")

                    # Save the plot
                    output_file = os.path.join(
                        plots_dir,
                        f"{base_name}_{metric_name}_method_comparison_{network}_{dist}.png",
                    )
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

    # Create summary tables
    results_df = pd.DataFrame(results_data)
    if not results_df.empty:
        # Save detailed results
        results_df.to_csv(
            os.path.join(plots_dir, "ablation_detailed_results.csv"), index=False
        )

        # Create pivot tables for easier analysis
        for metric_name in metrics:
            metric_df = results_df[results_df["Metric"] == metric_name]

            # Summary by distribution and method (averaged across networks)
            dist_method_pivot = metric_df.pivot_table(
                values="Final Value",
                index="Distribution",
                columns="Method",
                aggfunc="mean",
            )
            dist_method_pivot.to_csv(
                os.path.join(
                    plots_dir, f"{metric_name}_distribution_method_summary.csv"
                )
            )

            # Summary by network and method (averaged across distributions)
            network_method_pivot = metric_df.pivot_table(
                values="Final Value", index="Network", columns="Method", aggfunc="mean"
            )
            network_method_pivot.to_csv(
                os.path.join(plots_dir, f"{metric_name}_network_method_summary.csv")
            )

            # Create heatmap visualization
            plt.figure(figsize=(10, 8))
            network_dist_pivot = metric_df.pivot_table(
                values="Final Value",
                index="Network",
                columns=["Method", "Distribution"],
                aggfunc="mean",
            )

            # Plot heatmap if we have data
            if not network_dist_pivot.empty:
                sns.heatmap(
                    network_dist_pivot,
                    annot=True,
                    fmt=".4f",
                    cmap="viridis",
                    linewidths=0.5,
                    cbar_kws={"label": f"Final {metric_label}"},
                )

                plt.title(f"Ablation Study Summary - {metric_label}")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(plots_dir, f"{metric_name}_heatmap.png"), dpi=300
                )
                plt.close()

            # Create tick plots for final values comparison
            create_tick_plots(metric_df, metric_name, metric_label, plots_dir)


def create_tick_plots(metric_df, metric_name, metric_label, plots_dir):
    """Create tick plots for comparing final values across configurations."""

    # 1. Method comparison across networks and distributions
    plt.figure(figsize=(12, 8))

    # Prepare data for tick plot
    fm_data = metric_df[metric_df["Method"] == "FM"]["Final Value"].values
    gd_data = metric_df[metric_df["Method"] == "Geomdist"]["Final Value"].values

    # Only proceed if we have data for both methods
    if len(fm_data) > 0 and len(gd_data) > 0:
        # X positions for the groups
        n_groups = len(fm_data)
        index = np.arange(n_groups)
        bar_width = 0.35

        # Create tick marks with differences highlighted
        plt.bar(index, fm_data, bar_width, label="FM", color="royalblue", alpha=0.7)
        plt.bar(
            index + bar_width,
            gd_data,
            bar_width,
            label="Geomdist",
            color="crimson",
            alpha=0.7,
        )

        # Add value annotations
        for i, v in enumerate(fm_data):
            plt.text(
                i,
                v + 0.001,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                rotation=45,
            )

        for i, v in enumerate(gd_data):
            plt.text(
                i + bar_width,
                v + 0.001,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                rotation=45,
            )

        # Add percentage difference annotations
        for i in range(len(fm_data)):
            if fm_data[i] > 0 and gd_data[i] > 0:
                pct_diff = (gd_data[i] - fm_data[i]) / max(fm_data[i], gd_data[i]) * 100
                color = "green" if pct_diff < 0 else "red"  # Green if FM is better
                plt.text(
                    i + bar_width / 2,
                    max(fm_data[i], gd_data[i]) + 0.003,
                    f"{pct_diff:.1f}%",
                    ha="center",
                    va="bottom",
                    color=color,
                    fontweight="bold",
                    fontsize=8,
                )

        # Customize the plot
        plt.xlabel("Configuration Index")
        plt.ylabel(metric_label)
        plt.title(f"Method Comparison - {metric_label}")
        plt.xticks(index + bar_width / 2, [f"{i+1}" for i in range(n_groups)])
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            os.path.join(plots_dir, f"{metric_name}_method_tick_comparison.png"),
            dpi=300,
        )
        plt.close()

    # 2. Network comparison (aggregated)
    plt.figure(figsize=(10, 6))

    # Group by network and calculate mean
    network_summary = metric_df.groupby("Network")["Final Value"].mean().reset_index()

    # Sort networks by performance
    network_summary = network_summary.sort_values("Final Value")

    # Create tick plot
    ax = sns.barplot(
        x="Network", y="Final Value", data=network_summary, palette="viridis"
    )

    # Add value annotations
    for i, v in enumerate(network_summary["Final Value"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.title(f"Network Performance Comparison - {metric_label}")
    plt.ylabel(metric_label)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"{metric_name}_network_tick_comparison.png"), dpi=300
    )
    plt.close()

    # 3. Distribution comparison (aggregated)
    plt.figure(figsize=(10, 6))

    # Group by distribution and calculate mean
    dist_summary = metric_df.groupby("Distribution")["Final Value"].mean().reset_index()

    # Create tick plot
    ax = sns.barplot(
        x="Distribution", y="Final Value", data=dist_summary, palette="Set2"
    )

    # Add value annotations
    for i, v in enumerate(dist_summary["Final Value"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.title(f"Distribution Performance Comparison - {metric_label}")
    plt.ylabel(metric_label)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"{metric_name}_distribution_tick_comparison.png"),
        dpi=300,
    )
    plt.close()

    # 4. Create a comprehensive grid plot comparing all configurations
    plt.figure(figsize=(15, 10))

    # Create a proper grouping for grid
    grid_data = metric_df.pivot_table(
        values="Final Value", index=["Network", "Distribution"], columns=["Method"]
    ).reset_index()

    # Melt the data for easier plotting
    melted_data = pd.melt(
        grid_data,
        id_vars=["Network", "Distribution"],
        value_vars=["FM", "Geomdist"],
        var_name="Method",
        value_name="Final Value",
    )

    # Create the facet grid
    g = sns.catplot(
        data=melted_data,
        kind="bar",
        x="Method",
        y="Final Value",
        hue="Distribution",
        col="Network",
        height=5,
        aspect=0.8,
        palette="Set2",
        alpha=0.8,
        legend_out=False,
    )

    # Add value annotations to each bar
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    g.fig.suptitle(f"Comprehensive Configuration Comparison - {metric_label}", y=1.05)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"{metric_name}_comprehensive_grid.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def create_mean_plots(
    base_output_dir,
    experiment_configs,
    datasets,
    metrics=["chamfer_distance", "hausdorff_distance"],
):
    """Create plots showing the mean metrics across all shapes for each configuration."""
    # Create plots directory
    plots_dir = os.path.join(base_output_dir, "mean_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Set the seaborn style for better looking plots
    sns.set(style="whitegrid")

    # Map file names to display labels
    metric_display_names = {
        "chamfer_distance": "Chamfer distance",
        "hausdorff_distance": "Hausdorff distance",
    }

    # Dictionary to collect all curve data for averaging
    all_data = {}

    # First, collect all data for each configuration across all datasets
    for metric_name in metrics:
        search_pattern = metric_display_names.get(
            metric_name, metric_name.replace("_", " ")
        )
        metric_label = (
            "Chamfer Distance"
            if metric_name == "chamfer_distance"
            else "Hausdorff Distance"
        )

        # Initialize structure for this metric
        all_data[metric_name] = {}

        # For all configurations
        for config in experiment_configs:
            config_id = config["id"]
            all_data[metric_name][config_id] = {
                "epochs": [],
                "values": [],
                "datasets": [],
            }

            # Collect data from all datasets
            for dataset in datasets:
                base_name = os.path.splitext(os.path.basename(dataset))[0]
                output_dir = f"{base_output_dir}/{base_name}_{config_id}"
                metric_file = os.path.join(output_dir, f"{metric_name}.txt")

                if os.path.exists(metric_file):
                    epochs, values = parse_metric_file(metric_file, search_pattern)
                    if len(epochs) > 0:
                        # Store this dataset's values
                        all_data[metric_name][config_id]["epochs"].append(epochs)
                        all_data[metric_name][config_id]["values"].append(values)
                        all_data[metric_name][config_id]["datasets"].append(base_name)

    # Now create the mean plots for each configuration group
    for metric_name in metrics:
        metric_label = (
            "Chamfer Distance"
            if metric_name == "chamfer_distance"
            else "Hausdorff Distance"
        )

        # 1. Method comparison (averaged across all shapes)
        plt.figure(figsize=(15, 8))

        # Define a consistent color scheme for methods
        method_colors = {"FM": "blue", "Geomdist": "red"}

        # Track the best configurations for each method
        best_configs = {}

        # For each method
        for method in ["FM", "Geomdist"]:
            # For each network
            for network in ["Network", "MLP", "MLP_tiny"]:
                # For each distribution
                for dist in ["Gaussian", "Sphere", "Gaussian_Optimized"]:
                    # Find the matching config
                    for config in experiment_configs:
                        if (
                            config["method"] == method
                            and config["network"] == network
                            and config["distribution"] == dist
                        ):

                            config_id = config["id"]

                            # If we have data for this configuration
                            if (
                                config_id in all_data[metric_name]
                                and len(all_data[metric_name][config_id]["epochs"]) > 0
                            ):
                                # We need to interpolate to get a common x-axis for averaging
                                # Find the maximum common epoch range
                                all_epochs = all_data[metric_name][config_id]["epochs"]
                                all_values = all_data[metric_name][config_id]["values"]

                                if len(all_epochs) > 0:
                                    # Find common epoch range
                                    min_epoch = max(
                                        [epochs[0] for epochs in all_epochs]
                                    )
                                    max_epoch = min(
                                        [epochs[-1] for epochs in all_epochs]
                                    )

                                    # Create a common x-axis
                                    common_epochs = np.arange(
                                        min_epoch, max_epoch + 1, 100
                                    )  # Sample every 100 epochs

                                    # Interpolate all datasets to this common x-axis
                                    interpolated_values = []

                                    for i, (
                                        dataset_epochs,
                                        dataset_values,
                                    ) in enumerate(zip(all_epochs, all_values)):
                                        # Only include epochs in the common range
                                        mask = (dataset_epochs >= min_epoch) & (
                                            dataset_epochs <= max_epoch
                                        )
                                        if np.any(mask):
                                            # Interpolate
                                            interp_func = np.interp(
                                                common_epochs,
                                                dataset_epochs[mask],
                                                dataset_values[mask],
                                            )
                                            interpolated_values.append(interp_func)

                                    if interpolated_values:
                                        # Calculate mean and std across datasets
                                        mean_values = np.mean(
                                            interpolated_values, axis=0
                                        )
                                        std_values = np.std(interpolated_values, axis=0)

                                        # Determine line style based on network
                                        linestyle = (
                                            "-"
                                            if network == "Network"
                                            else ("--" if network == "MLP" else "-.")
                                        )

                                        # Determine alpha based on distribution
                                        if dist == "Gaussian":
                                            alpha = 0.9
                                        elif dist == "Sphere":
                                            alpha = 0.6
                                        else:  # Gaussian_Optimized
                                            alpha = 0.3
                                        # Base color on method
                                        color = method_colors[method]

                                        # Plot mean values
                                        label = f"{method} - {network} - {dist}"
                                        (line,) = plt.plot(
                                            common_epochs,
                                            mean_values,
                                            label=label,
                                            linestyle=linestyle,
                                            color=color,
                                            alpha=alpha,
                                        )

                                        # Add shaded area for standard deviation
                                        plt.fill_between(
                                            common_epochs,
                                            mean_values - std_values,
                                            mean_values + std_values,
                                            alpha=0.2,
                                            color=color,
                                        )

                                        # Track best configuration by final value
                                        final_mean = mean_values[-1]
                                        if (
                                            method not in best_configs
                                            or final_mean
                                            < best_configs[method]["value"]
                                        ):
                                            best_configs[method] = {
                                                "value": final_mean,
                                                "config": config_id,
                                            }

        # Highlight the best configuration for each method
        for method, best in best_configs.items():
            plt.plot(
                [],
                [],
                " ",
                label=f"Best {method}: {best['config']} ({best['value']:.4f})",
            )

        plt.title(f"Mean {metric_label} Across All Shapes")
        plt.xlabel("Epochs")
        plt.ylabel(f"Mean {metric_label}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.yscale("log")

        # Save the plot
        output_file = os.path.join(
            plots_dir, f"mean_{metric_name}_all_configurations.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Simplified method comparison - just the best configurations
        plt.figure(figsize=(12, 6))

        # For each method, show only the best network and distribution
        for method in ["FM", "Geomdist"]:
            if method in best_configs:
                best_config_id = best_configs[method]["config"]

                # Find the original config
                best_config = None
                for config in experiment_configs:
                    if config["id"] == best_config_id:
                        best_config = config
                        break

                if best_config:
                    # Get the data for this configuration
                    config_data = all_data[metric_name][best_config_id]

                    # Recalculate the mean values
                    all_epochs = config_data["epochs"]
                    all_values = config_data["values"]

                    if len(all_epochs) > 0:
                        # Find common epoch range
                        min_epoch = max([epochs[0] for epochs in all_epochs])
                        max_epoch = min([epochs[-1] for epochs in all_epochs])

                        # Create a common x-axis
                        common_epochs = np.arange(min_epoch, max_epoch + 1, 100)

                        # Interpolate all datasets to this common x-axis
                        interpolated_values = []

                        for i, (dataset_epochs, dataset_values) in enumerate(
                            zip(all_epochs, all_values)
                        ):
                            # Only include epochs in the common range
                            mask = (dataset_epochs >= min_epoch) & (
                                dataset_epochs <= max_epoch
                            )
                            if np.any(mask):
                                # Interpolate
                                interp_func = np.interp(
                                    common_epochs,
                                    dataset_epochs[mask],
                                    dataset_values[mask],
                                )
                                interpolated_values.append(interp_func)

                        if interpolated_values:
                            # Calculate mean and std across datasets
                            mean_values = np.mean(interpolated_values, axis=0)
                            std_values = np.std(interpolated_values, axis=0)

                            # Plot mean values
                            label = f"{method} ({best_config['network']}, {best_config['distribution']})"
                            color = method_colors[method]
                            plt.plot(
                                common_epochs,
                                mean_values,
                                label=label,
                                color=color,
                                linewidth=2,
                            )

                            # Add shaded area for standard deviation
                            plt.fill_between(
                                common_epochs,
                                mean_values - std_values,
                                mean_values + std_values,
                                alpha=0.2,
                                color=color,
                            )

        plt.title(f"Mean {metric_label} - Best Configurations Only")
        plt.xlabel("Epochs")
        plt.ylabel(f"Mean {metric_label}")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend()
        plt.yscale("log")

        # Save the plot
        output_file = os.path.join(
            plots_dir, f"mean_{metric_name}_best_configurations.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Create a plot showing final mean values for each configuration
        plt.figure(figsize=(14, 8))

        # Collect final mean values for each configuration
        config_names = []
        final_means = []
        final_stds = []
        config_methods = []

        for config in experiment_configs:
            config_id = config["id"]

            if (
                config_id in all_data[metric_name]
                and len(all_data[metric_name][config_id]["epochs"]) > 0
            ):
                # Calculate mean final value across datasets
                all_values = all_data[metric_name][config_id]["values"]

                if all_values:
                    # Extract final value from each dataset
                    final_values = [values[-1] for values in all_values]

                    if final_values:
                        mean_final = np.mean(final_values)
                        std_final = np.std(final_values)

                        config_names.append(config_id)
                        final_means.append(mean_final)
                        final_stds.append(std_final)
                        config_methods.append(config["method"])

        # Create a DataFrame for easier plotting
        final_data = pd.DataFrame(
            {
                "Configuration": config_names,
                "Final Mean": final_means,
                "Final Std": final_stds,
                "Method": config_methods,
            }
        )

        # Sort by performance
        final_data = final_data.sort_values("Final Mean")

        # Plot the bar chart
        ax = sns.barplot(
            x="Configuration",
            y="Final Mean",
            hue="Method",
            data=final_data,
            palette=method_colors,
        )

        # Add value annotations
        for i, v in enumerate(final_data["Final Mean"]):
            ax.text(
                i,
                v + 0.001,
                f"{v:.4f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=8,
                rotation=45,
            )

        plt.title(f"Mean Final {metric_label} by Configuration")
        plt.xlabel("Configuration")
        plt.ylabel(f"Mean Final {metric_label}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(plots_dir, f"mean_{metric_name}_final_values.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study across multiple configurations"
    )
    parser.add_argument(
        "--base_output_dir", default="ablation", help="Base directory for outputs"
    )
    parser.add_argument("--epochs", default=10000, type=int, help="Number of epochs")
    parser.add_argument(
        "--num_points", default=10000, type=int, help="Number of points for inference"
    )
    parser.add_argument("--num_steps", default=64, type=int, help="Number of steps")
    parser.add_argument(
        "--mesh_dir",
        default="../datasets/meshes/SHREC20/off/",
        help="Directory containing mesh files",
    )
    parser.add_argument(
        "--data_files", nargs="+", default=[], help="Specific data files to process"
    )
    parser.add_argument("--batch_size", default=10000, type=int, help="Batch size")
    parser.add_argument("--blr", default=5e-7, type=float, help="Base learning rate")
    parser.add_argument(
        "--learning_rate", default=0.01, type=float, help="Learning rate"
    )
    parser.add_argument(
        "--plot_only", action="store_true", help="Only generate plots without training"
    )
    args = parser.parse_args()

    # Find all mesh files in the specified directory if no specific files are provided
    data_files = args.data_files
    if not data_files:
        mesh_dir = args.mesh_dir
        for ext in ["*.off"]:
            data_files.extend(glob.glob(os.path.join(mesh_dir, ext)))

    if not data_files:
        print(f"No mesh files found in {args.mesh_dir}")
        return

    print(f"Found {len(data_files)} mesh files to process")

    # Define experiment configurations for the ablation study
    experiment_configs = []

    # Generate all combinations of parameters
    param_combinations = list(
        product(["Sphere"], ["Network"], ["FM"])  # distributions  # networks  # methods
    )

    # Create configuration objects
    for dist, network, method in param_combinations:
        model = "FMCond" if method == "FM" else "EDMPrecond"
        config_id = f"{method}_{network}_{dist}"

        experiment_configs.append(
            {
                "id": config_id,
                "distribution": dist,
                "network": network,
                "method": method,
                "model": model,
            }
        )

    if args.plot_only:
        # Generate plots without running training
        create_comparison_plots(args.base_output_dir, experiment_configs, data_files)
        return

    # Process each mesh file
    for data_file in data_files:
        # Extract base name without extension for directory naming
        base_name = os.path.splitext(os.path.basename(data_file))[0]

        # Run all experiment configurations
        for config in experiment_configs:
            # Create output directory name
            output_dir = f"{args.base_output_dir}/{base_name}_{config['id']}"

            # Build the command
            cmd = [
                "python",
                "main.py",
                "--blr",
                str(args.blr),
                "--output_dir",
                output_dir,
                "--log_dir",
                output_dir,
                "--data_path",
                data_file,
                "--train",
                "--inference",
                "--epochs",
                str(args.epochs),
                "--num-steps",
                str(args.num_steps),
                "--method",
                config["method"],
                "--model",
                config["model"],
                "--network",
                config["network"],
                "--batch_size",
                str(args.batch_size),
                "--num_points_train",
                str(args.num_points),
                "--learning_rate",
                str(args.learning_rate),
                "--distribution",
                config["distribution"],
                "--num_points_inference",
                str(args.num_points),
                "--device",
                "cuda:0",
            ]

            # Print the command being executed
            print(f"\n\n{'='*80}")
            print(f"Running: {' '.join(cmd)}")
            print(f"{'='*80}\n")

            # Execute the command
            subprocess.run(cmd)

        # After all configurations for this dataset, create comparison plots
        # create_comparison_plots(args.base_output_dir, experiment_configs, [data_file])

    # Final summary plots across all datasets
    create_mean_plots(args.base_output_dir, experiment_configs, data_files)
    create_comparison_plots(args.base_output_dir, experiment_configs, data_files)


if __name__ == "__main__":
    main()
