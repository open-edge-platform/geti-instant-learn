# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
from argparse import Namespace

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_priors_distrib(
    df: pd.DataFrame, metric: str, swap_rc: bool = False, plot: str = "violinplot", height: int = 5
) -> None:
    """This method plots a distribution.

    Args:
        df: Dataframe containing the calculated metrics.
        metric: The metric to show on the y-axis.
        swap_rc: Swap rows and columns of the facet grid of the catplot.
        plot: type of plot to use.
        height: height of the plot.
    """
    if metric not in {"f1score", "iou", "precision", "recall"}:
        msg = f"Unknown metric {metric}"
        raise ValueError(msg)
    if plot not in {"violin", "bar"}:
        msg = f"Unknown plot {plot}"
        raise ValueError(msg)
    plot_params = {"violin": {"cut": 0}, "bar": {}}
    # noinspection PyTypeChecker
    sns.catplot(
        data=df,
        x="pipeline_name",
        y=metric,
        col="dataset_name" if swap_rc else "backbone_name",
        row="backbone_name" if swap_rc else "dataset_name",
        kind=plot,
        height=height,
        aspect=1,
        hue="pipeline_name",
        **plot_params[plot],
    )

    plt.tight_layout()
    plt.show()


def plot_priors_distrib_melt(dfm: pd.DataFrame, metrics: list[str], backbone: str | None = None) -> None:
    """This method draws a distribution using a melted dataframe.

    Args:
        dfm: Melted Dataframe containing the calculated metrics.
        metrics: The metrics to show in the facet grid.
        backbone: The backbone to show.
    """
    dfm_part = dfm[dfm["metric"].isin(metrics)]
    if backbone is not None:
        dfm_part = dfm_part[(dfm_part["backbone_name"] == backbone)]
    sns.catplot(
        data=dfm_part,
        x="pipeline_name",
        y="value",
        col="dataset_name",
        row="metric",
        kind="violin",
        height=4,
        aspect=1.5,
        cut=0,
        hue="pipeline_name",
    )
    plt.tight_layout()
    plt.show()


def melt_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """This method created a melted dataframe.

    Args:
        df: Dataframe containing the calculated metrics.
    """
    return pd.melt(
        df,
        id_vars=["dataset_name", "pipeline_name", "backbone_name"],
        value_vars=[
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1score",
            "jaccard",
            "iou",
            "dice",
            "accuracy",
        ],
        var_name="metric",
        value_name="value",
    )


def plot_backbone_distrib_melt(
    dfm: pd.DataFrame,
    metrics: list[str],
    plot: str = "violinplot",
    height: int = 5,
    pipeline_name: str = "MatcherModular",
) -> None:
    """This shows the metrics per backbone of a single pipeline.

    Args:
        dfm: A melted dataframe.
        metrics: The metric to show in the facet grid.
        plot: The type of plot to use.
        height: The height of the plot
        pipeline_name: The name of the pipeline.
    """
    if set(metrics).difference({"f1score", "iou", "precision", "recall"}) != set():
        msg = "Unknown metric " + ",".join(set(metrics).difference({"f1score", "iou", "precision", "recall"}))
        raise ValueError(msg)
    if plot not in {"violin", "bar"}:
        msg = f"Unknown plot {plot}"
        raise ValueError(msg)
    dfm = dfm[dfm["metric"].isin(metrics)]
    dfm = dfm[dfm["pipeline_name"] == pipeline_name]

    plot_params = {"violin": {"cut": 0}, "bar": {}}

    # noinspection PyTypeChecker
    sns.catplot(
        data=dfm,
        x="backbone_name",
        y="value",
        col="metric",
        row="dataset_name",
        kind=plot,
        height=height,
        aspect=1,
        **plot_params[plot],
    )

    plt.tight_layout()
    plt.show()


def main(args: Namespace) -> None:
    """Main method.

    Args:
        args: The arguments from the command line.
    """
    filename = args.file

    # Read and process data
    metrics_df = pd.read_csv(filename)
    dfm = melt_metrics(metrics_df)

    # Plot backbone metrics for the MatcherModular pipeline
    plot_backbone_distrib_melt(
        dfm, metrics=["precision", "recall", "f1score"], plot="violin", height=5, pipeline_name="MatcherModular"
    )

    # Show the distribution over the priors and the categories
    plot_priors_distrib_melt(dfm, metrics=["precision", "recall", "f1score"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="The filename containing metrics")
    main(parser.parse_args())
