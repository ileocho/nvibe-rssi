import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import math

def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers is 6371
    m = 6371 * c * 1000
    return m

def plot_predictions(y_test: np.array, predictions: np.array, output: str):
    """Plots the predictions vs the targets in a scatter plot.

    Args:
        y_test: The test targets.
        predictions: The predictions.
        output: The path to save the plot.
    """
    fig = px.scatter(
        x=y_test[:, 0],
        y=y_test[:, 1],
        title="Predictions vs Targets",
        labels={"x": "Targets", "y": "Predictions"},
    )

    fig.add_trace(
        go.Scatter(
            x=predictions[:, 0],
            y=predictions[:, 1],
            mode="markers",
            marker=dict(color="orange", size=3, symbol="x"),
        )
    )

    fig.update_layout(width=600 * 1.34, height=600)
    fig.write_image(f"{output}/predictions_vs_targets.png")
    fig.show()


def plot_ecdf(y_test: np.array, predictions: np.array, output: str):
    """Plots the ECDF of the errors in meters.

    Args:
        y_test: the test targets.
        predictions: the predictions.
        output: the path to save the plot
    """

    # Calculate errors in meters
    errors = np.array(
        [
            haversine(lat1, long1, lat2, long2)
            for (lat1, long1), (lat2, long2) in zip(predictions, y_test)
        ]
    )

    # Calculate ECDF
    cdf = stats.cumfreq(errors, numbins=30)
    x = cdf.lowerlimit + np.linspace(0, cdf.binsize * cdf.cumcount.size, cdf.cumcount.size)

    # Plot ECDF
    fig = px.line(
        x=x,
        y=cdf.cumcount / cdf.cumcount[-1],
        title="ECDF of Errors",
        labels={"x": "Error (m)", "y": "Cumulative Count"},
    )

    # Add vertical line at 1 meter
    fig.add_shape(
        type="line",
        x0=1,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="red", width=1, dash="dash"),
    )

    # add 25th and 75th percentiles
    fig.add_shape(
        type="line",
        x0=np.percentile(errors, 25),
        y0=0,
        x1=np.percentile(errors, 25),
        y1=1,
        line=dict(color="green", width=1, dash="dash"),
    )

    fig.add_shape(
        type="line",
        x0=np.percentile(errors, 75),
        y0=0,
        x1=np.percentile(errors, 75),
        y1=1,
        line=dict(color="green", width=1, dash="dash"),
    )

    fig.add_shape(
        type="line",
        x0=np.percentile(errors, 95),
        y0=0,
        x1=np.percentile(errors, 95),
        y1=1,
        line=dict(color="green", width=1, dash="dash"),
    )

    fig.update_layout(width=600 * 1.34, height=600)
    fig.write_image(f"{output}/ecdf.png")
    fig.show()


def plot_error_hist(y_test: np.array, predictions: np.array, output: str):
    """Plots the an histogram of the errors in one meter intervals.

    Args:
        y_test: The test targets.
        predictions: The predictions.
        output: The path to save the plot.
    """

    # Calculate errors in meters
    errors = np.array(
        [
            haversine(lat1, long1, lat2, long2)
            for (lat1, long1), (lat2, long2) in zip(predictions, y_test)
        ]
    )

    # plot error histogram with plotly
    fig = px.histogram(x=errors, title="Error histogram")
    fig = fig.update_xaxes(title_text="Error (m)")
    fig = fig.update_yaxes(title_text="Frequency")
    fig.write_image(f"{output}/error_hist.png")
    fig.show()