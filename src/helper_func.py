import polars as pl
import polars.selectors as cs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.polar_viewer import show, show_tables
from scipy import stats
from sklearn.ensemble import IsolationForest
# -- typing --
from typing import TypeAlias
from pathlib import Path

DataSource: TypeAlias= (pl.DataFrame | pl.LazyFrame)

# -- NUMERIC TYPE --
def show_describe_numeric(data: DataSource) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    summary = pl.DataFrame({
        "columns": numeric_cols,
        "count": [data[col].count() for col in numeric_cols],
        "null_count": [data[col].null_count() for col in numeric_cols],
        "mean": [data[col].mean() for col in numeric_cols],
        "std": [data[col].std() for col in numeric_cols],
        "var": [data[col].var() for col in numeric_cols],
        "min": [data[col].min() for col in numeric_cols],
        "Q1": [data[col].quantile(0.25) for col in numeric_cols],
        "Q2": [data[col].median() for col in numeric_cols],
        "Q3": [data[col].quantile(0.75) for col in numeric_cols],
        "max": [data[col].max() for col in numeric_cols],
        "skew": [data[col].skew() for col in numeric_cols],
        "kurtosis": [data[col].kurtosis() for col in numeric_cols],
        })
    show(summary, title="show_describe_numeric")
    return

def show_transform_diagnostics(data: DataSource) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    row = []
    for col in numeric_cols:
        res = transform_diagnostics(data, col)
        row.append({"columns": col, **res})

    df = pl.DataFrame(row)
    show(df, title="show_transform_diagnostics")
    return 

def show_iqr_outlier_flags_matrix(data: DataSource,k: float=1.5) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    df = None
    for col in numeric_cols:

        col_flag = iqr_outlier_flags(data, col, k)
        if df is None:
            df = col_flag
            continue

        df = pl.concat([df, col_flag], how="horizontal")
    show(df, title="show_iqr_outlier_flags_matrix")
    return

def show_zscore_flags_matrix(data: DataSource, threshold: float = 3.0) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    df = None
    for col in numeric_cols:

        col_flag = zscore_flags(data, col, threshold)
        if df is None:
            df = col_flag
            continue

        df = pl.concat([df, col_flag], how="horizontal")
    show(df, title="show_zscore_flags_matrix")
    return

def show_mod_zscore_flags_matrie(data: DataSource, threshold: float = 3.5) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    df = None
    for col in numeric_cols:

        col_flag = modified_zscore_flags(data, col, threshold)
        if df is None:
            df = col_flag
            continue

        df = pl.concat([df, col_flag], how="horizontal")
    show(df, title="show_mod_zscore_flags_matrix")
    return

# -- CATEGORY TYPE --
def show_describe_category(data: DataSource) -> None:
    category_cols = data.select(cs.string()).columns
    summary = []
    for col in category_cols:
        res = summary_categorical(data,col)
        summary.append({"columns": col, 
                   "distinct_count": data[col].n_unique(),
                   **res})

    summary = pl.DataFrame(summary)
    show(summary, title="show_describe_category")
    return

def show_relative_category(data: DataSource) -> None:
    category_cols = data.select(cs.string()).columns
    freqs = []
    for col in category_cols:
        freq = get_cat_freq(data,col)
        freqs.append(freq)
    show_tables(freqs, title="show_relative_category")
    return

def show_rare_category(data: DataSource) -> None:
    threshold = len(data) * 0.01
    category_cols = data.select(cs.string()).columns

    rares = []
    for col in category_cols:
        rare = (
            data[col]
            .value_counts()
            .filter(pl.col("count") < threshold)
        )
        rares.append(rare)
    show_tables(rares, title="show_rare_category")
    return

# -- ALL --
def get_cat_freq(data: DataSource, col: str) -> DataSource:
    return (data[col]
            .value_counts()
            .sort("count", descending=True)
            .with_columns(
                (pl.col("count") / len(data)).alias("proportion")
                )
            )

def summary_categorical(data: DataSource, col: str) -> dict:
    freq = get_cat_freq(data,col)
    mode = freq[col][0]
    probs = freq["proportion"].to_numpy() 

    # in bits; 0 = diversity, log2(k) = uniform over k categories
    entropy = -(probs * np.log2(probs)).sum() 

    return {
        "mode": mode,
        "entropy_bits": entropy,
        "max_entropy_bits": np.log2(freq.height) # for comparison - how close to uniform
            }

def show_null_counts(data: DataSource) -> None:
    df = pl.DataFrame({
        "column": data.columns,
        "dtype": [data[col].dtype for col in data.columns],
        "null_counts": [data[col].null_count() for col in data.columns]
        })
    show(df, title="show_null_counts")
    return

def transform_diagnostics(data: DataSource, col: str) -> dict:
    s = data[col].drop_nulls()
    result = {"raw_skew": stats.skew(s)}

    if (s > 0).all():
        log_s = np.log(s)
        result["log_skew"] = stats.skew(log_s)

        bc, lam = stats.boxcox(s)
        result["boxcox_skew"] = stats.skew(bc)
        result["boxcox_lambda"] = lam

    else: # yeo handes zero/negative values while boxcox doesn't
        yj, lam = stats.yeojohnson(s)
        result["yeojohnson_skew"] = stats.skew(yj)
        result["yeojohnson_lambda"] = lam

    return result

def iqr_outlier_flags(data: DataSource, col: str, k: float=1.5) -> DataSource:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    return data.select(((
                pl.col(col) < lower) | (pl.col(col) > upper))
                .alias((f"{col}_iqr_outlier")))

def zscore_flags(data: DataSource, col, threshold: float = 3.0) -> DataSource:
    mean, std = data[col].mean(), data[col].std()
    return data.select(
            (((pl.col(col) - mean)/ std).abs() > threshold)
            .alias(f"{col}_zscore_outlier"))


def modified_zscore_flags(data: DataSource, col, threshold: float = 3.5) -> DataSource:
    median = data[col].median()
    mad = ((data[col] - median).abs()).median() # median absolute deviation
    # 0.6745 scales MAD to be comparable to std under normality
    return data.select(
            (((pl.col(col) - median) * 0.6745/ mad).abs() > threshold)
            .alias(f"{col}_mod_zscore_outlier"))

def show_isolation_forest_flags_matrix(data: DataSource,
                                       contamination: float = 0.02) -> None:
    X = data.select(cs.numeric()).to_numpy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X) # -1 = outlier, 1 = inlier
    scores = iso.decision_function(X) # lower = more anomalous

    flag_df = pl.DataFrame({
        "iso_outlier": preds==-1,
        "iso_score": scores,
        })
    show(flag_df, title="show_isolation_forest_flags_matrix")
    return

# -- Visualization --
def map_quit() -> None:
    '''
    add key map to quit matplotlib widget
    how to use:
        add this before plt.show()
    '''
    fig = plt.gcf()

    def on_key(event):
        if event.key == "escape":
            plt.close(event.canvas.figure)
    fig.canvas.mpl_connect("key_press_event", on_key)
    return

def plot_distribution_numeric(data: DataSource,
                              dist="norm",
                              path: Path | None = None) -> None:
    numeric_cols = data.select(cs.numeric()).columns

    for col in numeric_cols:
        fig, axes = plt.subplots(1,3, figsize=(20,6))

        # Histogram + KDE
        sns.histplot(data[col], kde=True, ax=axes[0])
        axes[0].set_title(f"{col}-hist_and_kde")

        # Boxplot for quick visualize IQR/Outliers
        sns.boxplot(x=data[col], ax=axes[1])
        axes[1].set_title(f"{col}-boxplot")

        # Q-Q plot against theoretical distribution
        stats.probplot(data[col], dist=dist, plot=axes[2])
        axes[2].set_title(f"{col}-Q-Q plot vs {dist}")

        plt.tight_layout()
        map_quit()
        if path:
            plt.savefig(path / f"{col}_distribution.png")

        plt.show()
    return
