# --import--
import polars as pl
import polars.selectors as cs
import numpy as np

from src.polar_viewer import show, show_tables

from .all import DataSource

# -- CATEGORY TYPE --
def show_describe_category(data: DataSource) -> None:
    """Display summary statistics for categorical columns.

    Shows the number of unique values, mode, and entropy-based diversity measures.
    """
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
    """Display frequency tables for categorical columns.

    Shows counts and relative proportions for each category.
    """
    category_cols = data.select(cs.string()).columns
    freqs = []
    for col in category_cols:
        freq = get_cat_freq(data,col)
        freqs.append(freq)
    show_tables(freqs, title="show_relative_category")
    return

def show_rare_category(data: DataSource) -> None:
    """Display rare categories for each categorical column.

    Categories occurring in less than 1% of rows are reported.
    """
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

def get_cat_freq(data: DataSource, col: str) -> DataSource:
    """Return category frequencies and proportions for a column."""
    return (data[col]
            .value_counts()
            .sort("count", descending=True)
            .with_columns(
                (pl.col("count") / len(data)).alias("proportion")
                )
            )

def summary_categorical(data: DataSource, col: str) -> dict:
    """Compute summary statistics for a categorical column.

    Returns the mode, entropy, and maximum possible entropy.
    """
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
