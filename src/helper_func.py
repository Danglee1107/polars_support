# --import--
import polars as pl
import polars.selectors as cs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# --visualize table--
from src.polar_viewer import show, show_tables

from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# -- typing --
from typing import TypeAlias, Literal
from pathlib import Path

DataSource: TypeAlias= (pl.DataFrame | pl.LazyFrame)

# -- NUMERIC TYPE --
def show_describe_numeric(data: DataSource) -> None:
    """Display descriptive statistics for all numeric columns.

    Shows count, null count, mean, variance, quartiles, skewness, and kurtosis.
    """
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
    """Display transformation diagnostics for numeric columns.

    Compares skewness before and after Box-Cox or Yeo-Johnson transformations.
    """
    numeric_cols = data.select(cs.numeric()).columns
    row = []
    for col in numeric_cols:
        res = transform_diagnostics(data, col)
        row.append({"columns": col, **res})

    df = pl.DataFrame(row)
    show(df, title="show_transform_diagnostics")
    return 

def show_iqr_outlier_flags_matrix(data: DataSource,k: float=1.5) -> None:
    """Display an IQR-based outlier flag matrix.

    Each column contains boolean flags indicating whether a value is an IQR outlier.
    """
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
    """Display a Z-score outlier flag matrix.

    Flags observations whose absolute Z-score exceeds the given threshold.
    """
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

def show_mod_zscore_flags_matrix(data: DataSource, threshold: float = 3.5) -> None:
    """Display a modified Z-score outlier flag matrix.

    Uses the median and MAD to identify robust outliers.
    """
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

# -- ALL --
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

def show_null_counts(data: DataSource) -> None:
    """Display the number of null values for every column."""
    df = pl.DataFrame({
        "column": data.columns,
        "dtype": [data[col].dtype for col in data.columns],
        "null_counts": [data[col].null_count() for col in data.columns]
        })
    show(df, title="show_null_counts")
    return

def transform_diagnostics(data: DataSource, col: str) -> dict:
    """Evaluate transformations for reducing skewness.

    Uses Box-Cox for strictly positive data and Yeo-Johnson otherwise.
    """
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
    """Return boolean flags for IQR-based outliers in a numeric column."""
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr

    return data.select(((
                pl.col(col) < lower) | (pl.col(col) > upper))
                .alias((f"{col}_iqr_outlier")))

def zscore_flags(data: DataSource, col, threshold: float = 3.0) -> DataSource:
    """Return boolean flags for Z-score outliers in a numeric column."""
    mean, std = data[col].mean(), data[col].std()
    return data.select(
            (((pl.col(col) - mean)/ std).abs() > threshold)
            .alias(f"{col}_zscore_outlier"))


def modified_zscore_flags(data: DataSource, col: str,
                          threshold: float = 3.5) -> DataSource:
    """Return boolean flags for modified Z-score outliers.

    Uses the median absolute deviation (MAD) for robust detection.
    """
    median = data[col].median()
    mad = ((data[col] - median).abs()).median() # median absolute deviation
    # 0.6745 scales MAD to be comparable to std under normality
    return data.select(
            (((pl.col(col) - median) * 0.6745/ mad).abs() > threshold)
            .alias(f"{col}_mod_zscore_outlier"))

def show_isolation_forest_flags_matrix(data: DataSource,
                                       contamination: float = 0.02) -> None:
    """Display Isolation Forest anomaly detection results.

    Shows predicted outlier labels and anomaly scores for numeric data.
    """
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
                              dist: str ="norm",
                              path: Path | None = None) -> None:
    """Plot the distribution of each numeric column.

    Generates a histogram with KDE, boxplot, and Q-Q plot, and optionally saves the figures.
    """
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

def plot_heatmap(data: DataSource,
                 method: Literal['pearson', 'spearman', 'kendall'] = 'pearson',
                 path: Path | None = None) -> None:
    '''correlation_matrix'''
    numeric_df = data.select(cs.numeric())
    corr = numeric_df.to_pandas().corr(method=method)
    n = len(corr.columns)

    # Adjust these values to your preference
    cell_size = 0.7      # inches per cell
    min_size = 6         # minimum figure size
    max_size = 20        # maximum figure size

    fig_size = max(min_size, min(max_size, n * cell_size))

    # Annotation font size
    annot_size = max(5, min(12, 16 - n // 2))

    plt.figure(figsize=(fig_size, fig_size))

    sns.heatmap(corr,
                annot=True,
                annot_kws={'size': annot_size},
                cmap='coolwarm',
                fmt='.2f',
                linewidths=0.5,
                linecolor='black',
                square=True,
                xticklabels=corr.columns,
                yticklabels=corr.columns)
    plt.title('Correlation matrix')

    if path:
        plt.savefig(path / "correlation_matrix.png")

    plt.show()
    return

# -- Bivariate/multivariate --
def show_cat_cat(data: DataSource) -> None:
    '''category-category'''
    results= []
    for c1, c2 in combinations(category_cols,2):
        ct = df.group_by([c1, c2]).agg(pl.len().alias("n")).pivot(
            values="n", index=c1, columns=c2, aggregate_function="first"
        ).fill_null(0)
        ct_np = ct.drop(c1).to_numpy()

        if ct_np.shape[0] < 2 or ct_np.shape[1] < 2:
            continue

        chi2, p, dof, exp = chi2_contingency(ct_np)
        n = ct_np.sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct_np.shape) -1 )))
        results.append({'col1': c1, 'col2': c2,
                        'chi2': chi2, 'p_value': p,
                        'cramers_v': cramers_v})
    cat_cat_results = pl.DataFrame(results).sort('cramers_v', descending=True)

    show(cat_cat_results)
    return

def show_num_cat(data: DataSource, show_boxplot: bool = False) -> None:
    '''
    numeric-category
    show_boxplot: bool is display top-N by eta_squared
    '''

    numeric_cols = data.select(cs.numeric()).columns
    category_cols = data.select(cs.string()).columns

    results = []
    for cat_col in category_cols:
        for num_col in numeric_cols:
            groups_df = data.select([cat_col, num_col]).drop_nulls()
            groups = [g[num_col].to_numpy() for _, g in groups_df.group_by(cat_col)]
            groups = [g for g in groups if len(g) > 1]

            if len(groups) < 2:
                continue

            f_stat, p_anova = f_oneway(*groups)
            h_stat, p_kw = kruskal(*groups)

            grand_mean = groups_df[num_col].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
            ss_total = ((groups_df[num_col] - grand_mean) ** 2).sum()
            eta_sq = ss_between / ss_total if ss_total > 0 else np.nan

            results.append({
                'cat_col': cat_col, 'num_col': num_col,
                'f_stat': f_stat, 'p_anova': p_anova,
                'h_stat': h_stat, 'p_kruskal': p_kw,
                'eta_squared': eta_sq
                })

    num_cat_results = pl.DataFrame(results).sort('eta_squared', descending=True)
    show(num_cat_results)

    if show_boxplot:
        top_pairs = num_cat_results.head(6).iter_rows(named=True)
        fig, axes = plt.subplots(2,3, figsize=(15,8))
        for ax, row in zip(axes.flat, top_pairs):
            sns.boxplot(data=data.select([row['cat_col'], row['num_col']]).to_pandas(),
                        x=row['cat_col'], y=row['num_col'], ax=ax)
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()

    return


def plot_pca_analysis(data: DataSource, path: Path | None = None) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    X = StandardScaler().fit_transform(data.select(numeric_cols).fill_null(0).to_numpy())
    pca = PCA().fit(X)

    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('n_components')
    plt.ylabel('cumulative explained variance')
    
    if path:
        plt.savefig(path / 'pca_analysis')

    plt.show()
    return

def plot_hierarchical_clustering(data: DataSource, path: Path | None = None) -> None:
    numeric_cols = data.select(cs.numeric()).columns
    pearson = data.select(numeric_cols).corr()
    corr_np = pearson.to_numpy()
    dist = 1 - np.abs(corr_np)
    np.fill_diagonal(dist, 0)
    condensed = dist[np.triu_indices_from(dist, k=1)]
    Z = linkage(condensed, method='average')

    plt.figure(figsize=(12,6))
    dendrogram(Z, labels=numeric_cols)
    plt.xticks(rotation=90)
    plt.title('hierarchical clustering of numeric features')

    if path:
        plt.savefig(path / 'hierarchical_clustering_of_numeric_features.png')

    plt.tight_layout()
    plt.show()
    return
