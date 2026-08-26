# --import--
import polars as pl
import polars.selectors as cs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from scipy.stats import chi2_contingency, f_oneway, kruskal
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from typing import Literal
from pathlib import Path

from src.polar_viewer import show

from .all import DataSource, map_quit

# -- Bivariate/multivariate --
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

    map_quit()
    plt.show()
    return

def show_cat_cat(data: DataSource) -> None:
    '''category-category'''
    category_cols = data.select(cs.string()).columns
    results= []
    for c1, c2 in combinations(category_cols,2):
        ct = data.group_by([c1, c2]).agg(pl.len().alias("n")).pivot(
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

        map_quit()
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

    map_quit()
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

    map_quit()
    plt.tight_layout()
    plt.show()
    return
