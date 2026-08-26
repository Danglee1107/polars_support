# -- import --
from pandas.io.formats.style_render import _parse_latex_header_span
import polars as pl
import pandas as pd
import numpy as np
import matplotlib
import pyarrow

# -- use gnome theme (ignore) -- 
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import seaborn as sns
import polars.selectors as cs
import os

from scipy import stats
from sklearn.ensemble import IsolationForest

import src.helper_func as hf
from src.polar_viewer import show, show_tables
from src.dir_conf import GRAPH

# --ignore Qt warning --
os.environ["QT_LOGGING_RULES"] = "qt.qpa.wayland.textinput=false"

df = pl.read_csv("sample.csv",
                 infer_schema_length=None,
                 null_values=["NA","N/A", "None", "null", "unknown"])

# --fill null value--
df = df.with_columns(
    [
        pl.col(col).fill_null(pl.col(col).median())
        for col in df.select(cs.numeric().exclude(["age", "languages", 
                                                   "timetaken", "year"])).columns
    ] 
    +[
        pl.col(col).fill_null(pl.col(col).mode().first())
        for col in ["gender","bagcarry","handed"]
    ]
    +[
        pl.col(col).fill_null(pl.col(col).median())
        .round()
        .cast(pl.Int64)
        for col in ["age","languages"]
    ]
    +[
        pl.col(["country","travel","favlearning",
            "fitlevel","prefquality","superpower"]).fill_null("Unknown")
    ]
)

numeric_cols = df.select(cs.numeric()).columns
category_cols = df.select(cs.string()).columns

hf.plot_pca_analysis(df, path=GRAPH)
hf.plot_hierarchical_clustering(df, path=GRAPH)
