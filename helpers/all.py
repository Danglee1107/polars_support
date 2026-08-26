# --import--
import polars as pl
import matplotlib.pyplot as plt

from typing import TypeAlias
from pathlib import Path

from src.polar_viewer import show

DataSource: TypeAlias = (pl.DataFrame | pl.LazyFrame)

# -- ALL --
def show_null_counts(data: DataSource) -> None:
    """Display the number of null values for every column."""
    df = pl.DataFrame({
        "column": data.columns,
        "dtype": [data[col].dtype for col in data.columns],
        "null_counts": [data[col].null_count() for col in data.columns]
        })
    show(df, title="show_null_counts")
    return

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
