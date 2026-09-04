import ipywidgets as widgets
import numpy as np
import pandas as pd


def show_columns_overview(df: pd.DataFrame, n_cols: int = 4) -> widgets.VBox:
    """Builds a widget overview of all DataFrame columns, grouped by dtype.

    Args:
        df: The DataFrame to inspect.
        n_cols: Number of columns in the display grid.

    Returns:
        A VBox widget listing numeric and non-numeric columns separately,
        each entry shown as "name (dtype)".
    """
    numeric_cols = sorted(df.select_dtypes(include=np.number).columns.tolist())
    other_cols = [c for c in df.columns if c not in numeric_cols]

    def make_grid(cols):
        entries = [
            widgets.HTML(
                f"<span style='font-family:monospace'>{c}</span> "
                f"<span style='color:grey;font-size:11px'>({df[c].dtype})</span>"
            )
            for c in cols
        ]
        return widgets.GridBox(
            entries, layout=widgets.Layout(grid_template_columns=f"repeat({n_cols}, minmax(150px, 1fr))")
        )

    sections = [
        widgets.HTML(f"<b>Numeric columns ({len(numeric_cols)}):</b>"),
        make_grid(numeric_cols) if numeric_cols else widgets.Label("None"),
    ]
    if other_cols:
        sections += [
            widgets.HTML(f"<b>Other columns ({len(other_cols)}):</b>"),
            make_grid(other_cols),
        ]
    # no interactivity possible if display is inside function, because then the widget is not returned to the notebook
    return widgets.VBox(sections)
