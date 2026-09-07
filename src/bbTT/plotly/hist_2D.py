import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import clear_output, display

import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HistogramPlotter2D:
    def __init__(
        self,
        df: pd.DataFrame,
        predefined_cuts: dict[str],
        column_overrides: dict[str] = None,
        n_bins_default: int = 30,
        n_bins_range: tuple[int, int] = (5, 100),
        clip_quantiles: tuple[float, float] = (0.01, 0.99),
        n_cols: int = 2,
        plot_pairs: list[dict] = None,
    ):
        """
        Interactive feature-vs-feature 2D plotter with per-plot cuts, a
        dynamic add menu, live column overrides, and sidebars to remove
        registered plots/overrides.

        Args:
            df (pd.DataFrame): DataFrame containing the event data.
            predefined_cuts (dict[str]): Name -> pandas query string (global, AND-combined).
            column_overrides (dict[dict[str]]) : Optional initial column name -> {"range": (lo, hi), "n_bins": int}.
                Can also be extended live through the UI.
            n_bins_default (int): Default bin count for columns without an override.
            n_bins_range (tuple[int, int]): (min, max) for the global bins slider.
            clip_quantiles (tuple[float, float]): (low, high) percentiles defining the default range.
            n_cols (int) : Number of columns in the subplot grid.
            plot_pairs (list[dict]): List of plots that should be displayed by default.
        """
        self.df = df
        self.predefined_cuts = predefined_cuts
        if not isinstance(predefined_cuts, dict):
            raise TypeError(f"Predefined cuts should be a dict and not {type(predefined_cuts)}")

        self.column_overrides = dict(column_overrides or {})
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.n_bins_default = n_bins_default
        self.n_bins_range = n_bins_range
        self.clip_quantiles = clip_quantiles
        self.n_cols = n_cols
        self.plot_pairs = plot_pairs if plot_pairs else []  # list of {"x": ..., "y": ..., "cut": ...}

        self._build_widgets()
        self._render_sidebar()
        self._render_override_sidebar()
        self._render_all()

    def _get_axis_edges(self, col, n_bins_global):
        lo, hi = self.column_overrides.get(col, {}).get("range") or tuple(self.df[col].quantile(self.clip_quantiles))
        n_bins = self.column_overrides.get(col, {}).get("n_bins", n_bins_global)
        return np.linspace(lo, hi, n_bins + 1)

    def _combine_masks(self, query_strings):
        """AND-combines multiple pandas query strings into one boolean mask."""
        mask = pd.Series(True, index=self.df.index)
        for q in query_strings:
            if not q.strip():
                continue
            mask &= self.df.eval(q)
        return mask

    def _make_filter_toggle(self, name):
        btn = widgets.ToggleButton(value=False, description=name, layout=widgets.Layout(width="auto"))
        btn.observe(self._on_toggle_style, names="value")
        return btn

    def _on_toggle_style(self, change):
        change.owner.button_style = "success" if change["new"] else ""

    def _build_widgets(self):
        self.cut_toggles = {name: self._make_filter_toggle(name) for name in self.predefined_cuts}
        self.custom_cut_text = widgets.Text(
            value="", description="Global custom cut:", layout=widgets.Layout(width="400px")
        )
        self.plot_cut_text = widgets.Text(value="", description="Plot cut:", layout=widgets.Layout(width="400px"))
        self.bins_slider = widgets.IntSlider(
            value=self.n_bins_default, min=self.n_bins_range[0], max=self.n_bins_range[1], description="N bins:"
        )
        self.x_dropdown = widgets.Dropdown(options=self.numeric_cols, description="X:")
        self.y_dropdown = widgets.Dropdown(options=self.numeric_cols, description="Y:")
        self.add_button = widgets.Button(description="+ Add plot")
        self.update_button = widgets.Button(description="Update", button_style="primary")
        self.status_label = widgets.Label(value="")
        self.output = widgets.Output()
        self.sidebar = widgets.VBox([])

        self.override_col_dropdown = widgets.Dropdown(options=self.numeric_cols, description="Column:")
        self.override_min_text = widgets.Text(value="", description="Range min:", layout=widgets.Layout(width="150px"))
        self.override_max_text = widgets.Text(value="", description="Range max:", layout=widgets.Layout(width="150px"))
        self.override_bins_text = widgets.Text(value="", description="Bins:", layout=widgets.Layout(width="100px"))
        self.add_override_button = widgets.Button(description="+ Add override")
        self.override_sidebar = widgets.VBox([])

        self.add_button.on_click(self._on_add_clicked)
        self.update_button.on_click(self._render_all)
        self.add_override_button.on_click(self._on_add_override_clicked)

        left = widgets.VBox(
            [
                widgets.HTML("<b>Clickable filters (global, AND-combined):</b>"),
                widgets.HBox(list(self.cut_toggles.values())),
                self.custom_cut_text,
                widgets.HTML("<b>Add a 2D plot:</b>"),
                widgets.HBox([self.x_dropdown, self.y_dropdown]),
                self.plot_cut_text,
                self.add_button,
                widgets.HTML("<b>Add a column override:</b>"),
                widgets.HBox(
                    [
                        self.override_col_dropdown,
                        self.override_min_text,
                        self.override_max_text,
                        self.override_bins_text,
                    ]
                ),
                self.add_override_button,
                self.bins_slider,
                self.update_button,
                self.status_label,
                self.output,
            ]
        )
        right = widgets.VBox(
            [
                widgets.HTML("<b>Registered plots:</b>"),
                self.sidebar,
                widgets.HTML("<b>Column overrides:</b>"),
                self.override_sidebar,
            ]
        )
        self.layout = widgets.HBox([left, right])

    # --- Sidebar for registered plots ---
    def _render_sidebar(self):
        """Rebuilds the plot sidebar. Each remove button carries its pair
        directly as an attribute (button._pair) — a single shared callback
        reads it off the button instance on_click passes in, avoiding the
        loop-closure pitfall entirely (no per-item lambda/default-arg needed)."""
        rows = []
        for pair in self.plot_pairs:
            remove_btn = widgets.Button(description="✕", layout=widgets.Layout(width="30px"))
            remove_btn._pair = pair
            remove_btn.on_click(self._on_remove_pair_clicked)
            cut_display = pair["cut"] if pair["cut"] else "(no plot cut)"
            label = widgets.VBox(
                [
                    widgets.Label(f"{pair['x']} vs {pair['y']}"),
                    widgets.HTML(f"<span style='font-size:10px;color:grey'>{cut_display}</span>"),
                ]
            )
            rows.append(widgets.HBox([label, remove_btn]))
        self.sidebar.children = rows if rows else [widgets.Label("No plots registered.")]

    def _on_remove_pair_clicked(self, button):
        pair = button._pair
        if pair in self.plot_pairs:
            self.plot_pairs.remove(pair)
            self._render_sidebar()
            self._render_all()

    # --- Sidebar for column overrides ---
    def _render_override_sidebar(self):
        rows = []
        for col, override in self.column_overrides.items():
            remove_btn = widgets.Button(description="✕", layout=widgets.Layout(width="30px"))
            remove_btn._col = col
            remove_btn.on_click(self._on_remove_override_clicked)
            parts = []
            if "range" in override:
                parts.append(f"range=({override['range'][0]:.3g}, {override['range'][1]:.3g})")
            if "n_bins" in override:
                parts.append(f"n_bins={override['n_bins']}")
            rows.append(widgets.HBox([widgets.Label(f"{col}: {', '.join(parts)}"), remove_btn]))
        self.override_sidebar.children = rows if rows else [widgets.Label("No column overrides.")]

    def _on_remove_override_clicked(self, button):
        self.column_overrides.pop(button._col, None)
        self._render_override_sidebar()
        self._render_all()

    def _on_add_override_clicked(self, _):
        col = self.override_col_dropdown.value
        entry = {}
        min_str, max_str = self.override_min_text.value.strip(), self.override_max_text.value.strip()
        if min_str and max_str:
            try:
                lo, hi = float(min_str), float(max_str)
            except ValueError:
                self.status_label.value = "Range min/max must be numeric."
                return
            if lo >= hi:
                self.status_label.value = "Range min must be smaller than max."
                return
            entry["range"] = (lo, hi)
        elif min_str or max_str:
            self.status_label.value = "Provide both range min and max, or leave both empty."
            return

        bins_str = self.override_bins_text.value.strip()
        if bins_str:
            try:
                n_bins = int(bins_str)
            except ValueError:
                self.status_label.value = "Bins must be an integer."
                return
            if n_bins < 2:
                self.status_label.value = "Bins must be at least 2."
                return
            entry["n_bins"] = n_bins

        if not entry:
            self.status_label.value = "Provide a range and/or a bin count to override."
            return

        self.column_overrides[col] = entry
        self.status_label.value = ""
        self._render_override_sidebar()
        self._render_all()

    def _on_add_clicked(self, _):
        new_pair = {"x": self.x_dropdown.value, "y": self.y_dropdown.value, "cut": self.plot_cut_text.value.strip()}
        if new_pair["x"] == new_pair["y"]:
            self.status_label.value = "X and Y must be different."
            return
        if new_pair in self.plot_pairs:
            self.status_label.value = "This exact plot+cut combination already exists."
            return
        self.plot_pairs.append(new_pair)
        self._render_sidebar()
        self._render_all()

    def _render_all(self, _=None):
        global_queries = [self.predefined_cuts[name] for name, btn in self.cut_toggles.items() if btn.value]
        if self.custom_cut_text.value.strip():
            global_queries.append(self.custom_cut_text.value)
        try:
            global_mask = self._combine_masks(global_queries)
        except Exception as e:
            self.status_label.value = f"⚠️ Invalid global cut: {e}"
            return
        self.status_label.value = ""

        if not self.plot_pairs:
            with self.output:
                clear_output(wait=True)
                print("No 2D plots added yet. Pick X/Y, optionally a plot cut, and click '+'.")
            return

        n_rows = int(np.ceil(len(self.plot_pairs) / self.n_cols))
        titles = [f"{p['x']} vs {p['y']}" for p in self.plot_pairs]
        fig = make_subplots(rows=n_rows, cols=self.n_cols, subplot_titles=titles)

        for idx, pair in enumerate(self.plot_pairs):
            row, c = idx // self.n_cols + 1, idx % self.n_cols + 1
            x_col, y_col, plot_cut = pair["x"], pair["y"], pair["cut"]
            try:
                combined_mask = global_mask & self.df.eval(plot_cut) if plot_cut.strip() else global_mask
            except Exception as e:
                fig.add_trace(go.Heatmap(x=[], y=[], z=[[]]), row=row, col=c)
                fig.layout.annotations[idx].text = f"{x_col} vs {y_col}<br>ERROR in plot cut: {e}"
                continue

            x_edges = self._get_axis_edges(x_col, self.bins_slider.value)
            y_edges = self._get_axis_edges(y_col, self.bins_slider.value)
            sub_x = self.df.loc[combined_mask, x_col]
            sub_y = self.df.loc[combined_mask, y_col]

            counts, xe, ye = np.histogram2d(sub_x, sub_y, bins=[x_edges, y_edges])
            x_centers = (xe[:-1] + xe[1:]) / 2
            y_centers = (ye[:-1] + ye[1:]) / 2

            fig.add_trace(
                go.Heatmap(x=x_centers, y=y_centers, z=counts.T, colorscale="Viridis", showscale=(idx == 0)),
                row=row,
                col=c,
            )
            fig.update_xaxes(title_text=x_col, row=row, col=c)
            fig.update_yaxes(title_text=y_col, row=row, col=c)

            n_selected = int(combined_mask.sum())
            n_outside = n_selected - int(counts.sum())
            cut_suffix = f" | cut: {plot_cut}" if plot_cut.strip() else ""
            fig.layout.annotations[idx].text = (
                f"{x_col} vs {y_col}{cut_suffix}<br>"
                f"<span style='font-size:10px;color:grey'>outside range: {n_outside} of {n_selected}</span>"
            )

        fig.update_layout(height=350 * n_rows, width=400 * self.n_cols, title_text="2D feature correlations")
        self.output_widget = go.FigureWidget(fig)
        with self.output:
            clear_output(wait=True)
            display(self.output_widget)

    def show(self):
        display(self.layout)
