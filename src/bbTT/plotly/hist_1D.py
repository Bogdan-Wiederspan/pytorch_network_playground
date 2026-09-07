import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class HistogramPlotter1D:
    def __init__(
        self,
        df: pd.DataFrame,
        predefined_cuts: dict[str],
        process_column: str,
        process_definitions: dict[str],
        columns: list[str] = None,
        n_bins: int = 30,
        n_bins_range: tuple[int, int] = (5, 100),
        n_cols: int = 3,
        clip_quantiles: tuple[float, float] = (0.01, 0.99),
        column_overrides: dict[str] = None,
        include_other: bool = True,
        weight_column_name: str = "",
        process_colors: dict[str] = None,
        plot_width: int = 350,
        plot_height: int = 300,
        opacity_filling: float = 0.5,
        family_hatch: dict[str] = None,
    ):
        """
        1D histogram plotter. Families of processes ('family:sub' naming)
        are rendered as CUMULATIVE STEP LINES (not filled bars) - lines never
        occlude each other regardless of overlap, so every sub-process AND every
        family's total remains simultaneously visible and comparable.

        Args:
            df (Pandas.DataFrame): Pandas dataframe with all columns that are plotted
            predefined_cuts (dict[str]): Dictionary of cuts that can be toggled on and off.
            process_column (str): Name of the column that defines the process ids
            process_definitions (dict[str]): Dictionary of all pids that belong to a process
            columns (list[str], optional): Columns in df that should be plotted. If None, all columns in df that are numerical will be plotted. Defaults to None.
            n_bins (int, optional): Default value for bins. Defaults to 30.
            n_bins_range (tuple[int, int], optional): Range for the bin slider. Defaults to (5, 100).
            n_cols (int, optional): Number of columns. Defaults to 3.
            clip_quantiles (tuple[float, float], optional): Tuple of quantile ranges. Defaults to (0.01, 0.99).
            column_overrides (dict[str], optional): Dictionary with settings to override plot defaults. Defaults to None.
            include_other (bool, optional): All processes that are not required by user are gathered in OTHER. Defaults to True.
            weight_column_name (str, optional): Name of the Column where the applied weights are stored. If not there, no event weights are applied.
            process_colors (dict[str], None): ,
            plot_width (int): Width of each plot,
            plot_height (int): Height of each plot,
            opacity_filling (float): Opacity of the filling - 1 is solid, 0 is no filling,
            family_hatch (None | dict[str]): Pattern used for different families. If None no pattern is used.


        Raises:
            ValueError: _description_
            ValueError: _description_
        """
        self.width = plot_width
        self.height = plot_height

        if process_column not in df.columns:
            raise ValueError(f"process_column '{process_column}' not found in df.")
        self.df = df
        self.predefined_cuts = predefined_cuts
        self.process_column = process_column
        self.column_overrides = dict(column_overrides or {})
        self.n_bins_range = n_bins_range
        self.n_cols = n_cols
        self.clip_quantiles = clip_quantiles
        self.weight_column_name = weight_column_name
        self.n_bins = n_bins
        self.process_color_overrides = process_colors
        self.opacity_filling = opacity_filling
        self.family_hatch = family_hatch if family_hatch else {}

        if columns is not None:
            missing = [c for c in columns if c not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in df: {missing}")
            self.numeric_cols = list(columns)
        else:
            self.numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != process_column]

        self.value_range = {}
        for col in self.numeric_cols:
            override_range = self.column_overrides.get(col, {}).get("range")
            self.value_range[col] = (
                override_range if override_range is not None else tuple(df[col].quantile(clip_quantiles))
            )

        if not isinstance(predefined_cuts, dict):
            raise TypeError(f"Predefined cuts should be a dict and not {type(predefined_cuts)}")

        self._build_process_masks(process_definitions, include_other)
        self._build_process_colors()
        self._build_process_families()
        self._build_figure()
        self._build_widgets()
        self.update_plots()

    def _build_columns(self, columns: None, df=None, exclude_columns: list[str] = ("pid")):
        """
        Helper to
        Builds the list of numeric columns to plot, either from the provided"""

        # use only given columns and remove excluded ones
        if columns is not None:
            missing = [column for column in columns if column not in df.columns]
            if missing:
                raise ValueError(f"Columns not found in df: {missing}")
            return sorted(list(set(columns) - set(exclude_columns)))

        numeric_columns = set(df.select_dtypes(include=np.number).columns) - set(exclude_columns)
        return sorted(list(numeric_columns))

    @staticmethod
    def _parse_process_name(raw_name: str):
        """
        Parses a process definition into the desired form.
        '*' marks a step overlay (dashed, scaled overlay) rather than a
        stacked background. ':' separates a family from its sub-process.

        Args:
            raw_name (str): Process Definition

        Returns:
            name (str): Cleaned process name with '*' removed and ':' whitespace normalized.
            family (str): Text before ':' (the family), or the full name if there is no ':'.
            is_step (bool): True if this process should be drawn as a scaled step overlay.
        """
        # remove whitespace and '*' for step overlay, normalize ':' spacing
        is_step = "*" in raw_name
        name = raw_name.replace("*", "").strip()
        if ":" in name:
            group, sub = name.split(":", 1)
            name = f"{group.strip()}:{sub.strip()}"
            family = group.strip()
        else:
            family = name
        return name, family, is_step

    def _build_process_masks(self, process_definitions, include_other):
        """
        Goes thought process definitions of form {'family:NAME*': {pids} } and create pid masks for each process groups.
        Delegates parsing of definitions to _parse_process_name.
        When *include_other* is set, all not used pids are gathered together under 'Other'.
        """
        seen_pids = {}
        self.process_masks = {}
        self.step_process_names = set()
        self.stack_process_names = []

        for raw_name, pids in process_definitions.items():
            name, _, is_step = self._parse_process_name(raw_name)

            # guarantee that a pid is only assigned to one process, and avoids double counting in the histograms
            current_pids = set(pids)
            overlap = seen_pids.keys() & current_pids
            if overlap:
                conflicting_with = {seen_pids[p] for p in overlap}
                raise ValueError(f"PIDs {overlap} in process '{name}' already assigned to {conflicting_with}.")

            # store the mapping of pids to process name for future overlap checks
            seen_pids.update({p: name for p in current_pids})

            # get a AND mask for all processes of current subprocess
            self.process_masks[name] = self.df[self.process_column].isin(current_pids)

            if is_step:
                self.step_process_names.add(name)
            else:
                self.stack_process_names.append(name)

        # every pid that was not covered is classified as other
        # covered is union of all pids seen
        if include_other:
            covered = pd.Series(False, index=self.df.index)
            for mask in self.process_masks.values():
                covered |= mask
            other_mask = ~covered

            if other_mask.any():
                self.process_masks["Other"] = other_mask
                self.stack_process_names.append("Other")

        # store every process in a separate list
        self.process_names = list(self.process_masks.keys())

    @staticmethod
    def _to_rgba(hex_color, alpha):
        """Converts a '#rrggbb' color (or named CSS color) to an rgba(...)
        string with the given alpha, for use as a semi-transparent fillcolor."""
        if hex_color.startswith("#"):
            h = hex_color.lstrip("#")
            r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        else:
            named = {"lightgrey": (211, 211, 211)}
            r, g, b = named.get(hex_color, (128, 128, 128))
        return f"rgba({r},{g},{b},{alpha})"

    def _build_process_colors(self):
        """
        How colors are picked per process. Color override is applied here
        When no override happen a color wheel is used (defaults plotly defaults)

        To get a figure of all colors run: px.colors.qualitative.swatches()
        """
        palette = px.colors.qualitative.Plotly
        self.process_colors = {}

        i = 0
        for name in self.process_names:
            if name in self.process_color_overrides:
                picked_color = self.process_color_overrides[name]
            elif name == "Other":
                picked_color = "lightgrey"
            else:
                picked_color = palette[i % len(palette)]
                i += 1
            self.process_colors[name] = picked_color

    def _build_process_families(self):
        """
        Maps each stack process to its family (text before ':', or the whole name).
        """
        self.process_family = {}
        self.family_members = {}
        for name in self.stack_process_names:
            _, fam, _ = self._parse_process_name(name)
            self.process_family[name] = fam
            self.family_members.setdefault(fam, []).append(name)
        self.family_order = list(dict.fromkeys(self.process_family.values()))

    def _build_figure(self):
        """
        Create Traces for each family member and add them to the figure.
        Using Scatter traces instead of bar plots due to limitation of stacking in bar plots.
        Instead the line plots (which is scatter) fills from 0 to y and to the next member of class.
        """
        # create grid of plots
        n_rows = int(np.ceil(len(self.numeric_cols) / self.n_cols))
        fig = make_subplots(rows=n_rows, cols=self.n_cols, subplot_titles=self.numeric_cols)
        self.trace_index = {}

        # fill plots
        for idx, col in enumerate(self.numeric_cols):
            row = idx // self.n_cols + 1
            c = idx % self.n_cols + 1

            # flat list of all traces inside figure
            trace_pos = len(fig.data)

            # create stacked plots
            # 3 depths are handled: feature -> family -> member
            for fam in reversed(self.family_order):
                members = self.family_members[fam]
                for current_member_index, name in enumerate(members):
                    fill_mode = "tozeroy" if current_member_index == 0 else "tonexty"
                    # a trace is plotly's a dataseries inside a figure
                    fig.add_trace(
                        go.Scatter(
                            x=[],
                            y=[],
                            name=name,
                            mode="lines",
                            line_shape="hv",  # creates bar look like (first horizontal then vertical step)
                            line=dict(color=self.process_colors[name], width=1.5),
                            fill=fill_mode,
                            fillcolor=self._to_rgba(self.process_colors[name], self.opacity_filling),
                            fillpattern=dict(
                                shape=self.family_hatch.get(fam, ""),
                                fgcolor=self.process_colors[name],
                                size=6,
                                solidity=0.3,
                            ),
                            showlegend=(idx == 0),
                            legendgroup=name,
                        ),
                        row=row,
                        col=c,
                    )
                    self.trace_index[(idx, name)] = trace_pos
                    trace_pos += 1

            # create step plots that are not stacked
            for name in self.step_process_names:
                fig.add_trace(
                    go.Scatter(
                        x=[],
                        y=[],
                        name=name,
                        mode="lines",
                        line_shape="hv",
                        line=dict(color=self.process_colors[name], width=2, dash="dash"),
                        showlegend=(idx == 0),
                        legendgroup=name,
                    ),
                    row=row,
                    col=c,
                )
                self.trace_index[(idx, name)] = trace_pos
                trace_pos += 1

        # title of the whole figure
        fig.update_layout(
            height=self.height * n_rows,
            width=self.width * self.n_cols,
            title_text="Feature distributions - cumulative step lines per family",
        )
        self.figure_widget = go.FigureWidget(fig)
        self.base_titles = [ann.text for ann in self.figure_widget.layout.annotations]

    def _combine_masks(self, query_strings):
        """AND-combines multiple pandas query strings into one boolean mask."""
        mask = pd.Series(True, index=self.df.index)
        for q in query_strings:
            if not q.strip():
                continue
            mask &= self.df.eval(q)
        return mask

    # by default ipython widgets req button instance as first argument
    def update_plots(self, _=None):
        """
        Fill the stored traces in Figure with actual data.
        """

        # add queries when button is toggled (value is true)
        active_queries = [self.predefined_cuts[name] for name, btn in self.cut_toggles.items() if btn.value]

        # show active cuts
        if active_queries:
            cuts_html = "<b>Active cuts:</b> " + " AND ".join(f"<code>{q}</code>" for q in active_queries)
        else:
            cuts_html = "<i>No cuts active</i>"
        self.active_cuts_label.value = cuts_html

        # add cut defined in the text field on top
        if self.custom_cut_text.value.strip():
            active_queries.append(self.custom_cut_text.value)

        # combine masks to 1 query or error message
        try:
            global_mask = self._combine_masks(active_queries)
        except Exception as e:
            self.status_label.value = f"⚠️ Invalid cut: {e}"
            return
        self.status_label.value = ""
        n_bins_global = self.bins_slider.value
        show_other = self.show_other_checkbox.value if self.show_other_checkbox is not None else True

        # actual updating
        with self.figure_widget.batch_update():
            for idx, col in enumerate(self.numeric_cols):
                # create binning space, needs to be done every time, for the case that bins change
                lo, hi = self.value_range[col]
                n_bins = self.column_overrides.get(col, {}).get("n_bins", n_bins_global)
                edges = np.linspace(lo, hi, n_bins + 1)
                centers = (edges[:-1] + edges[1:]) / 2

                # statistics, overflow bin
                total_selected, total_outside = 0, 0
                # highest family TOTAL, is used to scale signal to something seeable
                overall_peak = 0.0

                # --- Stacked data ---
                for fam in self.family_order:
                    # keep for each stack (family) a separate cumulative counter
                    cum = np.zeros(n_bins)
                    for name in self.family_members[fam]:  # bottom-to-top: accumulate going up
                        # gets trace information on bin base
                        # example having 10 bins and 5 processes leads to 55 traces
                        # trace index is (bin, process family): index
                        trace = self.figure_widget.data[self.trace_index[(idx, name)]]
                        if name == "Other" and not show_other:
                            trace.visible = False
                            continue

                        # process masks is AND over PIDS, global masks is second cut
                        combined_mask = global_mask & self.process_masks[name]
                        counts, _ = np.histogram(
                            self.df.loc[combined_mask, col],  # get filtered data
                            bins=edges,
                            weights=self.df.loc[combined_mask].get(self.weight_column_name, None),
                        )
                        cum = cum + counts
                        # setting x and y values of the scatter plot
                        trace.x, trace.y = centers, cum.copy()  # cumulative height -> the "stack"
                        trace.visible = True

                        n_sel = int(combined_mask.sum())
                        total_selected += n_sel
                        total_outside += n_sel - int(counts.sum())
                    overall_peak = max(overall_peak, cum.max() if cum.size else 0.0)

                # --- non stacked data ---
                amplifier_notes = []
                for name in self.step_process_names:
                    trace = self.figure_widget.data[self.trace_index[(idx, name)]]
                    combined_mask = global_mask & self.process_masks[name]
                    counts, _ = np.histogram(
                        self.df.loc[combined_mask, col],
                        bins=edges,
                        weights=self.df.loc[combined_mask].get(self.weight_column_name, None),
                    )
                    peak_step = counts.max()
                    amplifier = (overall_peak / peak_step) if peak_step > 0 else 1.0
                    trace.x, trace.y = centers, counts * amplifier
                    trace.name = f"{name} (×{amplifier:.1f})"
                    trace.visible = True
                    amplifier_notes.append(f"{name} ×{amplifier:.1f}")

                amp_suffix = f" | {', '.join(amplifier_notes)}" if amplifier_notes else ""
                self.figure_widget.layout.annotations[idx].text = (
                    f"{self.base_titles[idx]}{amp_suffix}<br>"
                    f"<span style='font-size:10px;color:grey'>outside range: {total_outside} of {total_selected}</span>"
                )

    # --- Widget Design ---
    def _make_filter_toggle(self, name):
        btn = widgets.ToggleButton(value=False, description=name, layout=widgets.Layout(width="auto"))
        btn.observe(self._on_toggle_style, names="value")
        return btn

    def _on_toggle_style(self, change):
        change.owner.button_style = "success" if change["new"] else ""

    def _build_widgets(self):
        self.cut_toggles = {name: self._make_filter_toggle(name) for name in self.predefined_cuts}
        self.custom_cut_text = widgets.Text(value="", description="Custom cut:", layout=widgets.Layout(width="400px"))
        self.bins_slider = widgets.IntSlider(
            value=self.n_bins, min=self.n_bins_range[0], max=self.n_bins_range[1], description="N bins:"
        )
        self.update_button = widgets.Button(description="Update", button_style="primary")
        self.status_label = widgets.Label(value="")

        self.active_cuts_label = widgets.HTML(value="<i> NO cuts active </i>")

        controls_row = [self.bins_slider]
        if "Other" in self.process_names:
            self.show_other_checkbox = widgets.Checkbox(value=True, description="Show 'Other' process")
            controls_row.append(self.show_other_checkbox)
        else:
            self.show_other_checkbox = None

        self.update_button.on_click(self.update_plots)
        # grouping of the widgets
        self.controls = widgets.VBox(
            [
                widgets.HTML("<b>Clickable filters (AND-combined):</b>"),
                widgets.HBox(list(self.cut_toggles.values())),
                self.custom_cut_text,
                widgets.HBox(controls_row),
                self.update_button,
                self.status_label,
                self.active_cuts_label,
            ]
        )

    def show(self):
        display(self.controls, self.figure_widget)
