import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# class HistogramPlotter1D:
#     """Interactive 1D feature histogram plotter with clickable filters, a
#     custom cut, per-column bin/range overrides, outlier protection, a
#     per-process stacked breakdown, and a separately-scaled step overlay for
#     tiny processes (e.g. signal).

#     Args:
#         df: DataFrame containing the event data.
#         predefined_cuts: Name -> pandas query string (AND-combined via toggles).
#         process_column: Column in df holding the process ID (pid) per event.
#         process_definitions: Process name -> iterable of pid values belonging
#             to that process, e.g. {"SIGNAL*": {21101}, "TT": {1100, 1200, 1300}}.
#             A "*" anywhere in the name marks that process as a STEP overlay:
#             drawn unfilled on top of the stack (not part of it) and rescaled
#             to match the stack's peak height, with the scale factor shown in
#             the subplot title. PIDs not covered by any group are collected
#             into an automatic "Other" bucket, which is always a regular
#             stacked process and never eligible for the step treatment.
#         columns: Explicit include-list of columns to plot. If None, all
#             numeric columns of df are used automatically (excluding process_column).
#         n_bins_range: (min, max) for the global bins slider.
#         n_cols: Number of columns in the subplot grid.
#         clip_quantiles: (low, high) percentiles defining the default range.
#         column_overrides: Optional column name -> {"range": (lo, hi), "n_bins": int}.
#         include_other: Whether uncovered pids get collected into "Other".

#     Raises:
#         ValueError: If process_column is missing, pid sets overlap between
#             processes, or `columns` references a missing/non-numeric column.
#     """

#     def __init__(
#         self,
#         df,
#         predefined_cuts,
#         process_column,
#         process_definitions,
#         columns=None,
#         n_bins=30,
#         n_bins_range=(5, 100),
#         n_cols=3,
#         clip_quantiles=(0.01, 0.99),
#         column_overrides=None,
#         include_other=True,
#         weight_column_name: str = "",
#     ):
#         if process_column not in df.columns:
#             raise ValueError(f"process_column '{process_column}' not found in df.")

#         self.df = df
#         self.predefined_cuts = predefined_cuts
#         self.process_column = process_column
#         self.column_overrides = dict(column_overrides or {})
#         self.n_bins_range = n_bins_range
#         self.n_cols = n_cols
#         self.clip_quantiles = clip_quantiles
#         self.weight_column_name = weight_column_name
#         self.n_bins = n_bins
#         self.stack_opacity = 0.8

#         if columns is not None:
#             missing = [c for c in columns if c not in df.columns]
#             if missing:
#                 raise ValueError(f"Columns not found in df: {missing}")
#             non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
#             if non_numeric:
#                 raise ValueError(f"Non-numeric columns cannot be plotted: {non_numeric}")
#             self.numeric_cols = list(columns)
#         else:
#             self.numeric_cols = [c for c in df.select_dtypes(include=np.number).columns if c != process_column]

#         self.value_range = {}
#         for col in self.numeric_cols:
#             override_range = self.column_overrides.get(col, {}).get("range")
#             self.value_range[col] = (
#                 override_range if override_range is not None else tuple(df[col].quantile(clip_quantiles))
#             )

#         self._build_process_masks(process_definitions, include_other)
#         self._build_process_colors()
#         self._build_process_families()
#         self._build_figure()
#         self._build_widgets()
#         self.update_plots()

#     def _build_process_families(self):
#         """Maps each stack process to its family (text before ':', or the
#         whole name). Also stores, per family, the bottom-to-top member order -
#         needed for the cumulative fake-stack calculation."""
#         self.process_family = {}
#         self.family_members = {}
#         for name in self.stack_process_names:
#             fam = name.split(":", 1)[0] if ":" in name else name
#             self.process_family[name] = fam
#             self.family_members.setdefault(fam, []).append(name)
#         self.family_order = list(dict.fromkeys(self.process_family.values()))

#     def _build_process_masks(self, process_definitions, include_other):
#         """
#         Validates process definitions, merges "group:sub" keys into one
#         process per group, splits the result into stacked vs. step (a "*"
#         anywhere in the key marks a step overlay), and computes one boolean
#         mask per process."""
#         seen_pids = {}
#         grouped_pids = {}  # merged group name -> union of pid sets
#         group_order = []  # first-seen order of group names
#         step_group_names = set()

#         for raw_name, pids in process_definitions.items():
#             is_step = "*" in raw_name
#             cleaned = raw_name.replace("*", "").strip()
#             group_name = cleaned.split(":", 1)[0].strip() if ":" in cleaned else cleaned

#             pid_set = set(pids)
#             overlap = seen_pids.keys() & pid_set
#             if overlap:
#                 conflicting_with = {seen_pids[p] for p in overlap}
#                 raise ValueError(
#                     f"PIDs {overlap} in '{raw_name}' (group '{group_name}') already assigned to {conflicting_with}."
#                 )
#             seen_pids.update({p: group_name for p in pid_set})

#             if group_name not in grouped_pids:
#                 grouped_pids[group_name] = set()
#                 group_order.append(group_name)
#             grouped_pids[group_name] |= pid_set
#             if is_step:
#                 step_group_names.add(group_name)

#         self.process_masks = {name: self.df[self.process_column].isin(pids) for name, pids in grouped_pids.items()}
#         self.step_process_names = step_group_names
#         self.stack_process_names = [name for name in group_order if name not in step_group_names]

#         if include_other:
#             covered = pd.Series(False, index=self.df.index)
#             for mask in self.process_masks.values():
#                 covered |= mask
#             other_mask = ~covered
#             if other_mask.any():
#                 self.process_masks["Other"] = other_mask
#                 self.stack_process_names.append("Other")  # "Other" is NEVER a step process, by construction

#         self.process_names = list(self.process_masks.keys())  # display/legend order

#     def _build_process_colors(self):
#         # Assigns a color to each process name, cycling through Plotly's qualitative palette.
#         palette = px.colors.qualitative.Plotly
#         self.process_colors = {}
#         i = 0
#         for name in self.process_names:
#             if name == "Other":
#                 self.process_colors[name] = "lightgrey"
#             else:
#                 self.process_colors[name] = palette[i % len(palette)]
#                 i += 1

#     def _build_figure(self):
#         # actual plot building happens here.
#         n_rows = int(np.ceil(len(self.numeric_cols) / self.n_cols))
#         fig = make_subplots(rows=n_rows, cols=self.n_cols, subplot_titles=self.numeric_cols)
#         self.trace_index = {}  # (feature_idx, process_name) -> position in fig.data

#         for idx, col in enumerate(self.numeric_cols):
#             row, c = idx // self.n_cols + 1, idx % self.n_cols + 1
#             trace_pos = len(fig.data)
#             for fam in self.family_order:
#                 for name in reversed(self.family_members[fam]):
#                     fig.add_trace(
#                         go.Bar(
#                             x=[],
#                             y=[],
#                             name=name,
#                             marker_color=self.process_colors[name],
#                             opacity=self.stack_opacity,
#                             showlegend=(idx == 0),
#                             legendgroup=name,
#                         ),
#                         row=row,
#                         col=c,
#                     )
#                     self.trace_index[(idx, name)] = trace_pos
#                     trace_pos += 1

#             # Scatter, not Bar -> never participates in barmode="stack",
#             # regardless of how many Bar traces share this subplot
#             for name in self.step_process_names:
#                 fig.add_trace(
#                     go.Scatter(
#                         x=[],
#                         y=[],
#                         name=name,
#                         mode="lines",
#                         line_shape="hv",
#                         line=dict(color=self.process_colors[name], width=2),
#                         showlegend=(idx == 0),
#                         legendgroup=name,
#                     ),
#                     row=row,
#                     col=c,
#                 )
#                 self.trace_index[(idx, name)] = trace_pos
#                 trace_pos += 1

#         # define plot grid
#         fig.update_layout(
#             height=500 * n_rows,
#             width=550 * self.n_cols,
#             barmode="overlay",
#             bargap=0.0,
#             title_text="Feature distributions stacked by process",
#         )
#         self.figure_widget = go.FigureWidget(fig)
#         self.base_titles = [ann.text for ann in self.figure_widget.layout.annotations]

#     def _make_filter_toggle(self, name):
#         btn = widgets.ToggleButton(value=False, description=name, layout=widgets.Layout(width="auto"))
#         btn.observe(self._on_toggle_style, names="value")
#         return btn

#     def _on_toggle_style(self, change):
#         change.owner.button_style = "success" if change["new"] else ""

#     def _build_widgets(self):
#         # create all widgets, but don't yet connect them to the update_plots() logic
#         self.cut_toggles = {name: self._make_filter_toggle(name) for name in self.predefined_cuts}
#         self.custom_cut_text = widgets.Text(value="", description="Custom cut:", layout=widgets.Layout(width="400px"))
#         self.bins_slider = widgets.IntSlider(
#             value=self.n_bins, min=self.n_bins_range[0], max=self.n_bins_range[1], description="N bins:"
#         )
#         self.update_button = widgets.Button(description="Update", button_style="primary")
#         self.status_label = widgets.Label(value="")

#         controls_row = [self.bins_slider]
#         if "Other" in self.process_names:
#             self.show_other_checkbox = widgets.Checkbox(value=True, description="Show 'Other' process")
#             controls_row.append(self.show_other_checkbox)
#         else:
#             self.show_other_checkbox = None

#         self.update_button.on_click(self.update_plots)

#         self.controls = widgets.VBox(
#             [
#                 widgets.HTML("<b>Clickable filters (AND-combined):</b>"),
#                 widgets.HBox(list(self.cut_toggles.values())),
#                 self.custom_cut_text,
#                 widgets.HBox(controls_row),
#                 self.update_button,
#                 self.status_label,
#             ]
#         )

#     def _combine_masks(self, query_strings):
#         """AND-combines multiple pandas query strings into one boolean mask."""
#         mask = pd.Series(True, index=self.df.index)
#         for q in query_strings:
#             if not q.strip():
#                 continue
#             mask &= self.df.eval(q)
#         return mask

#     def update_plots(self, _=None):
#         active_queries = [self.predefined_cuts[name] for name, btn in self.cut_toggles.items() if btn.value]
#         if self.custom_cut_text.value.strip():
#             active_queries.append(self.custom_cut_text.value)
#         try:
#             global_mask = self._combine_masks(active_queries)
#         except Exception as e:
#             self.status_label.value = f"⚠️ Invalid cut: {e}"
#             return
#         self.status_label.value = ""

#         n_bins_global = self.bins_slider.value
#         show_other = self.show_other_checkbox.value if self.show_other_checkbox is not None else True

#         with self.figure_widget.batch_update():
#             for idx, col in enumerate(self.numeric_cols):
#                 lo, hi = self.value_range[col]
#                 n_bins = self.column_overrides.get(col, {}).get("n_bins", n_bins_global)
#                 edges = np.linspace(lo, hi, n_bins + 1)
#                 centers = (edges[:-1] + edges[1:]) / 2

#                 total_selected, total_outside = 0, 0
#                 overall_peak = 0.0  # highest family TOTAL across all families -> signal scale reference

#                 for fam in self.family_order:
#                     cum = np.zeros(n_bins)
#                     for name in self.family_members[fam]:  # bottom-to-top: accumulate going up
#                         trace = self.figure_widget.data[self.trace_index[(idx, name)]]
#                         if name == "Other" and not show_other:
#                             trace.visible = False
#                             continue
#                         combined_mask = global_mask & self.process_masks[name]
#                         counts, _ = np.histogram(
#                             self.df.loc[combined_mask, col],
#                             bins=edges,
#                             weights=self.df.loc[combined_mask].get(self.weight_column_name, None),
#                         )
#                         cum = cum + counts
#                         trace.x, trace.y = centers, cum.copy()  # cumulative, not individual
#                         trace.visible = True
#                         n_sel = int(combined_mask.sum())
#                         total_selected += n_sel
#                         total_outside += n_sel - int(counts.sum())
#                     overall_peak = max(overall_peak, cum.max() if cum.size else 0.0)

#                 amplifier_notes = []
#                 for name in self.step_process_names:
#                     trace = self.figure_widget.data[self.trace_index[(idx, name)]]
#                     combined_mask = global_mask & self.process_masks[name]
#                     counts, _ = np.histogram(
#                         self.df.loc[combined_mask, col],
#                         bins=edges,
#                         weights=self.df.loc[combined_mask].get(self.weight_column_name, None),
#                     )
#                     peak_step = counts.max()
#                     amplifier = (overall_peak / peak_step) if peak_step > 0 else 1.0
#                     trace.x, trace.y = centers, counts * amplifier
#                     trace.name = f"{name} (×{amplifier:.1f})"
#                     amplifier_notes.append(f"{name} ×{amplifier:.1f}")

#                 amp_suffix = f" | {', '.join(amplifier_notes)}" if amplifier_notes else ""
#                 self.figure_widget.layout.annotations[idx].text = (
#                     f"{self.base_titles[idx]}{amp_suffix}<br>"
#                     f"<span style='font-size:10px;color:grey'>outside range: {total_outside} of {total_selected}</span>"
#                 )

#     def show(self):
#         display(self.controls, self.figure_widget)


class HistogramPlotter1D:
    """1D histogram plotter. Families of processes ('family:sub' naming)
    are rendered as CUMULATIVE STEP LINES (not filled bars) - lines never
    occlude each other regardless of overlap, so every sub-process AND every
    family's total remains simultaneously visible and comparable."""

    def __init__(
        self,
        df,
        predefined_cuts,
        process_column,
        process_definitions,
        columns=None,
        n_bins=30,
        n_bins_range=(5, 100),
        n_cols=3,
        clip_quantiles=(0.01, 0.99),
        column_overrides=None,
        include_other=True,
        weight_column_name="",
    ):
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

        self._build_process_masks(process_definitions, include_other)
        self._build_process_colors()
        self._build_process_families()
        self._build_figure()
        self._build_widgets()
        self.update_plots()

    def _build_process_masks(self, process_definitions, include_other):
        """'*' marks a step overlay. Every other name is kept exactly as
        given - no merging, ':' has no meaning here (only used later for
        family grouping)."""
        seen_pids = {}
        self.process_masks = {}
        self.step_process_names = set()
        self.stack_process_names = []
        for raw_name, pids in process_definitions.items():
            is_step = "*" in raw_name
            name = raw_name.replace("*", "").strip()
            if ":" in name:
                group, sub = name.split(":", 1)
                name = f"{group.strip()}:{sub.strip()}"  # normalize whitespace around ':'
            pid_set = set(pids)
            overlap = seen_pids.keys() & pid_set
            if overlap:
                conflicting_with = {seen_pids[p] for p in overlap}
                raise ValueError(f"PIDs {overlap} in process '{name}' already assigned to {conflicting_with}.")
            seen_pids.update({p: name for p in pid_set})
            self.process_masks[name] = self.df[self.process_column].isin(pid_set)
            if is_step:
                self.step_process_names.add(name)
            else:
                self.stack_process_names.append(name)
        if include_other:
            covered = pd.Series(False, index=self.df.index)
            for mask in self.process_masks.values():
                covered |= mask
            other_mask = ~covered
            if other_mask.any():
                self.process_masks["Other"] = other_mask
                self.stack_process_names.append("Other")
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
        """Every process gets its own independent color - no family relation."""
        palette = px.colors.qualitative.Plotly
        self.process_colors = {}
        i = 0
        for name in self.process_names:
            if name == "Other":
                self.process_colors[name] = "lightgrey"
            else:
                self.process_colors[name] = palette[i % len(palette)]
                i += 1

    def _build_process_families(self):
        """Maps each stack process to its family (text before ':', or the
        whole name). Stores, per family, the bottom-to-top member order -
        needed for the cumulative sum calculation."""
        self.process_family = {}
        self.family_members = {}
        for name in self.stack_process_names:
            fam = name.split(":", 1)[0] if ":" in name else name
            self.process_family[name] = fam
            self.family_members.setdefault(fam, []).append(name)
        self.family_order = list(dict.fromkeys(self.process_family.values()))

    def _build_figure(self):
        n_rows = int(np.ceil(len(self.numeric_cols) / self.n_cols))
        fig = make_subplots(rows=n_rows, cols=self.n_cols, subplot_titles=self.numeric_cols)
        self.trace_index = {}

        for idx, col in enumerate(self.numeric_cols):
            row, c = idx // self.n_cols + 1, idx % self.n_cols + 1
            trace_pos = len(fig.data)

            for fam in reversed(self.family_order):
                members = self.family_members[fam]
                for i, name in enumerate(members):
                    fill_mode = "tozeroy" if i == 0 else "tonexty"
                    fig.add_trace(
                        go.Scatter(
                            x=[], y=[], name=name, mode="lines", line_shape="hv",
                            line=dict(color=self.process_colors[name], width=1.5),
                            fill=fill_mode,
                            fillcolor=self._to_rgba(self.process_colors[name], 0.5),
                            showlegend=(idx == 0), legendgroup=name,
                        ),
                        row=row, col=c,
                    )
                    self.trace_index[(idx, name)] = trace_pos
                    trace_pos += 1

            for name in self.step_process_names:
                fig.add_trace(
                    go.Scatter(
                        x=[], y=[], name=name, mode="lines", line_shape="hv",
                        line=dict(color=self.process_colors[name], width=2, dash="dash"),
                        showlegend=(idx == 0), legendgroup=name,
                    ),
                    row=row, col=c,
                )
                self.trace_index[(idx, name)] = trace_pos
                trace_pos += 1

        fig.update_layout(
            height=300 * n_rows, width=350 * self.n_cols,
            title_text="Feature distributions - cumulative step lines per family",
        )
        self.figure_widget = go.FigureWidget(fig)
        self.base_titles = [ann.text for ann in self.figure_widget.layout.annotations]

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

        controls_row = [self.bins_slider]
        if "Other" in self.process_names:
            self.show_other_checkbox = widgets.Checkbox(value=True, description="Show 'Other' process")
            controls_row.append(self.show_other_checkbox)
        else:
            self.show_other_checkbox = None

        self.update_button.on_click(self.update_plots)
        self.controls = widgets.VBox(
            [
                widgets.HTML("<b>Clickable filters (AND-combined):</b>"),
                widgets.HBox(list(self.cut_toggles.values())),
                self.custom_cut_text,
                widgets.HBox(controls_row),
                self.update_button,
                self.status_label,
            ]
        )

    def _combine_masks(self, query_strings):
        """AND-combines multiple pandas query strings into one boolean mask."""
        mask = pd.Series(True, index=self.df.index)
        for q in query_strings:
            if not q.strip():
                continue
            mask &= self.df.eval(q)
        return mask

    def update_plots(self, _=None):
        active_queries = [self.predefined_cuts[name] for name, btn in self.cut_toggles.items() if btn.value]
        if self.custom_cut_text.value.strip():
            active_queries.append(self.custom_cut_text.value)
        try:
            global_mask = self._combine_masks(active_queries)
        except Exception as e:
            self.status_label.value = f"⚠️ Invalid cut: {e}"
            return
        self.status_label.value = ""

        n_bins_global = self.bins_slider.value
        show_other = self.show_other_checkbox.value if self.show_other_checkbox is not None else True

        with self.figure_widget.batch_update():
            for idx, col in enumerate(self.numeric_cols):
                lo, hi = self.value_range[col]
                n_bins = self.column_overrides.get(col, {}).get("n_bins", n_bins_global)
                edges = np.linspace(lo, hi, n_bins + 1)
                centers = (edges[:-1] + edges[1:]) / 2

                total_selected, total_outside = 0, 0
                overall_peak = 0.0  # highest family TOTAL -> signal-scaling reference

                for fam in self.family_order:
                    cum = np.zeros(n_bins)
                    for name in self.family_members[fam]:  # bottom-to-top: accumulate going up
                        trace = self.figure_widget.data[self.trace_index[(idx, name)]]
                        if name == "Other" and not show_other:
                            trace.visible = False
                            continue
                        combined_mask = global_mask & self.process_masks[name]
                        counts, _ = np.histogram(
                            self.df.loc[combined_mask, col],
                            bins=edges,
                            weights=self.df.loc[combined_mask].get(self.weight_column_name, None),
                        )
                        cum = cum + counts
                        trace.x, trace.y = centers, cum.copy()  # cumulative height -> the "stack"
                        trace.visible = True
                        n_sel = int(combined_mask.sum())
                        total_selected += n_sel
                        total_outside += n_sel - int(counts.sum())
                    overall_peak = max(overall_peak, cum.max() if cum.size else 0.0)

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

    def show(self):
        display(self.controls, self.figure_widget)
