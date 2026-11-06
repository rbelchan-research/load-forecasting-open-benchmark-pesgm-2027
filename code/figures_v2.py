"""Figures for the PES GM 2027 paper.

Design rules followed here, in order:
  form first, then color, then marks, then labels
  categorical hues come from a validated five-slot palette and are never cycled
  every legend sits outside the plot box, above the axes, so nothing overlaps a mark
  one measure per axis, recessive grid, thin marks, selective direct labels
  every series carries a marker and a dash pattern so the figure survives grayscale
IEEE single column is 3.4 in wide; stacked panels keep that width.
"""
from __future__ import annotations
import json, os
import paths
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# validated categorical slots, light surface (see dataviz validator, adjacent pairlist)
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
COL = [S1, S2, S3, S4, S5]
MRK = ["o", "s", "^", "D", "v"]
DSH = [(None, None), (4.5, 1.6), (1.4, 1.3), (6, 1.6, 1.4, 1.6), (3, 1.3, 1, 1.3)]
INK, INK2, INK3 = "#16171a", "#4e5157", "#787c84"
RULE = "#9a9a93"
W1, W2 = 3.4, 7.16
OUT = paths.FIGURES


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": INK3, "ytick.color": INK3,
        "axes.edgecolor": "#b5b5ae", "axes.linewidth": 0.6,
        "grid.linewidth": 0.4, "lines.linewidth": 1.5,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "grid.color": "#dededa", "grid.alpha": 1.0,
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.9, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.1, "legend.borderpad": 0.0, "legend.labelspacing": 0.35,
        "figure.dpi": 300, "savefig.dpi": 800,   # 600+ ppi at print size, the IEEE line-art guide
        "savefig.bbox": "tight", "savefig.pad_inches": 0.005,
    })



def overlaps(fig, pad=0.2):
    """Check every label in a figure, in display coordinates, and return each clash found.

    Checked: text against text (titles, axis labels, visible tick labels, legend entries,
    annotations), text against plotted lines and their markers, text against legend keys,
    text against arrows, and text against box or bar edges. A label drawn on an opaque
    white patch is allowed to sit over a line, because the line stops at the patch. Grid
    lines and axis spines are ignored."""
    from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    texts = []

    def visible_ticks(axis):
        lo, hi = sorted(axis.get_view_interval())
        return [t.label1 for t in axis.get_major_ticks()
                if lo - 1e-9 <= t.get_loc() <= hi + 1e-9 and t.label1.get_visible()]

    for ax in fig.axes:
        cand = [(t, "label") for t in ax.texts]
        cand += [(t, "label") for t in (ax.title, ax._left_title, ax._right_title,
                                        ax.xaxis.label, ax.yaxis.label)]
        cand += [(t, "tick") for t in visible_ticks(ax.xaxis) + visible_ticks(ax.yaxis)]
        leg = ax.get_legend()
        if leg:
            cand += [(t, "legend") for t in leg.get_texts()]
        for t, kind in cand:
            if t.get_visible() and t.get_text().strip():
                bb = t.get_window_extent(r).padded(-pad * px)
                shielded = t.get_bbox_patch() is not None and t.get_bbox_patch().get_facecolor()[3] > 0.9
                texts.append((t.get_text().replace("\n", " "), bb, shielded, t))
    out = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if texts[i][1].overlaps(texts[j][1]):
                out.append(f"text '{texts[i][0]}' overlaps text '{texts[j][0]}'")

    def through(pts, bb):
        return bool(((pts[:, 0] > bb.x0) & (pts[:, 0] < bb.x1) &
                     (pts[:, 1] > bb.y0) & (pts[:, 1] < bb.y1)).any())

    for ax in fig.axes:
        for ln in ax.lines:
            xy = ax.transData.transform(ln.get_xydata())
            if ln.get_linestyle() in ("None", "", " ") or len(xy) < 2:
                pts = xy
            else:
                pts = np.vstack([np.linspace(a, b, max(2, int(np.hypot(*(b - a))))) for a, b in zip(xy[:-1], xy[1:])])
            ms = ln.get_markersize() * px / 2 if ln.get_marker() not in (None, "None", "", " ") else 0
            for name, bb, shielded, _ in texts:
                hit = (not shielded) and through(pts, bb)
                if not hit and ms:
                    hit = any(x + ms > bb.x0 and x - ms < bb.x1 and y + ms > bb.y0 and y - ms < bb.y1 for x, y in xy)
                if hit:
                    out.append(f"line '{ln.get_label()}' runs through text '{name}'")
        leg = ax.get_legend()
        if leg:
            for h in leg.get_lines():
                hb = h.get_window_extent(r)
                half = max(h.get_linewidth(), h.get_markersize()) * px / 2
                hb = type(hb).from_extents(hb.x0, hb.y0 - half, hb.x1, hb.y1 + half)
                for name, bb, _, t in texts:
                    if t not in leg.get_texts() and hb.overlaps(bb):
                        out.append(f"legend key runs into text '{name}'")
            for h in leg.get_patches():
                hb = h.get_window_extent(r)
                for name, bb, _, t in texts:
                    if hb.overlaps(bb):
                        out.append(f"legend key runs into text '{name}'")
        for patch in list(ax.patches) + [c for c in ax.get_children() if isinstance(c, FancyArrowPatch)]:
            pb = patch.get_window_extent(r)
            for name, bb, _, t in texts:
                if not pb.overlaps(bb):
                    continue
                inside = pb.x0 + 1 * px <= bb.x0 and bb.x1 <= pb.x1 - 1 * px and pb.y0 + 1 * px <= bb.y0 and bb.y1 <= pb.y1 - 1 * px
                if isinstance(patch, FancyBboxPatch) and inside:
                    continue          # text placed inside its own box, clear of the edges
                kind = "arrow" if isinstance(patch, FancyArrowPatch) else ("box edge" if isinstance(patch, FancyBboxPatch) else "bar")
                out.append(f"{kind} runs into text '{name}'")
    for t in fig.texts:
        pass
    return sorted(set(out))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    clash = overlaps(fig)
    for c in clash:
        print(f"  [{name}] {c}")
    fig.savefig(f"{OUT}/{name}.png")
    fig.savefig(f"{OUT}/{name}.pdf")
    plt.close(fig)
    print("wrote", name)


def load():
    rows = [json.loads(l) for l in open(f"{paths.RESULTS}/results.jsonl")]
    d = pd.DataFrame(rows)
    d["tag"] = d.tag.fillna(d.strategy)
    return d


def head(ax, title, handles=None, labels=None, ncol=1):
    """Title on the top line, legend on the line below it, plot below that.

    Both sit outside the plot box, so no legend ever covers a mark. The title pad is
    computed from the number of legend rows rather than guessed, which is what kept
    the two colliding before.
    """
    rows = 0
    if handles:
        rows = int(np.ceil(len(handles) / ncol))
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.005, 1.0),
                  ncol=ncol, frameon=False, borderaxespad=0.0,
                  labelcolor=INK2, fontsize=6.7, handlelength=1.8,
                  handletextpad=0.45, columnspacing=1.0, labelspacing=0.3)
    ax.set_title(title, loc="left", fontsize=7.8, color=INK,
                 pad=2.5 + rows * 9.2)


def key(color, marker, dash, label):
    return Line2D([], [], color=color, marker=marker, markersize=3.4,
                  linewidth=1.5, dashes=dash if dash[0] else (None, None),
                  markeredgecolor="white", markeredgewidth=0.6, label=label)


def series(ax, x, y, i, label, ms=3.6, lw=1.5, zorder=3):
    ax.plot(x, y, color=COL[i], marker=MRK[i], markersize=ms, linewidth=lw,
            dashes=DSH[i] if DSH[i][0] else (None, None),
            markeredgecolor="white", markeredgewidth=0.6, label=label,
            solid_capstyle="round", zorder=zorder)


# --------------------------------------------------------------- Fig. 1 pipeline
def fig_pipeline():
    """Two parallel tracks, one for the load and one for the network, merge at the
    transformer panel, split again into the two things a forecast needs, and merge once
    more at fitting and scoring.

    Every connector is a straight vertical run. The earlier version routed elbows into a
    centre point, which produced a short horizontal jog next to each box that read as a
    step rather than a flow. Because the centred boxes already span both lane axes, the
    arrows can simply drop, which is both cleaner and easier to follow.
    """
    style()
    fig, ax = plt.subplots(figsize=(W1, 2.7))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(-1, 101); ax.grid(False)

    FILL_IN, FILL_MID, FILL_OUT = "#eaf1f9", "#fdf0e4", "#e9f5f0"
    EDGE, ARROW = "#4a4b50", "#5c5d62"

    def box(x, y, w, h, text, fc="#ffffff", fs=6.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=1.6",
                     fc=fc, ec=EDGE, lw=0.75, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4,
                fontweight="semibold" if bold else "normal", linespacing=1.12)

    def drop(x, y_top, y_bot, label=None, side=1):
        """A straight vertical connector from the bottom of one box to the top of the
        next, with the label set beside it rather than on it."""
        ax.add_patch(FancyArrowPatch((x, y_top), (x, y_bot), arrowstyle="-|>",
                     mutation_scale=7.5, lw=0.85, color=ARROW,
                     shrinkA=0, shrinkB=0.5, zorder=1))
        if label:
            ax.text(x + 1.8 * side, (y_top + y_bot) / 2, label, fontsize=6.2,
                    color=INK3, ha="left" if side > 0 else "right", va="center")

    # two lanes, one centred column; boxes sized to their text, gaps even
    LX, LW, RX, RW, CX, CW = 0, 46, 54, 46, 10, 80
    # the three-line input boxes need the extra height or their text touches the border
    H3, H2 = 16.5, 11.0
    r1, r2, r3, r4, r5, r6 = 84.5, 69.6, 52.2, 34.8, 17.4, 0.0
    lc, rc, cc_ = LX + LW / 2, RX + RW / 2, CX + CW / 2

    box(LX, r1, LW, H3, "Customer meter data\nhourly kW per\nservice point", FILL_IN)
    box(RX, r1, RW, H3, "Network model\ntransformer, feeder,\nnameplate, class", FILL_IN)
    box(LX, r2, LW, H2, "Sum to the\nservice transformer")
    box(RX, r2, RW, H2, "Calibrate kW to kVA\non a year of power flow")
    box(CX, r3, CW, H2, "Transformer panel\nhourly apparent power", FILL_MID, bold=True)
    box(LX, r4, LW, H2, "Grouping\nshape, attributes, feeder")
    box(RX, r4, RW, H2, "Day-ahead features\nlags, weather, calendar")
    box(CX, r5, CW, H2, "Fit the forecaster\nmean or upper quantile")
    box(CX, r6, CW, H2, "Score\nerror, peak bias, overload screening", FILL_OUT)

    drop(lc, r1, r2 + H2)
    drop(rc, r1, r2 + H2)
    drop(lc, r2, r3 + H2, "kW", side=1)
    drop(rc, r2, r3 + H2, "kVA per kW", side=-1)
    drop(lc, r3, r4 + H2)
    drop(rc, r3, r4 + H2)
    drop(lc, r4, r5 + H2, "group", side=1)
    drop(rc, r4, r5 + H2, "features", side=-1)
    drop(cc_, r5, r6 + H2, "24 h ahead", side=1)

    save(fig, "fig1_pipeline")


# ---------------------------------------------------------- Fig. 2 kVA construction
def fig_kva():
    style()
    v = pd.read_csv(f"{paths.RESULTS}/kva_validation.csv")
    faq = json.load(open(f"{paths.RESULTS}/faq.json"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.3))

    # (a) counts found by each construction, against the power-flow truth
    labs = ["Summed\ncustomer kW", "Flat measured\nfactor", "Calibrated\nmap"]
    keys = ["summed kW", "flat factor", "fitted"]
    x = np.arange(3); w = 0.36
    for i, reg in enumerate(["rural", "urban"]):
        sub = v[v.region == reg].set_index("variant")
        vals = [sub.loc[k, "n_over_100"] for k in keys]
        truth = sub.loc["fitted", "n_over_100_true"]
        bars = a1.bar(x + (i - 0.5) * w, vals, w * 0.9, color=COL[i],
                      edgecolor="white", linewidth=0.8, zorder=3)
        for rect, val in zip(bars, vals):
            a1.annotate(str(val), (rect.get_x() + rect.get_width() / 2, val),
                        xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.3, color=INK, zorder=5,
                        bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
        # one reference rule per substation, labelled at the right edge
        a1.axhline(truth, color=RULE, lw=0.9, ls=(0, (2.4, 1.6)), zorder=1)
        # rural (3) and urban (5) rules sit close together: put the lower label under its
        # rule and the upper one above, so the two never print on top of each other
        a1.annotate(f"truth {truth}", (2.62, truth), fontsize=6.4, color=INK3,
                    va="top" if i == 0 else "bottom", ha="left", annotation_clip=False)
    a1.set_xticks(x); a1.set_xticklabels(labs, fontsize=6.6)
    a1.set_ylabel("Units above\nnameplate", fontsize=7.2)
    a1.set_ylim(0, 28.5)   # headroom keeps the 23 label below the legend row
    a1.set_xlim(-0.52, 2.58)
    head(a1, "(a) What the loading screen finds",
         [Patch(facecolor=COL[0], edgecolor="none"),
          Patch(facecolor=COL[1], edgecolor="none"),
          Line2D([], [], color=RULE, lw=1.0, ls=(0, (2.4, 1.6)))],
         ["rural substation", "urban substation", "power flow truth"], ncol=3)

    # (b) measured ratio against loading level
    lm = faq["load_model_test"]
    qs = list(lm["rural"]["ratio_by_loading"].keys())
    xx = np.arange(len(qs))
    for i, reg in enumerate(["rural", "urban"]):
        series(a2, xx, list(lm[reg]["ratio_by_loading"].values()), i, f"{reg} substation")
    a2.set_xticks(xx)
    a2.set_xticklabels(["lightest", "2nd", "3rd", "4th", "heaviest"], fontsize=6.6)
    a2.set_xlabel("Loading quintile of the hour", labelpad=1.5)
    a2.set_ylabel("Measured kVA per kW", fontsize=7.2)
    a2.set_xlim(-0.3, len(qs) + 0.45)
    # the two lines converge at the heavy end, so the end labels are pulled apart
    ends = {reg: list(lm[reg]["ratio_by_loading"].values())[-1]
            for reg in ("rural", "urban")}
    off = 0.013
    for i, reg in enumerate(["rural", "urban"]):
        yv = ends[reg] + (off if reg == "rural" else -off)
        a2.annotate(f"{reg}", (len(qs) - 0.88, yv), fontsize=6.6, color=COL[i],
                    va="center", ha="left", annotation_clip=False)
    allv = [x for reg in ("rural", "urban")
            for x in lm[reg]["ratio_by_loading"].values()]
    a2.set_ylim(min(allv) - off * 1.9, max(allv) + off * 0.6)
    head(a2, "(b) The ratio is not a constant")
    fig.tight_layout(pad=0.2, h_pad=1.1)
    save(fig, "fig2_kva")


# ------------------------------------------------- Fig. 3 grouping and coverage
GN = {"shape_kmeans": "k-means, week", "daily_kmeans": "k-means, day",
      "feature_kmeans": "k-means, features", "ward_corr": "Ward, correlation",
      "dtw_kmedoids": "DTW k-medoids", "attribute": "attributes only \u2020",
      "feeder": "same feeder \u2020"}
CN = {"local_oracle": "its own meter history", "cold_global": "fleet model, cold start",
      "topdown_attr": "top down, attribute groups", "class_profile": "borrowed profile, whole fleet",
      "donor_shape": "borrowed profile, shape groups"}


def fig_group_cov_wide(d):
    """Same two panels side by side at full text width. This version is meant to sit in
    the one-column band that Table II already opens, so the band carries the table and
    the figure together instead of leaving the foot of the page empty."""
    style()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W2, 1.72))
    _panel_a(a1, d, short=True); _panel_b(a2, d)
    fig.tight_layout(pad=0.2, w_pad=3.0)
    save(fig, "fig3_wide")


def spread(ends, gap):
    """Nudge end-label heights apart by at least `gap` data units, keeping their order,
    so close line ends (cold start vs top down at 70% coverage) do not print on top of
    each other. Returns (y_label, label, color) with the original y kept for the line."""
    ends = sorted(ends)
    ys = [e[0] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    shift = (sum(e[0] for e in ends) - sum(ys)) / len(ys)
    ys = [y + shift for y in ys]
    for i in range(1, len(ys)):            # re-check after centering
        ys[i] = max(ys[i], ys[i - 1] + gap)
    return [(y, e[1], e[2]) for y, e in zip(ys, ends)]


def _panel_a(a1, d, short=False):
    g = d[(d.experiment == "grouping") & (d.strategy == "cluster")].copy()
    g["pct"] = 100 * g.nmae_med
    order = ["shape_kmeans", "daily_kmeans", "feature_kmeans", "ward_corr",
             "dtw_kmedoids", "attribute", "feeder"]
    lo, hi = g.pct.min(), g.pct.max()
    for row, kind in enumerate(order):
        y = len(order) - 1 - row
        sb = g[g.grouping == kind].sort_values("k")
        a1.plot([lo, hi], [y, y], color="#e3e3df", lw=0.7, zorder=1)
        a1.plot(sb.pct, [y] * len(sb), "o", ms=3.4, color=COL[0], alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([GN[k] for k in order][::-1], fontsize=6.0)
    a1.set_ylim(-0.6, len(order) - 0.4)
    a1.set_xlim(lo - 0.06, hi + 0.06)
    a1.set_xlabel("Normalized MAE (%)", fontsize=7.2, labelpad=1.5)
    a1.grid(axis="y", visible=False)
    head(a1, (f"(a) All seventeen settings inside {hi - lo:.1f} points" if short
              else f"(a) Seventeen grouping settings, all inside {hi - lo:.1f} points"))


def _panel_b(a2, d):
    c = d[d.experiment == "coverage"]
    tiers = {"ami15": 15, "ami35": 35, "ami70": 70}
    ends = []
    for i, (tag, lab) in enumerate(CN.items()):
        sub = c[c.tag == tag]
        xs = np.array([tiers[t] for t in sub.tier]); o = np.argsort(xs)
        ys = (100 * sub.nmae_med.to_numpy())[o]
        series(a2, xs[o], ys, i, lab, ms=3.0, lw=1.3)
        ends.append((ys[-1], lab, COL[i]))
    for yv, lab, cc in spread(ends, 2.3):
        a2.annotate(lab, (72.5, yv), fontsize=5.9, color=INK2, va="center",
                    ha="left", annotation_clip=False)
    a2.set_xticks([15, 35, 70])
    a2.set_xlabel("Metering coverage of the fleet (%)", labelpad=1.5)
    a2.set_ylabel("nMAE (%)", fontsize=7.2)
    a2.set_xlim(11, 71)
    head(a2, "(b) Forecasting a transformer that has no meter")


def fig_group_cov(d):
    """Panel (a) is a dot plot, not a line chart. Seven line series whose whole point
    is that they coincide is the wrong form: it produces spaghetti and forces a legend
    on top of the marks. One row per method, one dot per group count, says it directly."""
    style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.3),
                                 gridspec_kw={"height_ratios": [0.92, 1.0]})

    g = d[(d.experiment == "grouping") & (d.strategy == "cluster")].copy()
    g["pct"] = 100 * g.nmae_med
    order = ["shape_kmeans", "daily_kmeans", "feature_kmeans", "ward_corr",
             "dtw_kmedoids", "attribute", "feeder"]
    lo, hi = g.pct.min(), g.pct.max()
    for row, kind in enumerate(order):
        y = len(order) - 1 - row
        s = g[g.grouping == kind].sort_values("k")
        a1.plot([lo, hi], [y, y], color="#e3e3df", lw=0.7, zorder=1)
        a1.plot(s.pct, [y] * len(s), "o", ms=4.0, color=COL[0], alpha=0.95,
                markeredgecolor="white", markeredgewidth=0.7, zorder=3)
    a1.set_yticks(range(len(order)))
    a1.set_yticklabels([GN[k] for k in order][::-1], fontsize=6.0)
    a1.set_ylim(-0.6, len(order) - 0.4)
    a1.set_xlim(lo - 0.06, hi + 0.06)
    a1.set_xlabel("Normalized MAE (%)", fontsize=7.4, labelpad=1.5)
    a1.grid(axis="y", visible=False)
    head(a1, f"(a) Seventeen grouping settings, all inside {hi - lo:.1f} points")

    c = d[d.experiment == "coverage"]
    tiers = {"ami15": 15, "ami35": 35, "ami70": 70}
    ends = []
    for i, (tag, lab) in enumerate(CN.items()):
        sub = c[c.tag == tag]
        xs = np.array([tiers[t] for t in sub.tier]); o = np.argsort(xs)
        ys = (100 * sub.nmae_med.to_numpy())[o]
        series(a2, xs[o], ys, i, lab)
        ends.append((ys[-1], lab, COL[i]))
    # direct end labels carry identity here: the five lines are well separated at the
    # right edge, so a legend box would only repeat what the labels already say
    for yv, lab, cc in spread(ends, 2.3):
        a2.annotate(lab, (72.5, yv), fontsize=6.4, color=INK2, va="center",
                    ha="left", annotation_clip=False)
    a2.set_xticks([15, 35, 70])
    a2.set_xlabel("Metering coverage of the fleet (%)", labelpad=1.5)
    a2.set_ylabel("nMAE (%)", fontsize=7.2)
    a2.set_xlim(11, 71)
    head(a2, "(b) Forecasting a transformer that has no meter")
    fig.tight_layout(pad=0.2, h_pad=1.2)
    save(fig, "fig3_group_coverage")


# ------------------------------------------------------------- Fig. 4 screening
def fig_screen(d):
    style()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W1, 2.2))

    y = d[d.experiment == "growth"]
    lv = {"ev0": 0, "ev15": 15, "ev30": 30, "ev50": 50}
    # the two mean-trained arms sit on zero and would hide each other, so they are
    # drawn once as a single band with both names on it
    flat = ["cluster", "global_cluster"]
    for i, (tag, lab) in enumerate([("global_cluster_q90", "fleet model, 90th percentile"),
                                    ("seasonal_naive", "same hour last week")]):
        s = y[y.tag == tag]
        xs = np.array([lv[t] for t in s.tier]); o = np.argsort(xs)
        series(a1, xs[o], s.ovl100_f1.to_numpy()[o], i, lab)
    zs = y[y.tag.isin(flat)].groupby("tier").ovl100_f1.max()
    xs = np.array([lv[t] for t in zs.index]); o = np.argsort(xs)
    a1.plot(xs[o], zs.to_numpy()[o], color=COL[2], lw=2.6, marker="^", markersize=4.2,
            markeredgecolor="white", markeredgewidth=0.6, zorder=4,
            label="every mean-trained model")

    a1.set_xticks([0, 15, 30, 50])
    a1.set_xlabel("Share of homes with an electric vehicle (%)", labelpad=1.5)
    a1.set_ylabel("Screening F1", fontsize=7.2)
    a1.set_ylim(-0.03, 0.60); a1.set_xlim(-2, 72)
    for yv, lab, cc in spread([(0.508, "90th percentile", COL[0]),
                               (0.432, "same hour last week", COL[1]),
                               (0.017, "every mean-trained model", COL[2])], 0.10):
        a1.annotate(lab, (52.5, yv), fontsize=6.5, color=INK2, va="center", ha="left",
                    annotation_clip=False)
    head(a1, "(a) Screening quality as electric vehicles arrive")

    q = d[d.experiment == "quantile"]
    for i, (tag, lab) in enumerate([("global_cluster", "fleet model with groups"),
                                    ("local", "one model per transformer")]):
        s = q[q.tag == tag].sort_values("alpha")
        xs, ys = 100 * s.nmae_med.to_numpy(), s.ovl100_f1.to_numpy()
        series(a2, xs, ys, i, lab)
        # label selectively: the best point and the last point only
        best = int(np.argmax(ys))
        al = s.alpha.to_numpy()
        for j, dy, ha in ((best, 7.5 if i == 0 else -11, "center"),
                          (len(xs) - 1, 6 if i == 0 else 7.5, "center")):
            if j == best and j == len(xs) - 1:
                continue
            a2.annotate(f"q{al[j]:g}".replace("q0.", "q."), (xs[j], ys[j]),
                        fontsize=6.4, color=INK3, ha=ha,
                        xytext=(0, dy), textcoords="offset points")
    a2.set_xlabel("Normalized MAE (%)", labelpad=1.5)
    a2.set_ylabel("Screening F1", fontsize=7.2)
    a2.set_ylim(-0.08, 0.74)    # headroom so the q.95 label stays inside the plot
    h, l = a2.get_legend_handles_labels()
    # the lower-right corner is empty, and both q labels sit above their points
    a2.legend(h, l, loc="lower right", frameon=False, fontsize=6.4, labelcolor=INK2,
              handlelength=1.8, handletextpad=0.45, labelspacing=0.3, borderaxespad=0.2)
    head(a2, "(b) What the screen costs in accuracy")
    fig.tight_layout(pad=0.2, h_pad=1.2)
    save(fig, "fig4_screening")


if __name__ == "__main__":
    d = load()
    fig_pipeline()
    fig_kva()
    fig_group_cov(d)
    # fig_group_cov_wide(d)  # kept for reference; the stacked form lays out better
    fig_screen(d)
