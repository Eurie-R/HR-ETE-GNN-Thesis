"""Three deck figures: the alpha dial, the GNN, and the RQ2 topology contrast.

Everything is drawn from the repo's own numbers: returns from prices_cache.csv,
graphs reproduced by repro_graphs.py (verified identical to T4/T5), architecture
read off hetegnn_base.py (Config + ETEGNN).
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Circle

mpl.use("Agg")

ROOT = "/home/uriel/repositories/HR-ETE-GNN-THESIS_FINAL"
sys.path.insert(0, os.path.join(ROOT, ".build"))
sys.path.insert(0, ROOT)
SP = os.path.dirname(os.path.abspath(__file__))
OUT = f"{ROOT}/deck_figures"

INK = "#14213D"
ACCENT = "#1D5B8F"
RED = "#A8352A"
GRAY = "#5A6472"
LIGHT = "#E9F0F7"
ZEBRA = "#F4F7FA"
GOLD = "#B8860B"
BODY = "#2A3345"
NEUTRAL = GRAY

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": GRAY, "ytick.color": GRAY,
    "axes.edgecolor": "#C9D2DC", "axes.linewidth": 0.9,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

TICKERS = ["EWA", "EWC", "EWG", "EWJ", "EWT", "EWU", "EWW", "EWY", "EWZ", "EZA"]
COUNTRY = {"EWA": "Australia", "EWC": "Canada", "EWG": "Germany", "EWJ": "Japan",
           "EWT": "Taiwan", "EWU": "UK", "EWW": "Mexico", "EWY": "South Korea",
           "EWZ": "Brazil", "EZA": "South Africa"}

px = pd.read_csv(f"{ROOT}/preliminary_results/prices_cache.csv", index_col=0, parse_dates=True)
log_ret = np.log(px).diff().dropna()
train_ret = log_ret.loc[:"2021-11-24", TICKERS]
G = np.load(f"{SP}/graphs.npz", allow_pickle=True)


# =============================================================== FIGURE 1
def fig_alpha_dial():
    """What the alpha dial actually does to the weight a day carries."""
    r = 100 * train_ret.values.ravel()
    r = r[np.isfinite(r)]
    absr = np.abs(r)

    edges = [0, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
    labels = ["0–0.5%\nquiet", "0.5–1%\nnormal", "1–2%\nbusy",
              "2–3%\nrough", "3–5%\nsevere", ">5%\ncrash"]
    counts = np.array([np.sum((absr >= lo) & (absr < hi)) for lo, hi in zip(edges[:-1], edges[1:])], float)
    p = counts / counts.sum()

    def escort_multiplier(p, alpha):
        q = p ** alpha
        q = q / q.sum()
        return q / p

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.0),
                             gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.24})

    # -- left: the real distribution of daily moves ------------------------
    ax = axes[0]
    bars = ax.bar(range(len(p)), 100 * p, color=ACCENT, width=0.72, zorder=3)
    for b, c in zip(bars[-2:], [GOLD, RED]):
        b.set_color(c)
    for i, (b, v, n) in enumerate(zip(bars, 100 * p, counts)):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color=b.get_facecolor())
        ax.text(b.get_x() + b.get_width() / 2, v + 4.4, f"{int(n):,} days",
                ha="center", va="bottom", fontsize=8.4, color=GRAY)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("share of all trading days (%)", fontsize=10.5)
    ax.set_title("The days a crisis is made of are rare",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.set_ylim(0, 100 * p.max() * 1.34)
    ax.grid(axis="y", color="#E6EBF1", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    tail = 100 * p[-2:].sum()
    ax.annotate(f"the >3% days are only\n{tail:.1f}% of the sample —\n"
                "and the only ones\nanyone worries about",
                xy=(4.62, 100 * p[-2:].max() + 0.8),
                xytext=(3.05, 100 * p.max() * 1.10),
                fontsize=9.8, color=RED, style="italic", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.20"))

    # -- right: how much each day counts, per alpha ------------------------
    ax = axes[1]
    w = 0.26
    xs = np.arange(len(p))
    series = [(0.5, RED, "α = 0.5  — tail-sensitive (Rényi)"),
              (1.0, GRAY, "α = 1.0  — every day equal (Shannon)"),
              (1.5, ACCENT, "α = 1.5  — bulk-sensitive (Rényi)")]
    for off, (a, c, lab) in zip((-w, 0, w), series):
        m = escort_multiplier(p, a)
        ax.bar(xs + off, m, width=w, color=c, label=lab, zorder=3)
    ax.axhline(1.0, color=GRAY, lw=1.1, ls="--", zorder=2)
    ax.text(-0.46, 1.06, "equal weight", fontsize=8.6, color=GRAY, style="italic")

    m05 = escort_multiplier(p, 0.5)
    ax.annotate(f"a crash day counts\n{m05[-1]:.1f}× more under α = 0.5",
                xy=(len(p) - 1 - w, m05[-1] * 0.995),
                xytext=(len(p) - 3.45, m05[-1] * 0.90),
                fontsize=10.2, color=RED, fontweight="bold", ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.3,
                                connectionstyle="arc3,rad=-0.22"))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("weight the estimator gives that day\n(1.0 = ordinary transfer entropy)", fontsize=10.5)
    ax.set_title("Turning α below 1 is what makes crash days count",
                 fontsize=13, fontweight="bold", pad=12, loc="left")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(axis="y", color="#E6EBF1", zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(m05.max(), 1.2) * 1.22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.text(0.008, 0.015,
             f"Daily log returns, all ten country ETFs, graph-estimation window "
             f"{train_ret.index.min().date()} to {train_ret.index.max().date()} "
             f"({len(train_ret):,} days × 10 = {len(r):,} observations). "
             "Weight shown is the Rényi escort weight pᵅ/Σpᵅ relative to p.",
             fontsize=8.2, color=GRAY)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.145)
    fig.savefig(f"{OUT}/S15_alpha_dial.png", dpi=200)
    plt.close(fig)
    print(f"S15_alpha_dial.png   (crash-day multiplier at α=0.5: {m05[-1]:.2f}×)")


# =============================================================== graph layout
def circle_pos(n, r=1.0, start=np.pi / 2):
    ang = start - np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([r * np.cos(ang), r * np.sin(ang)])


def draw_network(ax, M, title, subtitle, highlight=None, pos=None, node_c=ACCENT):
    """M is [source, target]; draw curved arrows source -> target."""
    n = len(TICKERS)
    pos = circle_pos(n) if pos is None else pos
    outdeg = (M > 0).sum(axis=1)
    wmax = M.max() or 1.0

    for i in range(n):
        for j in range(n):
            if M[i, j] <= 0:
                continue
            lw = 0.9 + 3.0 * (M[i, j] / wmax)
            col = RED if (highlight and TICKERS[i] in highlight) else "#9AB3C8"
            alpha = 0.95 if (highlight and TICKERS[i] in highlight) else 0.72
            ax.add_patch(FancyArrowPatch(
                pos[i], pos[j], connectionstyle="arc3,rad=0.17",
                arrowstyle="-|>", mutation_scale=11, lw=lw,
                color=col, alpha=alpha, shrinkA=17, shrinkB=19, zorder=2))

    for i, t in enumerate(TICKERS):
        d = outdeg[i]
        rad = 0.085 + 0.030 * d
        c = RED if (highlight and t in highlight) else (node_c if d else "#B9C4D0")
        ax.add_patch(Circle(pos[i], rad, facecolor=c, edgecolor="white",
                            lw=1.8, zorder=4))
        ax.text(pos[i][0], pos[i][1], str(d), ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="white", zorder=5)
        lx, ly = pos[i] * 1.42
        ax.text(lx, ly, COUNTRY[t], ha="center", va="center", fontsize=9.6,
                color=INK if d else GRAY,
                fontweight="bold" if (highlight and t in highlight) else "normal",
                zorder=5)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=30)
    if subtitle:
        ax.text(0.5, 1.017, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=10.2, color=GRAY)
    ax.set_xlim(-1.78, 1.78)
    ax.set_ylim(-1.72, 1.72)
    ax.set_aspect("equal")
    ax.axis("off")


# =============================================================== FIGURE 2
def fig_gnn():
    """What the graph neural network actually does, with the real measured graph."""
    fig = plt.figure(figsize=(17.8, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 1.30, 0.92], wspace=0.02,
                          left=0.012, right=0.988, top=0.815, bottom=0.088)

    M = G["M_0.5"]
    pos = circle_pos(10)

    # ---------- stage 1: what each country brings ------------------------
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.175, 0.845, "1 · Each country's own recent history",
             ha="center", fontsize=12.5, fontweight="bold", color=INK)

    rng = np.random.default_rng(7)
    show = ["EWG", "EWZ", "EWJ"]
    for k, t in enumerate(show):
        y = 0.78 - 0.155 * k
        ax.add_patch(FancyBboxPatch((0.07, y - 0.055), 0.80, 0.105,
                                    boxstyle="round,pad=0.008,rounding_size=0.012",
                                    facecolor=ZEBRA, edgecolor="#C9D2DC", lw=0.9))
        seg = 100 * train_ret[t].values[-260 - 60 * k:][:20]
        xs = np.linspace(0.12, 0.72, len(seg))
        ys = y + 0.035 * seg / (np.abs(seg).max() or 1)
        ax.plot(xs, ys, color=ACCENT, lw=1.3)
        ax.axhline(y, xmin=0.12, xmax=0.72, color="#C9D2DC", lw=0.6)
        ax.text(0.79, y, COUNTRY[t], fontsize=9.2, va="center", color=INK)
    ax.text(0.47, 0.335, "⋮", ha="center", fontsize=15, color=GRAY)
    ax.text(0.47, 0.295, "all 10 countries", ha="center", fontsize=9.2,
            color=GRAY, style="italic")
    ax.annotate("", xy=(0.47, 0.78 + 0.075), xytext=(0.47, 0.905),
                arrowprops=dict(arrowstyle="-", color="white"))
    ax.text(0.5, 0.20, "the last 20 days of returns,\nfor each country",
            ha="center", fontsize=10, color=GRAY, style="italic")

    ax.add_patch(FancyBboxPatch((0.10, 0.045), 0.74, 0.115,
                                boxstyle="round,pad=0.01,rounding_size=0.016",
                                facecolor=LIGHT, edgecolor=ACCENT, lw=1.3))
    ax.text(0.47, 0.128, "multi-scale 1-D convolution",
            ha="center", fontsize=10.4, fontweight="bold", color=INK)
    ax.text(0.47, 0.078, "three filters — 3, 5 and 7 days wide, 12 each\n"
                         "→ a 576-number summary per country",
            ha="center", fontsize=9.0, color=GRAY)

    # ---------- stage 2: the measured graph ------------------------------
    ax = fig.add_subplot(gs[1])
    draw_network(ax, M, "", "", highlight={"EWG"}, pos=pos)
    fig.text(0.512, 0.845, "2 · The measured graph decides who listens to whom",
             ha="center", fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.512, 0.048,
             "arrows: significant ER-TE flow, source → receiver   ·   "
             "number in circle = out-degree   ·   red = Germany, the top transmitter",
             ha="center", fontsize=9.1, color=GRAY)
    fig.text(0.512, 0.018,
             "measured from training returns only (α = 0.5, top-20 edges)",
             ha="center", fontsize=9.1, color=GRAY, style="italic")

    # ---------- stage 3: message passing + head --------------------------
    ax = fig.add_subplot(gs[2])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.text(0.845, 0.845, "3 · Three rounds of listening",
             ha="center", fontsize=12.5, fontweight="bold", color=INK)

    rows = [("graph layer 1", "576 → 64"), ("graph layer 2", "64 → 32"),
            ("graph layer 3", "32 → 1")]
    for k, (name, dims) in enumerate(rows):
        y = 0.800 - 0.140 * k
        ax.add_patch(FancyBboxPatch((0.10, y - 0.052), 0.80, 0.104,
                                    boxstyle="round,pad=0.008,rounding_size=0.014",
                                    facecolor=LIGHT, edgecolor=ACCENT, lw=1.2))
        ax.text(0.28, y, name, fontsize=10.2, va="center", color=INK, fontweight="bold")
        ax.text(0.79, y, dims, fontsize=9.6, va="center", ha="right", color=GRAY)
        if k < 2:
            ax.annotate("", xy=(0.50, y - 0.086), xytext=(0.50, y - 0.058),
                        arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=1.4))

    ax.add_patch(FancyBboxPatch((0.055, 0.300), 0.89, 0.125,
                                boxstyle="round,pad=0.01,rounding_size=0.014",
                                facecolor="white", edgecolor="#C9D2DC", lw=1.0))
    ax.text(0.5, 0.388, "each round, every country updates itself as",
            ha="center", fontsize=9.3, color=GRAY)
    ax.text(0.5, 0.333, r"$h_i \leftarrow \mathrm{ReLU}\left(h_iW_1 + "
                        r"\sum_{j\rightarrow i} A_{ji}\,h_jW_2\right)$",
            ha="center", fontsize=13.5, color=INK)

    ax.text(0.5, 0.253, "its own state, plus what flows in along the arrows",
            ha="center", fontsize=9.4, color=GRAY, style="italic")

    ax.add_patch(FancyBboxPatch((0.10, 0.055), 0.80, 0.135,
                                boxstyle="round,pad=0.01,rounding_size=0.016",
                                facecolor=RED, edgecolor="none"))
    ax.text(0.5, 0.150, "one number per country", ha="center", fontsize=10.6,
            fontweight="bold", color="white")
    ax.text(0.5, 0.090, "tomorrow's realized volatility\nRV$_{t+1}$, all 10 at once",
            ha="center", fontsize=9.6, color="white")

    # ---------- connectors between the three stages ----------------------
    for x in (0.338, 0.676):
        fig.add_artist(mpl.lines.Line2D([x, x], [0.10, 0.80], color="#E1E7EE", lw=1.0))
        fig.add_artist(FancyArrowPatch((x - 0.014, 0.45), (x + 0.014, 0.45),
                                       arrowstyle="-|>", mutation_scale=17,
                                       color=ACCENT, lw=2.0))

    fig.suptitle("How the HR-ETE-GNN turns ten countries into ten forecasts",
                 fontsize=16.5, fontweight="bold", color=INK, y=0.978)
    fig.text(0.5, 0.905,
             "the graph is measured from the data, not assumed — that is the only thing "
             "that changes between the models compared in RQ1",
             ha="center", fontsize=10.6, color=GRAY)
    fig.savefig(f"{OUT}/S14_gnn_architecture.png", dpi=200)
    plt.close(fig)
    print("S14_gnn_architecture.png")


# =============================================================== FIGURE 3
def fig_topology():
    """RQ2: same data, same estimator, same test — only alpha changes."""
    M_sh, M_re = G["M_1.0"], G["M_0.5"]
    A_sh, A_re = G["A_1.0"], G["A_0.5"]
    pos = circle_pos(10)

    od_sh = (M_sh > 0).sum(axis=1)
    od_re = (M_re > 0).sum(axis=1)
    movers = {TICKERS[i] for i in range(10)
              if abs(pd.Series(-od_sh).rank(method="min")[i]
                     - pd.Series(-od_re).rank(method="min")[i]) >= 3}

    fig = plt.figure(figsize=(17.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.235], hspace=0.06, wspace=0.03,
                          left=0.015, right=0.985, top=0.790, bottom=0.115)

    ax = fig.add_subplot(gs[0, 0])
    draw_network(ax, M_sh, "Shannon   (α = 1.0)",
                 "the existing method — every day counts the same",
                 highlight=movers, pos=pos, node_c=GRAY)

    ax = fig.add_subplot(gs[0, 1])
    draw_network(ax, M_re, "Rényi   (α = 0.5)",
                 "tail-sensitive — crash days count more",
                 highlight=movers, pos=pos, node_c=ACCENT)

    # ---- the two summary strips -----------------------------------------
    for col, (lab, A, od) in enumerate([
            ("α = 1.0", A_sh, od_sh), ("α = 0.5", A_re, od_re)]):
        ax = fig.add_subplot(gs[1, col])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        top = [COUNTRY[TICKERS[i]] for i in np.argsort(-od)[:3] if od[i] > 0]
        n_fdr = int((A > 0).sum())
        c = ACCENT if col else GRAY
        ax.add_patch(FancyBboxPatch((0.045, 0.10), 0.91, 0.78,
                                    boxstyle="round,pad=0.01,rounding_size=0.02",
                                    facecolor=LIGHT if col else ZEBRA,
                                    edgecolor=c, lw=1.2))
        ax.text(0.09, 0.62, f"{n_fdr}", fontsize=21, fontweight="bold", color=c, va="center")
        ax.text(0.09, 0.30, "edges survive\nBH-FDR", fontsize=8.8, color=GRAY, va="center")
        ax.text(0.34, 0.62, "top transmitters:", fontsize=9.4, color=GRAY, va="center")
        ax.text(0.34, 0.31, ", ".join(top) if top else "—",
                fontsize=11.2, fontweight="bold", color=INK, va="center")
    fig.suptitle("Same data. Same estimator. Same test. Only α changes.",
                 fontsize=17.5, fontweight="bold", color=INK, y=0.972)
    fig.text(0.5, 0.940,
             "Top-20 edges by measured flow, so the two networks are compared at matched density.\n"
             "Circle size and the number inside = out-degree   ·   red marks a country whose rank moves by 3 places or more.",
             ha="center", va="top", fontsize=10.4, color=GRAY, linespacing=1.5)
    fig.text(0.5, 0.030,
             "12 of the 20 edges are shared — 4.4 expected by chance, hypergeometric p < 0.0001 — so the two networks agree more than chance.\n"
             "But Kendall τ = 0.453 on the out-degree rankings, and Australia, Canada and South Africa each move three places or more.",
             ha="center", va="bottom", fontsize=9.6, color=GRAY, style="italic", linespacing=1.5)
    fig.savefig(f"{OUT}/S28_topology_alpha_contrast.png", dpi=200)
    plt.close(fig)
    print(f"S28_topology_alpha_contrast.png   (movers: {sorted(COUNTRY[m] for m in movers)})")




# =============================================================== FIGURE 4
def fig_four_scenarios():
    """The four toy scenarios, with every RV panel on ONE shared scale.

    Same scenarios and same RV recipe as explanation.ipynb cell 19; the only
    change is that the realized-volatility column no longer auto-scales each
    panel separately, which made a flat line at 0.3 and a flat line at 200
    look identical.
    """
    rng2 = np.random.default_rng(3)
    n_toy, win_toy = 120, 20

    scen = {
        "Steadily rising": (100 * 1.003 ** np.arange(n_toy + 1), ACCENT,
                            "the same tiny +0.30% every day"),
        "Oscillating +100 / −100": (np.array([100.0 if i % 2 == 0 else -100.0
                                              for i in range(n_toy + 1)]), RED,
                                    "the largest possible move, every day"),
        "Very volatile": (100 * np.exp(np.r_[0, np.cumsum(rng2.normal(0, 0.06, n_toy))]),
                          GOLD, "big, erratic, patternless jumps"),
        "Flat / unchanged": (np.full(n_toy + 1, 100.0), GRAY, "nothing moves at all"),
    }

    def rv_of(path):
        pct = 100 * (path[1:] - path[:-1]) / path[:-1]
        return pd.Series(pct).pow(2).rolling(win_toy, min_periods=1).mean().pow(0.5).values

    rvs = {k: rv_of(np.asarray(v[0], float)) for k, v in scen.items()}
    rv_top = max(np.nanmax(r) for r in rvs.values())

    fig = plt.figure(figsize=(13.4, 12.4))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, 0.62],
                          width_ratios=[1, 1.15], hspace=0.62, wspace=0.17,
                          left=0.075, right=0.975, top=0.872, bottom=0.065)

    for row, (name, (path, col, blurb)) in enumerate(scen.items()):
        path = np.asarray(path, float)
        rv = rvs[name]

        ax = fig.add_subplot(gs[row, 0])
        ax.plot(path, color=INK, lw=1.4)
        ax.axhline(0, color="#C9D2DC", lw=0.7)
        ax.set_title(f"{name}  —  the path", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK, pad=26)
        ax.text(0, 1.022, blurb, transform=ax.transAxes, fontsize=9.4,
                color=GRAY, style="italic", va="bottom")
        ax.tick_params(labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        ax = fig.add_subplot(gs[row, 1])
        ax.plot(rv, color=col, lw=2.2, zorder=3)
        ax.fill_between(range(len(rv)), 0, rv, color=col, alpha=0.13, zorder=2)
        ax.set_ylim(0, rv_top * 1.09)
        ax.set_title(f"{name}  —  realized volatility", loc="left", fontsize=11.5,
                     fontweight="bold", color=INK, pad=26)
        ax.text(0, 1.022, "same scale on all four panels", transform=ax.transAxes,
                fontsize=9.4, color=GRAY, style="italic", va="bottom")
        ax.grid(axis="y", color="#EDF1F5", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        settled = rv[-1]
        if settled > 0.25 * rv_top:
            ax.annotate(f"RV ≈ {settled:.0f}%", xy=(len(rv) * 0.52, settled),
                        xytext=(len(rv) * 0.30, settled * 0.62),
                        fontsize=13, fontweight="bold", color=col,
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        else:
            ax.annotate(f"RV ≈ {settled:.2f}%" if settled else "RV = 0",
                        xy=(len(rv) * 0.62, settled),
                        xytext=(len(rv) * 0.34, rv_top * 0.70),
                        fontsize=13, fontweight="bold", color=col, va="center",
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.4,
                                        connectionstyle="arc3,rad=0.25"))
            ax.text(len(rv) * 0.34, rv_top * 0.45,
                    "dwarfed by the oscillator above —\nsee the comparison at the bottom",
                    fontsize=9.3, color=GRAY, style="italic", va="center")

    # ---- the comparison strip: the whole point, on one axis ---------------
    ax = fig.add_subplot(gs[4, :])
    box = ax.get_position()
    ax.set_position([box.x0 + 0.093, box.y0, box.width - 0.093, box.height])
    names = list(scen.keys())
    vals = np.array([rvs[n][-1] for n in names])
    cols = [scen[n][1] for n in names]
    order = np.argsort(vals)
    y = np.arange(len(names))
    ax.barh(y, vals[order] + 1e-3, color=[cols[i] for i in order], height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order], fontsize=10.5)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlim(0, rv_top * 3.2)
    ax.set_xlabel("realized volatility, % per step   (log scale — the only way to show "
                  "0, 0.30 and 200 together)", fontsize=10)
    for yi, i in enumerate(order):
        v = vals[i]
        ax.text(max(v, 0.1) * 1.35, yi, "0" if v == 0 else (f"{v:.2f}" if v < 10 else f"{v:.0f}"),
                va="center", fontsize=11, fontweight="bold", color=cols[i])
    ax.grid(axis="x", color="#EDF1F5", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("All four, finally on one axis", loc="left", fontsize=12.5,
                 fontweight="bold", color=INK, pad=9)

    fig.suptitle("Four scenarios, one ruler", fontsize=18, fontweight="bold",
                 color=INK, y=0.980)
    fig.text(0.5, 0.947,
             "The oscillator's RV is flat — but flat at 200, not flat at zero. "
             "Auto-scaling each panel separately hides exactly that.",
             ha="center", fontsize=11.2, color=GRAY)
    fig.savefig(f"{OUT}/S06_S07_four_scenarios.png", dpi=190)
    plt.close(fig)
    print("S06_S07_four_scenarios.png   " +
          ", ".join(f"{n.split()[0]}={rvs[n][-1]:.2f}" for n in names))




# =============================================================== FIGURE 5
def _h1b_worked_example():
    """Six real test days, real RV targets, real regime flags.

    The two forecast columns are illustrative — the pilot did not save per-day
    forecasts — but every step from forecast to beta is the real arithmetic.
    """
    import core as hr

    rv_all = (100 * log_ret).pow(2).rolling(20).mean().pow(0.5).dropna()
    sp = hr.make_splits(rv_all[TICKERS].values, lookback=20, train_frac=.60, val_frac=.20)
    test_dates = rv_all.index[sp["t_test"]]

    def hurst_rs(series):
        s = np.asarray(series, float); N = len(s)
        if N < 20:
            return np.nan
        out = []
        for lag in np.unique(np.logspace(1, np.log10(N // 2), 10).astype(int)):
            vals = []
            for j in range(N // lag):
                seg = s[j * lag:(j + 1) * lag]; sd = seg.std(ddof=1)
                if sd > 0:
                    dev = np.cumsum(seg - seg.mean()); vals.append((dev.max() - dev.min()) / sd)
            if vals:
                out.append((lag, np.mean(vals)))
        if len(out) < 2:
            return np.nan
        L, R = zip(*out)
        return np.polyfit(np.log(L), np.log(R), 1)[0]

    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regime_test.csv")
    if os.path.exists(cache):
        reg = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
    else:
        h = log_ret["URTH"].rolling(250).apply(hurst_rs, raw=True).dropna()
        reg = (h < 0.5).astype(float).shift(1).dropna().reindex(test_dates).fillna(0.0)
        reg.to_frame("turbulent").to_csv(cache)

    first = list(test_dates).index(reg[reg > 0].index[0])
    win = list(test_dates[first - 3: first]) + list(test_dates[first: first + 3])
    etf3 = ["EWG", "EWZ", "EWJ"]
    actual = rv_all.loc[win, etf3]
    turb = reg.loc[win].values.astype(bool)

    rng = np.random.default_rng(11)
    A = actual.to_numpy(copy=True)
    sh = A * (1 + rng.normal(0.00, 0.045, A.shape))
    re = A * (1 + rng.normal(0.01, 0.045, A.shape))
    re[turb] = A[turb] * (1 + rng.normal(0.005, 0.02, (turb.sum(), 3)))
    sh[turb] = A[turb] * (1 + rng.normal(0.11, 0.05, (turb.sum(), 3)))

    def qlike(a, f):
        r = a / f
        return r - np.log(r) - 1.0

    L_sh, L_re = qlike(A, sh), qlike(A, re)
    return dict(dates=[d.date().isoformat() for d in win], turb=turb, actual=A,
                sh=sh, re=re, L_sh=L_sh, L_re=L_re,
                day_sh=L_sh.mean(axis=1), day_re=L_re.mean(axis=1),
                d=L_sh.mean(axis=1) - L_re.mean(axis=1), etf3=etf3)


def _panel(ax, title, step):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                facecolor="white", edgecolor="#D5DDE5", lw=1.1))
    fig = ax.figure
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    ar = (pos.width * fw) / (pos.height * fh)
    r = 0.030 if ar < 3 else 0.030 / ar * 2.2
    ax.add_patch(Ellipse((0.028 + r, 0.90), 2 * r, 2 * r * ar,
                         facecolor=ACCENT, edgecolor="none"))
    ax.text(0.028 + r, 0.90, step, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color="white")
    ax.text(0.028 + 2 * r + 0.022, 0.90, title, ha="left", va="center", fontsize=12.6,
            fontweight="bold", color=INK)


def _grid(ax, cols, rows, x0, y0, w, h, fs=10.2, head_c=ACCENT, row_flags=None):
    """Simple text table in axes coordinates."""
    n = len(cols)
    cw = [w * c[1] for c in cols]
    xs = np.cumsum([x0] + cw[:-1])
    rh = h / (len(rows) + 1)
    ax.add_patch(FancyBboxPatch((x0 - 0.008, y0 + h - rh), w + 0.016, rh,
                                boxstyle="square,pad=0", facecolor=head_c, edgecolor="none"))
    for x, (lab, _) in zip(xs, cols):
        ax.text(x, y0 + h - rh / 2, lab, ha="left", va="center", fontsize=fs - 0.6,
                fontweight="bold", color="white")
    for ri, row in enumerate(rows):
        y = y0 + h - rh * (ri + 2) + rh / 2
        flag = (row_flags or [None] * len(rows))[ri]
        if flag == "turb":
            ax.add_patch(FancyBboxPatch((x0 - 0.008, y - rh / 2), w + 0.016, rh,
                                        boxstyle="square,pad=0",
                                        facecolor="#FBEDEA", edgecolor="none"))
        elif ri % 2 == 0:
            ax.add_patch(FancyBboxPatch((x0 - 0.008, y - rh / 2), w + 0.016, rh,
                                        boxstyle="square,pad=0",
                                        facecolor=ZEBRA, edgecolor="none"))
        for x, cell in zip(xs, row):
            txt, col, bold = cell if isinstance(cell, tuple) else (cell, BODY, False)
            ax.text(x, y, txt, ha="left", va="center", fontsize=fs,
                    color=col, fontweight="bold" if bold else "normal")
    return y0 + h - rh * (len(rows) + 1)


def fig_h1b_walkthrough():
    E = _h1b_worked_example()
    d = E["d"]
    calm, turb = d[~E["turb"]], d[E["turb"]]
    b0, b1 = calm.mean(), turb.mean() - calm.mean()

    fig = plt.figure(figsize=(17.6, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.42], width_ratios=[1.30, 0.92, 1.02],
                          hspace=0.10, wspace=0.035, left=0.012, right=0.988,
                          top=0.855, bottom=0.035)

    # ---------------- panel 1: score every forecast -----------------------
    ax = fig.add_subplot(gs[0, 0])
    _panel(ax, "Score every forecast with QLIKE", "1")
    ax.text(0.048, 0.828, "Germany, six real test days — the real run: 10 markets × 314 days",
            fontsize=9.0, color=GRAY, style="italic", va="center")
    rows, flags = [], []
    for i, dt in enumerate(E["dates"]):
        t = E["turb"][i]
        rows.append([(dt, INK, t), ("TURBULENT" if t else "calm", RED if t else GRAY, t),
                     f"{E['actual'][i,0]:.3f}", f"{E['sh'][i,0]:.3f}", f"{E['re'][i,0]:.3f}",
                     (f"{E['L_sh'][i,0]:.5f}", BODY, False),
                     (f"{E['L_re'][i,0]:.5f}", BODY, False)])
        flags.append("turb" if t else None)
    _grid(ax, [("date", .20), ("regime", .17), ("RV actual", .13), ("Shannon fc", .14),
               ("Rényi fc", .12), ("QLIKE sh", .13), ("QLIKE re", .12)],
          rows, 0.048, 0.355, 0.905, 0.425, fs=9.6, row_flags=flags)

    ax.text(0.048, 0.298, "QLIKE — a golf score for a forecast: 0 is perfect, higher is worse",
            fontsize=10, color=INK, fontweight="bold", va="center")
    ax.text(0.048, 0.185, r"$\mathrm{QLIKE}=\dfrac{RV}{\widehat{RV}}-"
                          r"\ln\dfrac{RV}{\widehat{RV}}-1$",
            fontsize=14, color=INK, va="center")
    r_sh = E["actual"][3, 0] / E["sh"][3, 0]
    r_re = E["actual"][3, 0] / E["re"][3, 0]
    ax.text(0.41, 0.246, "worked, for 2023-09-29 (Germany):", fontsize=9.2,
            color=GRAY, va="center")
    ax.text(0.41, 0.192, f"Shannon:  {E['actual'][3,0]:.3f} / {E['sh'][3,0]:.3f} = {r_sh:.4f}",
            fontsize=9.2, color=BODY, va="center")
    ax.text(0.455, 0.146,
            f"→  {r_sh:.4f} − ln({r_sh:.4f}) − 1 = {E['L_sh'][3,0]:.5f}",
            fontsize=9.2, color=BODY, va="center")
    ax.text(0.41, 0.094, f"Rényi:       {E['actual'][3,0]:.3f} / {E['re'][3,0]:.3f} = {r_re:.4f}",
            fontsize=9.2, color=BODY, va="center")
    ax.text(0.455, 0.048,
            f"→  {r_re:.4f} − ln({r_re:.4f}) − 1 = {E['L_re'][3,0]:.5f}",
            fontsize=9.2, color=BODY, va="center")
    ax.text(0.048, 0.072, "Rényi scored lower,\nso Rényi forecast better.",
            fontsize=9.6, color=RED, style="italic", va="center", fontweight="bold",
            linespacing=1.4)

    # ---------------- panel 2: average, then subtract ---------------------
    ax = fig.add_subplot(gs[0, 1])
    _panel(ax, "Average the markets, subtract", "2")
    ax.text(0.048, 0.828, "averaged over the 3 markets — panel 1 showed only Germany",
            fontsize=9.0, color=GRAY, style="italic", va="center")
    rows, flags = [], []
    for i, dt in enumerate(E["dates"]):
        t = E["turb"][i]
        rows.append([(dt, INK, t), f"{E['day_sh'][i]:.5f}", f"{E['day_re'][i]:.5f}",
                     (f"{d[i]:+.5f}", RED if d[i] > 0 else NEUTRAL, True),
                     ("Rényi" if d[i] > 0 else "Shannon", RED if d[i] > 0 else NEUTRAL, False)])
        flags.append("turb" if t else None)
    _grid(ax, [("date", .26), ("L Shannon", .19), ("L Rényi", .17), ("margin  d", .20),
               ("who won", .18)],
          rows, 0.048, 0.355, 0.905, 0.425, fs=9.8, row_flags=flags)
    ax.text(0.048, 0.300, "the margin is the baseline's loss minus ours:",
            fontsize=10, color=INK, fontweight="bold", va="top")
    ax.text(0.048, 0.228, r"$d_t = \mathrm{QLIKE}_t(\mathrm{Shannon}) - "
                          r"\mathrm{QLIKE}_t(\mathrm{R\acute{e}nyi})$",
            fontsize=13, color=INK, va="top")
    ax.text(0.048, 0.135, "so a LOWER loss for us makes d LARGER.\n"
                          "d > 0 means we won that day.",
            fontsize=9.9, color=RED, va="top", style="italic", fontweight="bold")

    # ---------------- panel 3: two piles ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    _panel(ax, "Sort into two piles, average each", "3")
    ax.text(0.048, 0.828, "a 0/1 dummy makes OLS equal the group means",
            fontsize=9.0, color=GRAY, style="italic", va="center")

    ax.add_patch(FancyBboxPatch((0.05, 0.505), 0.42, 0.28,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=ZEBRA, edgecolor="#C9D2DC", lw=1.0))
    ax.text(0.26, 0.742, f"CALM  ({len(calm)} days)", ha="center", fontsize=10.4,
            fontweight="bold", color=NEUTRAL)
    ax.text(0.26, 0.668, "  ".join(f"{v:+.5f}" for v in calm), ha="center",
            fontsize=8.8, color=BODY)
    ax.text(0.26, 0.600, "average", ha="center", fontsize=9.3, color=GRAY)
    ax.text(0.26, 0.545, f"β₀ = {b0:+.5f}", ha="center", fontsize=13.5,
            fontweight="bold", color=NEUTRAL)

    ax.add_patch(FancyBboxPatch((0.53, 0.505), 0.42, 0.28,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor="#FBEDEA", edgecolor=RED, lw=1.1))
    ax.text(0.74, 0.742, f"TURBULENT  ({len(turb)} days)", ha="center", fontsize=10.4,
            fontweight="bold", color=RED)
    ax.text(0.74, 0.668, "  ".join(f"{v:+.5f}" for v in turb), ha="center",
            fontsize=8.8, color=BODY)
    ax.text(0.74, 0.600, "average", ha="center", fontsize=9.3, color=GRAY)
    ax.text(0.74, 0.545, f"β₀+β₁ = {turb.mean():+.5f}", ha="center", fontsize=13.5,
            fontweight="bold", color=RED)

    ax.annotate("", xy=(0.50, 0.575), xytext=(0.50, 0.575),
                arrowprops=dict(arrowstyle="-", color=GRAY))
    ax.text(0.50, 0.650, "−", ha="center", va="center", fontsize=17, color=GRAY)

    ax.add_patch(FancyBboxPatch((0.05, 0.260), 0.90, 0.195,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=LIGHT, edgecolor=ACCENT, lw=1.4))
    ax.text(0.50, 0.398, "β₁  =  turbulent average  −  calm average", ha="center",
            fontsize=10.6, color=INK, va="center")
    ax.text(0.50, 0.315, f"= {turb.mean():+.5f}  −  ({b0:+.5f})  =  {b1:+.5f}",
            ha="center", fontsize=13.5, fontweight="bold", color=ACCENT, va="center")

    ax.text(0.50, 0.185, "β₁ > 0  →  our margin is BIGGER when markets are turbulent",
            ha="center", fontsize=10.6, fontweight="bold", color=INK, va="center")
    ax.text(0.50, 0.098, "which is exactly what tail-sensitivity is supposed to buy.\n"
                         "H₀: β₁ = 0  ·  H₁: β₁ > 0  ·  one-sided, HAC standard errors",
            ha="center", fontsize=9.5, color=GRAY, va="center", style="italic")

    # ---------------- bottom: what the real run produced ------------------
    ax = fig.add_subplot(gs[1, :])
    _panel(ax, "What this produced on the real 314-day test block", "4")
    ax.text(0.036, 0.735, "the numbers above are a six-day illustration; these are the "
            "actual pilot results (T11)", fontsize=9.3, color=GRAY, style="italic", va="top")

    cells = [
        ("turbulent days", "4 of 314", RED, "1.3% of the block"),
        ("β̂₀  calm margin", "−0.000673", NEUTRAL, "slightly behind in calm markets"),
        ("β̂₁  extra in turbulence", "−0.002039", RED, "wrong sign — margin got worse"),
        ("t-statistic", "−2.02", NEUTRAL, "one-sided, HAC"),
        ("one-sided p", "0.978", NEUTRAL, "nowhere near 0.05"),
    ]
    x = 0.036
    for lab, val, col, note in cells:
        w = 0.175
        ax.add_patch(FancyBboxPatch((x, 0.215), w - 0.012, 0.40,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    facecolor=ZEBRA, edgecolor="#C9D2DC", lw=1.0))
        ax.text(x + 0.012, 0.545, lab, fontsize=9.4, color=GRAY, va="center")
        ax.text(x + 0.012, 0.415, val, fontsize=15, fontweight="bold", color=col, va="center")
        ax.text(x + 0.012, 0.292, note, fontsize=8.7, color=GRAY, va="center", style="italic")
        x += w

    ax.add_patch(FancyBboxPatch((0.912, 0.215), 0.076, 0.40,
                                boxstyle="round,pad=0.008,rounding_size=0.02",
                                facecolor=GOLD, edgecolor="none"))
    ax.text(0.950, 0.470, "VERDICT", ha="center", fontsize=9.4, color="white",
            fontweight="bold", va="center")
    ax.text(0.950, 0.340, "NOT\nEVALUABLE", ha="center", fontsize=11.5, color="white",
            fontweight="bold", va="center", linespacing=1.3)

    ax.text(0.036, 0.135,
            "The four flagged days carried Hurst values of 0.4999, 0.4970, 0.4957 and 0.4987 — "
            "one crossed the threshold by a ten-thousandth. β̂₁ is the average of four days that "
            "were never crises, so no regime claim is made from this run.",
            fontsize=9.8, color=INK, va="center")

    fig.suptitle("H1b, end to end: from a forecast to a verdict",
                 fontsize=18, fontweight="bold", color=INK, y=0.972)
    fig.text(0.5, 0.923,
             "Real test dates, real RV targets, real Hurst regime flags. The two forecast "
             "columns in panel 1 are illustrative — the pilot did not store per-day forecasts — "
             "but every step of the arithmetic is the real one.",
             ha="center", fontsize=10.4, color=GRAY)
    fig.savefig(f"{OUT}/S21_h1b_walkthrough.png", dpi=190)
    plt.close(fig)
    print(f"S21_h1b_walkthrough.png   (illustration: β₀={b0:+.5f}, β₁={b1:+.5f})")


if __name__ == "__main__":
    fig_alpha_dial()
    fig_gnn()
    fig_topology()
    fig_four_scenarios()
    fig_h1b_walkthrough()
