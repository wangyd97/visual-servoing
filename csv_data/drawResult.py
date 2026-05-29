"""
plot_comparison_experiment_1A.py
================================
Experiment I-A: Translation-Dominant Large Error / Regulation comparison

6 子图（2行×3列）：
  Row 0 — 平移误差分量: e_x (mm) | e_y (mm) | e_z (mm)
  Row 1 — 旋转误差分量: r_x (deg)| r_y (deg)| r_z (deg)

四条曲线：
  R1 = Ribeiro-safe
  R2 = Ribeiro-matched / high-Kp-low-Kd baseline
  R3 = Ribeiro-damped / high-Kp-high-Kd baseline
  P  = Proposed method
"""

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import MaxNLocator


# ── 路径设置 ──────────────────────────────────────────────────────────────────
folder_name = "/home/mdl/mycode/python/26_0525_exp/e1/csv_data"

FILES = {
    "R1": os.path.join(folder_name, "SOPD_R1.csv"),
    "R2": os.path.join(folder_name, "SOPD_R2.csv"),
    "R3": os.path.join(folder_name, "SOPD_R3.csv"),
    "P":  os.path.join(folder_name, "SOPDPSMC.csv"),
}

LABELS = {
    "R1": "Baseline-ordinary",
    "R2": "Baseline-small Kd",
    "R3": "Baseline-large Kd",
    "P":  "Proposed method",
}

# 颜色建议：Proposed 用黑色压住，其余 baseline 用区分度较高的颜色
COLORS = {
    "R1": "#1F77B4",   # blue
    "R2": "#D62728",   # red
    "R3": "#9B30D0",   # purple
    "P":  "#1A1A1A",   # black
}

LINESTYLES = {
    "R1": "-",
    "R2": "-",
    "R3": "-",
    "P":  "-",
}

LINEWIDTHS = {
    "R1": 1.0,
    "R2": 1.0,
    "R3": 1.0,
    "P":  1.0,
}

ZORDERS = {
    "R1": 2,
    "R2": 3,
    "R3": 4,
    "P":  5,
}


# ── 读取数据 ──────────────────────────────────────────────────────────────────
dfs = {}

for key, file_path in FILES.items():
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)

    # 防止 lost frame 或异常行导致 t/误差列为 NaN
    df = df.copy()
    df["t"] = pd.to_numeric(df["t"], errors="coerce")

    dfs[key] = df


def valid_time(df: pd.DataFrame) -> np.ndarray:
    return df["t"].to_numpy(dtype=float)


T_MAX = max(
    np.nanmax(valid_time(df))
    for df in dfs.values()
    if np.any(np.isfinite(valid_time(df)))
)


# ── 子图规格 ──────────────────────────────────────────────────────────────────
#  (列名, 顶部标题, y轴单位后缀)
SPECS = [
    [("ex_mm",  r"$e_x$", "(mm)"),
     ("ey_mm",  r"$e_y$", "(mm)"),
     ("ez_mm",  r"$e_z$", "(mm)")],
    [("rx_deg", r"$r_x$", r"($^\circ$)"),
     ("ry_deg", r"$r_y$", r"($^\circ$)"),
     ("rz_deg", r"$r_z$", r"($^\circ$)")],
]


# ── 全局样式 ──────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "axes.labelsize": 9,
    "figure.dpi": 1200,
})


# ── 绘图 ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 3,
    figsize=(9, 5.0),
    gridspec_kw={"hspace": 0.38, "wspace": 0.35},
)

plot_order = ["R1", "R2", "R3", "P"]

for r, row_specs in enumerate(SPECS):
    for c, (col, title, unit) in enumerate(row_specs):
        ax = axes[r, c]

        for key in plot_order:
            df = dfs[key]

            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in {FILES[key]}")

            t = df["t"].to_numpy(dtype=float)
            y = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

            # lost frame 或异常行会自然变成 NaN，matplotlib 会断线
            ax.plot(
                t, y,
                color=COLORS[key],
                linestyle=LINESTYLES[key],
                lw=LINEWIDTHS[key],
                zorder=ZORDERS[key],
                label=LABELS[key],
            )

        # 零误差参考线
        ax.axhline(0, color="#777777", lw=0.7, linestyle=":", zorder=1)

        # x 轴
        ax.set_xlim(0, T_MAX)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))

        if r == 1:
            ax.set_xlabel("Time [s]", fontsize=15)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        # y 轴
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))

        # 顶部标题
        ax.set_title(
            f"{title} {unit}",
            fontsize=15,
            pad=4,
            fontweight="bold"
        )

        # 网格在曲线下方
        ax.set_axisbelow(True)

        # 刻度样式
        ax.tick_params(
            which="both",
            direction="in",
            top=True,
            right=True,
            labelsize=8
        )


# ── 图例 ──────────────────────────────────────────────────────────────────────
# ── 图例 ──────────────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D(
        [], [], color=COLORS[key],
        linestyle=LINESTYLES[key],
        lw=LINEWIDTHS[key] + 0.4,
        label=LABELS[key]
    )
    for key in plot_order
]

fig.legend(
    handles=handles,
    loc="upper center",
    ncol=2,                 # 关键：两列，因此 4 个 label 会变成两排
    frameon=False,
    fontsize=12,
    bbox_to_anchor=(0.5, 1.075),
    columnspacing=1.8,
    handlelength=2.4,
    handletextpad=0.6,
)

# ── 保存 ──────────────────────────────────────────────────────────────────────
file_name = os.path.join(folder_name, "exp1.png")
fig.savefig(file_name, dpi=300, bbox_inches="tight", facecolor="white")
print("Saved -> " + file_name)