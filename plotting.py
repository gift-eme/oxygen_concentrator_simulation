"""The single 7 x 5 in figure allowed in the technical note."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from isotherms import langmuir

COL = {"Romulus": "#2a78d6", "Remus": "#eb6834"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _label(ax, s):
    ax.text(-0.16, 1.04, s, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=INK)


def make_figure(data, iso, curves, sens, prm, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8})
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    ax_i, ax_b, ax_v, ax_l = axs.ravel()

    # (a) isotherms
    for a in ("Romulus", "Remus"):
        for gas, ls, mk, fill in (("N2", "-", "o", True), ("O2", "--", "s", False)):
            P, q = data[a][gas]
            Pf = np.linspace(0, 1e6, 300)
            x = iso[a][0] if gas == "N2" else iso[a][1]
            ax_i.plot(Pf / 1e3, langmuir(Pf, *x), ls, color=COL[a], lw=1.6)
            ax_i.plot(P / 1e3, q, mk, ms=4.5, mew=1.1, color=COL[a],
                      mfc=COL[a] if fill else "white",
                      label=f"{a} {gas.replace('2', '₂')}")
    ax_i.axvline(prm["P"] / 1e3, color=MUTED, lw=0.8, ls=":")
    ax_i.text(prm["P"] / 1e3 + 12, 1.45, "process\n100 kPa", fontsize=6.5,
              color=MUTED, va="bottom")
    ax_i.set_xlabel("Pressure [kPa]")
    ax_i.set_ylabel("q [mol kg⁻¹]")
    ax_i.set_xlim(0, 1000)
    ax_i.set_ylim(0, 2.75)
    ax_i.legend(fontsize=6.5, frameon=False, loc="upper left", ncol=2,
                bbox_to_anchor=(0.0, 1.02), columnspacing=1.0,
                handletextpad=0.3)
    ax_i.text(0.98, 0.04, "markers: data\nlines: Langmuir fit", ha="right",
              transform=ax_i.transAxes, fontsize=6, color=MUTED)
    _style(ax_i)
    _label(ax_i, "(a)")

    # (b) base-case breakthrough
    base = {m["adsorbent"]: m for m in sens if m["tag"] == "v x1.0"}
    for a in ("Romulus", "Remus"):
        c, m = curves[a], base[a]
        # bed is saturated after the run ends, so hold the last value
        t_plot = np.append(c["t"], 3500.0)
        y_plot = np.append(c["y_out"], c["y_out"][-1])
        ax_b.plot(t_plot, y_plot, color=COL[a], lw=1.8, label=f"{a} (model)")
        ax_b.axvline(m["t_eq"], color=COL[a], lw=0.9, ls="--")
        ax_b.plot([m["t_b"]], [0.05], "o", ms=5, color=COL[a], mec="white", mew=1)
        ax_b.annotate(f"$t_b$ = {m['t_b']:.0f} s", (m["t_b"], 0.05),
                      xytext=(-6, 14), textcoords="offset points", ha="right",
                      fontsize=6.5, color=INK2)
    ax_b.plot([], [], ls="--", color=MUTED, lw=0.9, label="equilibrium theory")
    ax_b.axhline(0.05, color=MUTED, lw=0.6, ls=":")
    ax_b.set_xlabel("Time [s]")
    ax_b.set_ylabel("Outlet N₂ mole fraction [-]")
    ax_b.set_xlim(0, 3500)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.legend(fontsize=6.5, frameon=False, loc="center right")
    _style(ax_b)
    _label(ax_b, "(b)")

    # (c) breakthrough time vs feed velocity
    for a in ("Romulus", "Remus"):
        rows = sorted([m for m in sens if m["tag"].startswith("v x")],
                      key=lambda m: m["v_in"])
        rows = [m for m in rows if m["adsorbent"] == a]
        v = np.array([m["v_in"] for m in rows]) * 1e3
        ax_v.plot(v, [m["t_eq"] for m in rows], "--", color=COL[a], lw=0.9)
        ax_v.plot(v, [m["t_b"] for m in rows], "-o", color=COL[a], lw=1.6,
                  ms=4.5, mec="white", mew=0.8, label=a)
        for m, vv in zip(rows, v):
            ax_v.annotate(f"{100 * m['utilisation']:.0f}%", (vv, m["t_b"]),
                          xytext=(4, -9), textcoords="offset points",
                          fontsize=6, color=INK2)
    ax_v.plot([], [], "--", color=MUTED, lw=0.9, label="equilibrium theory")
    ax_v.set_xscale("log")
    ax_v.set_yscale("log")
    ax_v.set_xlabel("Feed interstitial velocity [mm s⁻¹]")
    ax_v.set_ylabel("Breakthrough time $t_b$ [s]")
    ax_v.set_xticks([2.4, 4.9, 9.7, 19.5])
    ax_v.set_xticklabels(["2.4", "4.9", "9.7", "19.5"])
    ax_v.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax_v.set_xlim(2.0, 26)
    ax_v.set_ylim(150, 8000)
    ax_v.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax_v.text(0.03, 0.04, "labels: bed utilisation $t_b/t_{st}$",
              transform=ax_v.transAxes, fontsize=6, color=MUTED)
    _style(ax_v)
    _label(ax_v, "(c)")

    # (d) bed utilisation vs column length
    for a in ("Romulus", "Remus"):
        rows = [m for m in sens if m["adsorbent"] == a and
                (m["tag"].startswith("L=") or m["tag"] == "v x1.0")]
        rows.sort(key=lambda m: m["L"])
        L = [m["L"] for m in rows]
        ax_l.plot(L, [100 * m["utilisation"] for m in rows], "-o",
                  color=COL[a], lw=1.6, ms=4.5, mec="white", mew=0.8, label=a)
        dy = 6 if a == "Romulus" else -11
        for m in rows:
            ax_l.annotate(f"{m['t_b']:.0f} s", (m["L"], 100 * m["utilisation"]),
                          xytext=(0, dy), textcoords="offset points",
                          ha="center", fontsize=6, color=INK2)
    ax_l.set_xlabel("Column length L [m]")
    ax_l.set_ylabel("Bed utilisation $t_b/t_{st}$ [%]")
    ax_l.set_xticks([0.16, 0.32, 0.64])
    ax_l.set_xlim(0.12, 0.68)
    ax_l.set_ylim(60, 100)
    ax_l.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax_l.text(0.03, 0.92, "labels: $t_b$ at 4.9 mm s⁻¹",
              transform=ax_l.transAxes, fontsize=6, color=MUTED)
    _style(ax_l)
    _label(ax_l, "(d)")

    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    fig.savefig(path, dpi=300)
    plt.close(fig)
