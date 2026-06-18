# docs/source/make_pseudospectral_visualizations.py

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import wecopttool as wot


mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "axes.unicode_minus": False,
    }
)

HERE = Path(__file__).resolve().parent
STATIC_DIR = HERE / "_static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def generate_frequency_domain_signal(nfreq: int, Np: int = 4, seed: int = 17, sigma: float = 0.55) -> np.ndarray:
    k = np.arange(nfreq + 1)
    mu = np.log(Np) + sigma**2

    mag = np.zeros_like(k, dtype=float)
    kk = k[1:]
    mag[1:] = (1 / (kk * sigma * np.sqrt(2 * np.pi))) * np.exp(-(np.log(kk) - mu) ** 2 / (2 * sigma**2))

    rng = np.random.default_rng(seed)
    phase = rng.uniform(-np.pi, np.pi, size=nfreq + 1)
    phase[0] = 0.0
    phase[-1] = 0.0

    X_fd_1d = mag * np.exp(1j * phase)
    X_fd_1d[0] = np.real(X_fd_1d[0])
    X_fd_1d[-1] = np.real(X_fd_1d[-1])

    return X_fd_1d[:, None]


def build_pseudospectral_figure(
    f1: float,
    nfreq: int,
    X_fd: np.ndarray,
    nsub: int = 10,
    fig=None,
    ax=None,
    zlim=None,
    zero_freq: bool = False,
    title_str: str | None = None,
    plot_collocation_points: bool = True,
):
    freqs = wot.frequency(f1, nfreq, zero_freq=zero_freq)
    t = wot.time(f1, nfreq)
    t_full = np.insert(t, 2 * nfreq, 1 / freqs[1])
    t_sub = wot.time(f1, nfreq, nsubsteps=nsub)

    x_td = wot.fd_to_td(X_fd, f1=f1, nfreq=nfreq, zero_freq=zero_freq).squeeze()
    x_td_interp = wot.fd_to_td(X_fd, f1=f1, nfreq=nfreq, nsubsteps=nsub, zero_freq=zero_freq).squeeze()

    xk_td = []
    for k in range(X_fd.shape[0]):
        Xk = np.zeros_like(X_fd)
        Xk[k, 0] = X_fd[k, 0]
        xk_td.append(wot.fd_to_td(Xk, f1=f1, nfreq=nfreq, nsubsteps=nsub, zero_freq=zero_freq).squeeze())
    xk_td = np.array(xk_td)

    if fig is None or ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax.cla()

    if zlim is not None:
        ax.set_zlim(zlim[0], zlim[1])

    dxaxis = 0.35
    y_xax_maj = -dxaxis * freqs[-1]
    y_xax_min = (1 + dxaxis) * freqs[-1]
    ax.set_ylim(y_xax_maj, y_xax_min * 0.8)

    dyaxis = 0.35
    t_freq_ax_mag = (1 + dyaxis) * t[-1]
    t_freq_ax_ph = -t_freq_ax_mag + t[-1]
    ax.set_xlim(-0.02 * t[-1], t_freq_ax_mag)

    ax.xaxis.pane.set_visible(False)
    ax.yaxis.pane.set_visible(False)
    ax.zaxis.pane.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()

    xmin, xmax = -0.05 * t[-1], 1.05 * t[-1]

    ax.plot([xmin, xmax], [y_xax_maj, y_xax_maj], [0, 0], color="k", lw=1)
    ax.plot(t_full, np.full_like(t_full, y_xax_maj), np.zeros_like(t_full), color="b", linestyle="None", alpha=0.4, marker="|", markersize=4)
    ax.plot([xmin, xmax], [y_xax_min] * 2, [0, 0], color="k", lw=1)
    ax.plot(t_full, np.full_like(t_full, y_xax_min), np.zeros_like(t_full), color="b", linestyle="None", alpha=0.4, marker="|", markersize=4)

    ax.text(1 / freqs[1], y_xax_min, 0, r"$T_f=\frac{1}{f_1}$", ha="left", va="center")
    if zero_freq:
        ax.text(0, y_xax_min, 0, r"0", ha="left", va="bottom")
    ax.text(t[1], y_xax_min, 0, r"$\Delta t = \frac{1}{2 n_\mathrm{fq} f_1}$", "x", ha="left", va="center")
    ax.text((t[-1] - t[1]) / 2, y_xax_maj, 0, r"$x_t$", "x", ha="center", va="top")

    ax.plot([t_freq_ax_mag] * 2, np.array([-0.05, 1.05]) * freqs[-1], [0, 0], color="k", lw=1)
    ax.plot(np.full_like(freqs, t_freq_ax_mag), freqs, np.zeros_like(freqs), color="k", linestyle="None", alpha=0.4, marker="|", markersize=4)
    ax.plot([t_freq_ax_ph] * 2, np.array([-0.05, 1.05]) * freqs[-1], [0, 0], color="k", lw=1)
    ax.plot(np.full_like(freqs, t_freq_ax_ph), freqs, np.zeros_like(freqs), color="k", linestyle="None", alpha=0.4, marker="|", markersize=4)

    ax.plot(np.full_like(freqs, t_freq_ax_mag), freqs, np.abs(X_fd.squeeze()), color="k", lw=1.0, linestyle=":")
    z_ph_ax = zlim[1] / 3 if zlim is not None else 1.0
    ax.plot(np.full_like(freqs, t_freq_ax_ph), freqs, z_ph_ax, color="k", lw=0.7, linestyle="--")
    ax.plot(np.full_like(freqs, t_freq_ax_ph), freqs, -z_ph_ax, color="k", lw=0.7, linestyle="--")

    ax.text(t_freq_ax_mag, freqs[1], 0, r"$f_1$", ha="left", va="top")
    ax.text(t_freq_ax_mag, freqs[-1], 0, r"$n_\mathrm{fq} f_1$", ha="left", va="top")
    ax.text(t_freq_ax_mag * 1.1, freqs[-1] / 2, 0, r"$|X_k|$", "y", ha="center", va="top")
    ax.text(t_freq_ax_ph - 0.1 * t_freq_ax_mag, freqs[-1] / 2, z_ph_ax, r"$\measuredangle X_k$", "y", ha="center", va="top")

    for k in range(nfreq + 1):
        y = np.full_like(t_sub, freqs[k])
        (ln,) = ax.plot(t_sub, y, xk_td[k], lw=1.0, alpha=0.7)
        ax.plot([t_freq_ax_mag] * 2, [freqs[k]] * 2, [0, np.abs(X_fd[k, 0])], color=ln.get_color(), lw=1.0)
        ax.scatter(t_sub[-1], freqs[k], xk_td[k][-1], color=ln.get_color(), s=5, facecolors="none")
        ax.scatter(t_freq_ax_ph, freqs[k], np.angle(X_fd[k, 0]) / np.pi * z_ph_ax, color=ln.get_color(), s=6)
        ax.plot([t[0], t[-1]], [freqs[k]] * 2, [0, 0], color="b", lw=0.4, alpha=0.25, linestyle="--")
        ax.plot([t_freq_ax_ph] * 2, [freqs[k]] * 2, [-z_ph_ax, z_ph_ax], color=ln.get_color(), linestyle="--", lw=0.7)
        if plot_collocation_points:
            (clpts,) = ax.plot(
                t,
                np.full_like(t, freqs[k]),
                xk_td[k, ::nsub],
                color="b",
                linestyle="None",
                marker="s",
                markerfacecolor="None",
                markersize=1,
                alpha=0.7,
            )
            clpts.set_zorder(1)

    for n in t:
        ax.plot([n] * 2, [freqs[0], freqs[-1]], [0, 0], color="b", lw=0.4, alpha=0.25, linestyle="--")

    ax.plot(t, np.full_like(t, y_xax_maj), x_td, color="k", lw=1, linestyle="--")
    ax.plot(t, np.full_like(t, y_xax_maj), x_td, color="b", linestyle="None", marker="s", markerfacecolor="None", markersize=3)
    ax.plot(t_sub, np.full_like(t_sub, y_xax_maj), x_td_interp, color="k", lw=0.7, linestyle="-.", alpha=0.7)

    if title_str is not None:
        ax.set_title(title_str, pad=10)

    plt.tight_layout()
    return fig, ax


def animate_xopt_history(
    x_opt: np.ndarray,
    f1: float,
    nfreq: int,
    outfile: Path,
    fps: int = 1,
    nsub: int = 10,
    zlim: tuple[float, float] | None = None,
    title_prefix: str = "WecOptTool Tutorial 1:\nWaveBot electrical power\nIteration:",
    dpi: int = 150,
):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def row_to_Xfd(row):
        x_opt_mat = row[:, None]
        X_fd = np.array(wot.real_to_complex(x_opt_mat, zero_freq=True))
        X_fd[0, 0] = np.real(X_fd[0, 0])
        X_fd[-1, 0] = np.real(X_fd[-1, 0])
        return X_fd

    def update(i):
        X_fd = row_to_Xfd(x_opt[i, :])
        title_str = rf"{title_prefix} {i+1}/{x_opt.shape[0]}" "\n" rf"$f_1={f1:.4g}\ \mathrm{{Hz}},\ n_\mathrm{{fq}}={nfreq}$"
        build_pseudospectral_figure(
            f1=f1,
            nfreq=nfreq,
            X_fd=X_fd,
            zero_freq=True,
            nsub=nsub,
            fig=fig,
            ax=ax,
            zlim=zlim,
            title_str=title_str,
            plot_collocation_points=False,
        )
        return []

    anim = FuncAnimation(fig, update, frames=x_opt.shape[0], interval=int(1000 / fps), blit=False)
    anim.save(str(outfile), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def make_equivalence_figure():
    f1 = 0.02
    nfreq = 5
    X_fd = generate_frequency_domain_signal(nfreq, Np=3, seed=9)
    fig, _ax = build_pseudospectral_figure(f1=f1, nfreq=nfreq, X_fd=X_fd, zero_freq=True, nsub=10, zlim=(-0.7, 0.7))
    fig.savefig(STATIC_DIR / "pseudospectral_equivalence.png", dpi=300)
    plt.close(fig)


def make_wavebot_animation():
    # The WaveBot animation was generated from iterate_history_wavebot_ex1_elec.npz
    # derived from Tutorial 1 (objective electrical power) optimization history.
    npz_path = HERE / "iterate_history_wavebot_ex1_elec.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {npz_path}\n"
            "Place iterate_history_wavebot_ex1_elec.npz in docs/source/ before running this script."
        )

    data = np.load(npz_path, allow_pickle=True)
    f1_sol = float(data["f1"])
    nfreq_sol = int(data["nfreq"])
    x_opt = data["x_opt"]
    data.close()

    animate_xopt_history(
        x_opt=x_opt,
        f1=f1_sol,
        nfreq=nfreq_sol,
        outfile=STATIC_DIR / "wavebot_ex_xopt_iterates.gif",
        fps=1,
        nsub=10,
        zlim=(-750, 750),
        dpi=150,
    )


def main():
    make_equivalence_figure()
    make_wavebot_animation()
    print("Wrote:")
    print(f"  {STATIC_DIR / 'pseudospectral_equivalence.png'}")
    print(f"  {STATIC_DIR / 'wavebot_ex_xopt_iterates.gif'}")


if __name__ == "__main__":
    main()