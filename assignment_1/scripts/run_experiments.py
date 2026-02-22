"""Run experiments as a script and save plots + animations.

Usage:
    python run_experiments.py [--quick] [--outdir OUTPUT_DIR]

--quick: run with smaller N and shorter T for fast local checks.
"""
import sys
from pathlib import Path
# ensure repo root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src import fdm_schemes


def save_wave_animation(u_time_space, x, dt, out_filename='wave_animation.html', frame_step=None, dpi=150, speed=1.0):
    """Create and save an animation of a 1D wave solution.

    Prefers interactive HTML (anim.to_jshtml) when out_filename ends with .html.
    Falls back to GIF/MP4 when requested or when HTML export fails.
    """
    nt, nx = u_time_space.shape

    if frame_step is None:
        frame_step = max(1, int(nt / 200))

    frames = list(range(0, nt, frame_step))
    interval_ms = dt * 1000 * frame_step * speed

    fig, ax = plt.subplots()
    line, = ax.plot(x, u_time_space[0])
    ax.set_xlim(x.min(), x.max())
    y_margin = 0.1 * (u_time_space.max() - u_time_space.min())
    if y_margin == 0:
        y_margin = 1.0
    ax.set_ylim(u_time_space.min() - y_margin, u_time_space.max() + y_margin)
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    ax.set_title('1D wave')

    def update(i):
        line.set_ydata(u_time_space[i])
        ax.set_title(f'Time = {i*dt:.3f} s')
        return (line,)

    anim = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=True)

    # Try interactive HTML first
    if out_filename.lower().endswith('.html'):
        try:
            html = anim.to_jshtml()
            with open(out_filename, 'w', encoding='utf-8') as f:
                f.write(html)
            plt.close(fig)
            print(f"Saved interactive HTML animation to: {out_filename}")
            return
        except Exception as e:
            print(f"Interactive HTML export failed (to_jshtml): {e}. Falling back to gif/mp4.")

    # Fallback: try PillowWriter for GIF
    try:
        from matplotlib.animation import PillowWriter
        if out_filename.lower().endswith('.gif'):
            fps = max(1, int(1000 / interval_ms))
            writer = PillowWriter(fps=fps)
            anim.save(out_filename, writer=writer, dpi=dpi)
            plt.close(fig)
            print(f"Saved GIF animation to: {out_filename}")
            return
        else:
            # MP4 or other formats (requires ffmpeg)
            anim.save(out_filename, dpi=dpi)
            plt.close(fig)
            print(f"Saved animation to: {out_filename}")
            return
    except Exception as e:
        # Final fallback: try default save
        try:
            anim.save(out_filename, dpi=dpi)
            plt.close(fig)
            print(f"Saved animation to: {out_filename}")
            return
        except Exception as ee:
            plt.close(fig)
            print(f"Failed to save animation: {ee}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='Run  the experiments')
    parser.add_argument('--outdir', default='../outputs', help='Directory to save figures and animations')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # plotting defaults
    base_fs = 16
    mpl.rcParams.update({
        'font.size': base_fs,
        'axes.titlesize': base_fs * 1.1,
        'axes.labelsize': base_fs,
        'xtick.labelsize': base_fs * 0.9,
        'ytick.labelsize': base_fs * 0.9,
        'legend.fontsize': base_fs * 0.9,
        'figure.figsize': (8, 4.5),
    })

    # Problem parameters (match notebook by default)
    L = 1.0
    if args.quick:
        N = 400
        T = 0.5
    else:
        N = 1000
        T = 2.0
    dx = L / (N - 1)
    dt = 0.001
    c = 1.0

    # initial conditions
    x = np.linspace(0, L, N)
    u0s = [np.sin(2 * np.pi * x),
           np.sin(5 * np.pi * x),
           np.sin(5 * np.pi * x).copy()]
    u0s[2][:int(np.ceil(1/5*(N-1)))] = 0
    u0s[2][int(np.floor(2/5*(N-1))+1):] = 0

    # run simulations
    results = []
    for i, u0 in enumerate(u0s):
        print(f"Running simulation {i+1}/3 (N={N}) ...")
        u_time_space = fdm_schemes.wave_equation_1d(u0, c, dx, dt, T)
        results.append(u_time_space)

    # save imshow of all three (time on vertical axis)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    initial_conditions = ['sin(2πx)', 'sin(5πx)', 'sin(5πx) with zero outside [0.2,0.4]']
    vmin = min(res.min() for res in results)
    vmax = max(res.max() for res in results)
    for i, res in enumerate(results):
        im = axs[i].imshow(res, aspect='auto', extent=(0, L, T, 0), vmin=vmin, vmax=vmax)
        axs[i].set_title(f'Initial condition:\n{initial_conditions[i]}')
        axs[i].set_xlabel('x')
        axs[i].set_ylabel('Time')
    cbar = fig.colorbar(im, ax=axs, location='right', label='Amplitude (u)')
    plt.suptitle('1D wave equation solution over time for different initial conditions', y=1.06)
    out_im = outdir / 'solutions_imshow.png'
    fig.savefig(out_im, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_im}")

    # stacked snapshots with color gradient like notebook
    periods = np.array([1, 1/2.5, 2])
    step_sizes = 100 * periods
    fig, axs = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True)
    stagger_offsets = [0.05, 0.00, -0.05]
    pad = -0.9
    for i, res in enumerate(results):
        period = periods[i]
        # determine step_size (smaller when quick flag is used)
        if args.quick:
            step_size = max(1, int(max(1, np.floor(step_sizes[i] / 4))))
        else:
            step_size = max(1, int(np.floor(step_sizes[i])))

        nt = res.shape[0]
        # compute maximum index for this period but cap to available time-steps
        max_idx = min(nt - 1, int(np.floor(period / (2 * dt))))
        idxs = np.arange(0, max_idx + 1, step_size)
        if idxs.size == 0:
            idxs = np.array([0], dtype=int)
        times = idxs * dt

        cmap = plt.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=times.min(), vmax=times.max())

        for j, idx in enumerate(idxs):
            axs[i].plot(x, res[int(idx), :], color=cmap(norm(times[j])), lw=1)

        axs[i].set_xlabel('x')
        axs[i].set_ylabel('Amplitude (u)')
        axs[i].set_title('')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs[i], orientation='vertical', pad=0.02)
        cbar.set_label('time')

    # place staggered vertical side labels
    for i, ax in enumerate(axs):
        pos = ax.get_position()
        x_text = pos.x0 - pad
        y = pos.y0 + pos.height / 2 + stagger_offsets[i]
        label = fr'$u_0$ = {initial_conditions[i]}'
        fig.text(x_text, y, label, rotation='vertical', va='center', ha='center', fontsize=base_fs * 0.95,
                 bbox=dict(facecolor='white', alpha=0.0, edgecolor='none'))

    plt.suptitle('1D wave snapshots for different initial conditions', y=1.04)
    out_snap = outdir / 'stacked_snapshots.png'
    fig.savefig(out_snap, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_snap}")

    # animations: save three files
    for i, res in enumerate(results):
        fname_html = outdir / f'wave_animation_{i+1}.html'
        # smaller frame_step in quick mode
        frame_step = None if not args.quick else max(1, int(res.shape[0] / 100))
        save_wave_animation(res, x, dt, out_filename=str(fname_html), frame_step=frame_step)

    print('All done.')


if __name__ == '__main__':
    main()

