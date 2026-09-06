"""Generate Fig. 2 e/f/g panels from the anonymized dataset (sub_01..sub_07.mat).

Python port of the MATLAB figure code, matched to the published figure's
styling (colors, geometry in points, fonts, tick scheme). Panels:
  e - visual navigation: learning curves (all subjects + mean) and an
      example-subject scatter per environment
  f - mental navigation: same layout
  g - avoidance of catastrophic forgetting (per-subject lines + mean +- sd)

Data fields per subject .mat (one value per trial):
  talong    true (target) vector duration, s
  tplong    produced vector duration, s
  trainlong 1 = visual navigation (vnav), 0 = mental navigation (mnav)
  env       environment index (1-3)
  sessionlong session index (0-5)
  (plus rtlong, curr, targ, att, masklong, blocklong, seq, seenonly,
   seenpair, condlong; see the paper for definitions)

Run:  python3 plot_figures.py     (writes PDFs + PNGs into figures/)
"""
import os
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, 'figures')

SUBJECTS = [1, 2, 3, 4, 5, 6, 7]
EG_SUB = 4                     # example subject shown in the scatters
WINDOW = 36                    # regression window (trials - 1)
SLIDE = 10                     # window step
TRLIMIT = 100                  # analyse first 100 trials of each condition
MAXEP = 5                      # epochs shown in the learning curves

# colors from the published figure
ENV_COLOR = {1: (0, 0, 0),
             2: (13 / 255, 130 / 255, 64 / 255),
             3: (106 / 255, 189 / 255, 69 / 255)}
RED = (237 / 255, 31 / 255, 36 / 255)

plt.rcParams.update({
    'font.family': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 4,
    'axes.linewidth': 0.2,
    'xtick.major.width': 0.2, 'ytick.major.width': 0.2,
    'xtick.major.size': 1.5, 'ytick.major.size': 1.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.pad': 1, 'ytick.major.pad': 1,
    'axes.labelpad': 1.5, 'axes.titlepad': 2,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})


def load_subject(sub):
    S = sio.loadmat(os.path.join(HERE, f'sub_{sub:02d}.mat'))
    return {k: S[k].ravel() for k in
            ['talong', 'tplong', 'trainlong', 'env', 'sessionlong']}


def sliding_slopes(ta, tp):
    """Regression slope of produced vs true duration in sliding windows."""
    slp = []
    for jj in range(0, len(tp) - WINDOW, SLIDE):
        slp.append(np.polyfit(ta[jj:jj + WINDOW + 1],
                              tp[jj:jj + WINDOW + 1], 1)[0])
    return np.array(slp)


def learning_curves(dat):
    """curves[env][vm] with vm 1 = vnav, 0 = mnav."""
    tr = np.arange(len(dat['env']))
    curves = {}
    for env in (1, 2, 3):
        curves[env] = {}
        for vm in (1, 0):
            cond = (dat['env'] == env) & (dat['trainlong'] == vm)
            idx = np.flatnonzero(cond)[0]
            g = cond & (tr < idx + TRLIMIT)
            curves[env][vm] = sliding_slopes(dat['talong'][g],
                                             dat['tplong'][g])
    return curves


def forgetting(dat):
    """Env-1 mnav performance grouped by the environment trained before."""
    g = (dat['env'] == 1) & (dat['trainlong'] == 0)
    starts = list(np.flatnonzero(~g[:-1] & g[1:]) + 1)
    ends = list(np.flatnonzero(g[:-1] & ~g[1:]))
    envt = [int(dat['env'][s - 1]) for s in starts]
    if g[0]:
        starts, envt = [0] + starts, [1] + envt
    if g[-1]:
        ends = ends + [len(g) - 1]
    slp = np.array([np.polyfit(dat['talong'][s:e + 1],
                               dat['tplong'][s:e + 1], 1)[0]
                    for s, e in zip(starts, ends)])
    envt = np.array(envt)
    if np.sum(envt == 1) > 2:            # drop the initial learning epoch
        envt, slp = envt[1:], slp[1:]
    return np.array([np.nanmean(slp[envt == env]) if np.any(envt == env)
                     else np.nan for env in (1, 2, 3)])


def make_ax(fig, x, y, w, h, W, H):
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return ax


def save_panel(fig, name):
    fig.savefig(os.path.join(OUTDIR, name + '.pdf'))
    fig.savefig(os.path.join(OUTDIR, name + '.png'), dpi=600)
    plt.close(fig)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {s: load_subject(s) for s in SUBJECTS}
    curves = {s: learning_curves(data[s]) for s in SUBJECTS}
    eg = data[EG_SUB]
    rng = np.random.default_rng(0)

    # layout in points (from the published figure)
    cw, ch = 22, 35          # curve axis box
    sw = sh = 26.3           # scatter axis box
    pitch, xleft = 30.9, 26
    ysc = 17
    ycv = ysc + sh + 16
    W = xleft + 2 * pitch + sw + 5
    H = ycv + ch + 10

    # ---------------- panels e (visual) and f (mental) ------------------
    for vm, name, ttl, tcol in [(1, 'fig2e_visualnav', 'Visual nav', 'k'),
                                (0, 'fig2f_mentalnav', 'Mental nav', RED)]:
        fig = plt.figure(figsize=(W / 72, H / 72))
        for env in (1, 2, 3):
            col = ENV_COLOR[env]
            ax = make_ax(fig, xleft + (env - 1) * pitch, ycv, cw, ch, W, H)
            M = np.full((len(SUBJECTS), MAXEP), np.nan)
            for si, s in enumerate(SUBJECTS):
                y = curves[s][env][vm][:MAXEP]
                M[si, :len(y)] = y
                ax.plot(np.arange(1, len(y) + 1), y, '-', color=col,
                        alpha=0.5, lw=0.4)
            ax.plot(np.arange(1, MAXEP + 1), np.nanmean(M, axis=0), '-',
                    color=col, lw=1)
            ax.set_xlim(1, MAXEP)
            ax.set_ylim(-0.1, 1.2)
            ax.set_xticks([1, 2, 4], ['', '2', '4'])
            ax.set_yticks([0, .4, .8, 1.2],
                          ['0', '', '0.8', ''] if env == 1 else [''] * 4)
            ax.text(0.5, 0.12, f'Env {env}', transform=ax.transAxes,
                    ha='center', fontsize=5, color=col)
            if env == 1:
                ax.set_ylabel('Performance\n(regr slope)', fontsize=4.5)
            if env == 2:
                ax.set_xlabel('Epoch (36 trials)', fontsize=4.5)
                ax.set_title(ttl, fontsize=5, color=tcol)

            # example-subject scatter, first TRLIMIT trials of the condition
            ax = make_ax(fig, xleft + (env - 1) * pitch, ysc, sw, sh, W, H)
            tr = np.arange(len(eg['env']))
            if vm == 1:
                idx = np.flatnonzero(eg['env'] == env)[0]
                g = (eg['env'] == env) & (eg['trainlong'] == 1) \
                    & (tr < idx + TRLIMIT)
            else:
                cond = (eg['env'] == env) & (eg['trainlong'] == 0)
                idx = np.flatnonzero(cond)[0]
                g = cond & (tr < idx + TRLIMIT)
            ax.plot([-5, 5], [-5, 5], '--', color=(.55, .55, .55), lw=0.2)
            ax.scatter(eg['talong'][g] + 0.2 * (rng.random(g.sum()) - 0.5),
                       eg['tplong'][g], s=1.2, c=[col], linewidths=0)
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_xticks([-4, 0, 4])
            ax.set_yticks([-4, 0, 4], [] if env > 1 else None)
            ax.set_aspect('equal')
            if env == 1:
                ax.set_ylabel('Produced\ndist (s)', fontsize=4.5)
            if env == 2:
                ax.set_xlabel('True distance (s)', fontsize=4.5)
        save_panel(fig, name)

    # ---------------- panel g -------------------------------------------
    catforg = np.array([forgetting(data[s]) for s in SUBJECTS])
    gw, gh = 28.5, 67
    Wg, Hg = 34 + gw + 16, 30 + gh + 20
    fig = plt.figure(figsize=(Wg / 72, Hg / 72))
    ax = make_ax(fig, 34, 28, gw, gh, Wg, Hg)
    ax.plot([1, 2, 3], catforg.T, '-', color='k', alpha=0.45, lw=0.4)
    ax.errorbar([1, 2, 3], catforg.mean(axis=0), catforg.std(axis=0, ddof=1),
                color='k', lw=1, capsize=1.5, capthick=1)
    ax.set_xlim(0.8, 3.2)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 2, 3], ['Env 1', 'Env 2', 'Env 3'],
                  rotation=45, ha='right')
    ax.set_yticks([0, .4, .8], ['0', '0.4', '0.8'])
    ax.set_ylabel('Env1 performance\n(regression slope)', fontsize=4.5)
    ax.set_xlabel('Train Environment', fontsize=5)
    ax.set_title('Avoidance of\nCatastrophic Forgetting', fontsize=5,
                 fontweight='bold')
    save_panel(fig, 'fig2g_forgetting')

    print(f'DONE. Figures saved to {OUTDIR}')


if __name__ == '__main__':
    main()
