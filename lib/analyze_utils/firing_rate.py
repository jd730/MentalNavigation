"""Firing-rate visualizations: per-trial, aligned, shifted, and stretched."""
import os

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm


def run_firing_rate_each_trial(hiddens, new_traj, save_dir, is_onset=False, _format='pdf'):
    """Save per-trial firing-rate heatmaps + line plots for the top neurons.

    Args:
        hiddens: List of per-trial hidden-state arrays of shape ``(T_i, 1, n_units)``.
        new_traj: List of per-trial position sequences, used to label panels.
        save_dir: Output directory.
        is_onset: If True, align trials at their onset; otherwise at offset.
        _format: Figure extension. Default ``'pdf'``.

    Returns:
        None. Writes one ``firing_each_trial_<i>.<_format>`` figure per
        ranked neuron into ``save_dir``.
    """
    # convert new_traj -> start and target pairs.
    num_images = 6
    starts = [e[0]//num_images//2 for e in new_traj]
    targets = [e[-1]//num_images//2 for e in new_traj]
    min_h = min([e.min() for e in hiddens])
    max_h = max([e.max() for e in hiddens])
    save_dir = os.path.join(save_dir, 'indiv_each_trial')
    os.makedirs(save_dir, exist_ok=True)
    # hiddens: list of [L, 1, 256]
    D = hiddens[0].shape[2]
    if type(hiddens[0]) is torch.Tensor:
        hiddens = [e.cpu() for e in hiddens]

    XX = int(D ** 0.5)
    YY = int(np.ceil(D / XX))
    Ls = np.asarray([len(e) for e in hiddens])

    max_L = max(Ls)
    unique_L = np.unique(Ls)

#    cmap = cm.get_cmap('tab10')
    cmap = cm.get_cmap('inferno')
    colors = cmap.colors
#    color_dict = {e: c for e, c in zip(unique_L, colors[:len(unique_L)])}
    color_dict = {e: c for e, c in zip(range(1,num_images+1), colors[::int(len(colors)/num_images)][:num_images])}
#    colors = [[4/255, 6/255, 7/255], [53/255, 22/255, 20/255], [117/255, 31/255, 30/255], [176/255, 50/255, 47/255], [219/255, 60/255, 52/255]]
    for d in range(D):
        i = d // YY
        j = d % YY
        fig = plt.figure()
        ax = fig.gca()
        fig = plt.figure(figsize=(num_images*3, num_images*3))
        gs = gridspec.GridSpec(num_images, num_images, figure=fig)
        for h, s, t in zip(hiddens, starts, targets):
            firing_rate = h[:, 0, d]
            if is_onset:
                X = np.arange(len(firing_rate))
            else:
                X = np.arange(max_L-len(firing_rate), max_L) 
            c=color_dict[abs(s-t)]
            ax = fig.add_subplot(gs[s, t])
            ax.axhline(0, color='grey', alpha=0.5)
            ax.axhline(0.5, color='grey', alpha=0.5)
            ax.axhline(-0.5, color='grey', alpha=0.5)
            ax.axvline(num_images*2, color='grey', alpha=0.5)
            ax.axvline(num_images*4, color='grey', alpha=0.5)
            ax.axvline(num_images*6, color='grey', alpha=0.5)
            ax.axvline(num_images*8, color='grey', alpha=0.5)
            ax.axvline(num_images*10, color='grey', alpha=0.5)

            ax.plot(X, firing_rate, color=c)
            ax.set_ylim(min_h, max_h)
            ax.set_xlim(0,60)
#        max_x = X.max()
#        x = [e for e in range(0,max_x+1, 12)]
#        label = [-e for e in reversed(range(0, max_x+1, 12))]
#        plt.xticks(x, label)
        if is_onset:
            plt.savefig(f'{save_dir}/each_trial_firing_rate_onset_{d:03}.{_format}')
        else:
            plt.savefig(f'{save_dir}/each_trial_firing_rate_{d:03}.{_format}')



def run_firing_rate(hiddens, save_dir, is_onset=False, save_indiv=False, _format='pdf', options={}):
    """Save trial-averaged firing-rate heatmap + per-neuron rate traces.

    Args:
        hiddens: List of per-trial hidden-state arrays of shape ``(T_i, 1, n_units)``.
        save_dir: Output directory.
        is_onset: If True, align trials at their onset; otherwise at offset.
        save_indiv: If True, also save one trace plot per individual neuron.
        _format: Figure extension. Default ``'pdf'``.
        options: Optional knobs forwarded from the caller; supported keys
            include ``'direction'`` (``'left'``/``'right'``/``''``) for
            unidirectional trial filtering and rendering tweaks.

    Returns:
        None. Writes the aggregate ``firing.<_format>`` and (when
        ``save_indiv``) per-neuron ``firing_<i>.<_format>`` into ``save_dir``.
    """
    ignore_last = options.get('ignore_last', False)
    sigma = options.get('smoothing_sigma', 0)
    truncate = options.get('smoothing_truncate', 0)
    direction = options.get('direction', '')

    if ignore_last:
        save_dir = os.path.join(save_dir, f'indiv_ignore_last{direction}')
    else:
        save_dir = os.path.join(save_dir, f'indiv{direction}')
    if sigma > 0:
        save_dir += f'Sig{sigma}Truncate{truncate}'
    os.makedirs(save_dir, exist_ok=True)
    # hiddens: list of [L, 1, 256]
    D = hiddens[0].shape[2]
    if type(hiddens[0]) is torch.Tensor:
        hiddens = [e.cpu() for e in hiddens]

    if sigma > 0:
        hiddens = [gaussian_filter1d(v, sigma=sigma, axis=0, truncate=truncate) for v in hiddens]

    XX = int(D ** 0.5)
    YY = int(np.ceil(D / XX))
    if not save_indiv:
        fig = plt.figure(figsize=(YY*3, XX*3))
        gs = gridspec.GridSpec(XX, YY, figure=fig)

    if options.get('directional', False):
        breakpoint()
    Ls = np.asarray([len(e) for e in hiddens])

    max_L = max(Ls)
    unique_L = np.unique(Ls)

#    cmap = cm.get_cmap('tab10')
    cmap = cm.get_cmap('inferno')
    colors = cmap.colors
#    color_dict = {e: c for e, c in zip(unique_L, colors[:len(unique_L)])}
    color_dict = {e: c for e, c in zip(unique_L, colors[::int(len(colors)/len(unique_L))][:len(unique_L)])}
    colors = [[4/255, 6/255, 7/255], [53/255, 22/255, 20/255], [117/255, 31/255, 30/255], [176/255, 50/255, 47/255], [219/255, 60/255, 52/255]]
    colors = ['#EC2526', '#BF2146', '#7D287D', '#4D429A', '#3C54A4']
    
    if ignore_last:
        color_dict = { e-2:c for e ,c in zip(unique_L, colors)}
    else:
        color_dict = { e:c for e ,c in zip(unique_L, colors)}
    for d in range(D):
        i = d // YY
        j = d % YY
        if save_indiv:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.add_subplot(gs[i, j])

        for l in unique_L:
            targets = np.nonzero(l == Ls)[0]
            if ignore_last:
                firing_rates = np.stack([hiddens[t][:-2, 0, d] for t in targets])
            else:
                firing_rates = np.stack([hiddens[t][:, 0, d] for t in targets])
            firing_rate = firing_rates.mean(0)
            std = firing_rates.std(0)
            ste = std / (len(firing_rates)**0.5)

            if is_onset:
                X = np.arange(len(firing_rate))
            else:
                X = np.arange(max_L-len(firing_rate), max_L) 

            peak_idx = scipy.signal.find_peaks(firing_rate)[0]
            try:
                c=color_dict[len(firing_rate)]
            except Exception:
                c='black'
#            for pi in peak_idx[::-1]:
#                ax.text(X[pi], firing_rate[pi], len(firing_rate) - pi, color=c)
            ax.plot(X, firing_rate, color=c)
            ax.fill_between(X, firing_rate-ste, firing_rate+ste, color=c, alpha=0.1)
    # change range to -24.
        max_x = X.max()
        x = [e for e in range(0,max_x+1, 12)]
        label = [-e for e in reversed(range(0, max_x+1, 12))]
        plt.xticks(x, label)
        if save_indiv:
            if is_onset:
                plt.savefig(f'{save_dir}/firing_rate_onset_{d:03}.{_format}')
            else:
                plt.savefig(f'{save_dir}/firing_rate_{d:03}.{_format}')
            plt.close(fig)  # 512+ per-neuron figs otherwise pile up and OOM
    if not save_indiv:
        if is_onset:
            plt.savefig(f'{save_dir}/firing_rate_onset.{_format}')
        else:
            plt.savefig(f'{save_dir}/firing_rate.{_format}')
        plt.close()


def run_firing_rate_shifted(hiddens, save_dir, save_indiv=False, _format='pdf', options={}):
    """
    Standalone function (independent of run_visualization) that plots firing rates
    per distance condition with horizontal + vertical shifts applied so all curves
    align as closely as possible.

    For each neuron d independently, tries all possible integer X-shifts (lags) and
    picks the one that minimises MSE after the closed-form optimal Y-shift:
        v_opt = mean(reference_overlap) - mean(curve_overlap)
        MSE   = mean((reference_overlap - curve_overlap - v_opt)^2)

    Three alignment options are saved in separate subdirectories:
      option1/ — each distance aligned independently to the longest-distance curve only.
      option2/ — sequential: align second-longest to longest, then third-longest to the
                 consensus of longest+second, and so on from long to short.
      option3/ — Procrustes (iterative): build consensus from all distances, re-align
                 each to consensus, repeat until lags stop changing.
    """
    ignore_last = options.get('ignore_last', False)
    sigma       = options.get('smoothing_sigma', 0)
    truncate    = options.get('smoothing_truncate', 0)
    ref_dist    = options.get('ref_dist', None)
    max_iter    = options.get('max_iter', 20)

    out_dir = os.path.join(save_dir, 'indiv_ignore_last_shifted' if ignore_last else 'indiv_shifted')
    if sigma > 0:
        out_dir += f'Sig{sigma}Truncate{truncate}'
    os.makedirs(out_dir, exist_ok=True)

    D = hiddens[0].shape[2]
    _hiddens = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in hiddens]
    if sigma > 0:
        _hiddens = [gaussian_filter1d(v, sigma=sigma, axis=0, truncate=truncate) for v in _hiddens]

    Ls       = np.asarray([len(e) for e in _hiddens])
    unique_L = np.unique(Ls)
    max_L    = int(max(Ls))

    colors     = ['#EC2526', '#BF2146', '#7D287D', '#4D429A', '#3C54A4']
    color_dict = {(l - 2 if ignore_last else l): c for l, c in zip(unique_L, colors)}

    # Mean firing rate per distance: {L: ndarray [T, D]}
    mean_rates = {}
    for l in unique_L:
        idxs  = np.nonzero(l == Ls)[0]
        stack = np.stack([_hiddens[i][:-2, 0, :] if ignore_last else _hiddens[i][:, 0, :] for i in idxs])
        mean_rates[l] = stack.mean(0)  # [T, D]

    ref_L = ref_dist if ref_dist is not None else unique_L[-1]

    # ------------------------------------------------------------------ helpers
    def _build_consensus(lags, v_offs, curves):
        """Place all curves at their lags+v_offs on [0, max_L], return (consensus, valid)."""
        cons  = np.zeros(max_L)
        count = np.zeros(max_L, dtype=int)
        for l in unique_L:
            c   = curves[l] + v_offs[l]
            lag = lags[l]
            t0  = max(0,     lag)
            t1  = min(max_L, lag + len(c))
            c0  = max(0,    -lag)
            n   = t1 - t0
            if n > 0:
                cons[t0:t1] += c[c0:c0 + n]
                count[t0:t1] += 1
        valid = count > 0
        cons[valid] /= count[valid]
        return cons, valid

    def _best_align(c, cons, valid):
        """
        Exhaustive X-shift search for curve c against a fixed consensus array.
        Returns (best_lag, best_v_off).
        v_opt = mean(cons_overlap) - mean(c_overlap)  [closed-form, minimises MSE]
        """
        T = len(c)
        best_lag, best_v, best_mse = max_L - T, 0.0, np.inf  # right-aligned default
#        for lag in range(-(T - 1), max_L):
        for lag in range(0, max_L-T):
            t0 = max(0,     lag)
            t1 = min(max_L, lag + T)
            c0 = max(0,    -lag)
            n  = t1 - t0
            if n <= 0:
                continue
            valid_seg = valid[t0:t1]
            if not valid_seg.any():
                continue
            cons_seg = cons[t0:t1][valid_seg]
            c_seg    = c[c0:c0 + n][valid_seg]
            v_opt    = cons_seg.mean() - c_seg.mean()
            mse      = np.mean((cons_seg - c_seg - v_opt) ** 2)
            if mse < best_mse:
                best_mse = mse
                best_lag = lag
                best_v   = v_opt
        return best_lag, best_v

    alignment = options.get('alignment', 'option3')  # 'option1' | 'option2' | 'option3'
    opt_dir   = os.path.join(out_dir, alignment)
    os.makedirs(opt_dir, exist_ok=True)

    # ------------------------------------------------------------------ compute shifts
    h_shifts = {l: np.zeros(D, dtype=int) for l in unique_L}
    v_shifts = {l: np.zeros(D)            for l in unique_L}
    print(f'[shifted {alignment}] computing {D} neurons ...', flush=True)
    for d in range(D):
        if d % 50 == 0:
            print(f'  neuron {d}/{D}', flush=True)
        curves = {l: mean_rates[l][:, d] for l in unique_L}

        if alignment == 'option1':
            # Each distance independently aligned to longest only.
            ref_lag              = max_L - len(curves[ref_L])
            cons                 = np.zeros(max_L)
            valid                = np.zeros(max_L, dtype=bool)
            cons[ref_lag:max_L]  = curves[ref_L]
            valid[ref_lag:max_L] = True

            for l in unique_L[:-1]:
                if l == ref_L:
                    h_shifts[l][d] = ref_lag
                    v_shifts[l][d] = 0.0
                else:
                    lag, v = _best_align(curves[l], cons, valid)
                    h_shifts[l][d] = lag
                    v_shifts[l][d] = v
        elif alignment == 'option2':
            # Sequential from longest to shortest.
            lags   = {l: max_L - len(curves[l]) for l in unique_L}
            v_offs = {l: 0.0                    for l in unique_L}

            for l in sorted(unique_L[:-1], reverse=True):
                cons  = np.zeros(max_L)
                count = np.zeros(max_L, dtype=int)
                for l2 in unique_L:
                    if l2 <= l:
                        continue
                    c2   = curves[l2] + v_offs[l2]
                    lag2 = lags[l2]
                    t0 = max(0,     lag2)
                    t1 = min(max_L, lag2 + len(c2))
                    c0 = max(0,    -lag2)
                    n  = t1 - t0
                    if n > 0:
                        cons[t0:t1]  += c2[c0:c0 + n]
                        count[t0:t1] += 1
                valid = count > 0
                if not valid.any():
                    continue   # longest: keep right-aligned default
                cons[valid] /= count[valid]
                lag, v = _best_align(curves[l], cons, valid)
                lags[l]   = lag
                v_offs[l] = v

            for l in unique_L:
                h_shifts[l][d] = lags[l]
                v_shifts[l][d] = v_offs[l]

        else:  # option3: Procrustes iterative
            lags   = {l: max_L - len(curves[l]) for l in unique_L}
            v_offs = {l: 0.0                    for l in unique_L}

            for it in range(max_iter):
                cons, valid = _build_consensus(lags, v_offs, curves)
                new_lags, new_v_offs = {}, {}
                for l in unique_L[:-1]:
                    lag, v = _best_align(curves[l], cons, valid)
                    new_lags[l]   = lag
                    new_v_offs[l] = v
                new_v_offs[unique_L[-1]] = 0.0  # longest always v_off=0 (reference)
                new_lags[unique_L[-1]]   = 0
                converged = all(new_lags[l] == lags[l] for l in unique_L)
                lags   = new_lags
                v_offs = new_v_offs
                if converged:
                    if d % 50 == 0:
                        print(f'    converged at iter {it+1}', flush=True)
                    break

            for l in unique_L:
                h_shifts[l][d] = lags[l]
                v_shifts[l][d] = v_offs[l]
    print(f'[shifted {alignment}] done. lags for neuron 0:', flush=True)
    for l in unique_L:
        print(f'  L={l}: h_shift={h_shifts[l][0]}, v_shift={v_shifts[l][0]:.3f}', flush=True)
    print(f'[shifted {alignment}] saving plots ...', flush=True)

    # ------------------------------------------------------------------ plot & save
    XX = int(D ** 0.5)
    YY = int(np.ceil(D / XX))

    if not save_indiv:
        fig       = plt.figure(figsize=(YY * 3, XX * 3))
        gs_layout = gridspec.GridSpec(XX, YY, figure=fig)

    for d in range(D):
        if save_indiv:
            fig = plt.figure()
            ax  = fig.gca()
        else:
            ax = fig.add_subplot(gs_layout[d // YY, d % YY])

        X_all = []
        for l in unique_L:
            curve = mean_rates[l][:, d]
            lag   = h_shifts[l][d]
            v_off = v_shifts[l][d]
            X_shifted = np.arange(lag, lag + len(curve))
            c_color   = color_dict.get(l - 2 if ignore_last else l, 'black')
            lw = 1 #2.0 if l == ref_L else 1.0
            ax.plot(X_shifted, curve + v_off, color=c_color, linewidth=lw)
            X_all.append(X_shifted)

        all_x  = np.concatenate(X_all)
        x_min  = max(0, int(all_x.min()))
        x_max  = int(all_x.max()) + 1
        x_ticks = list(range(x_min, x_max, 12))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([-(x_max - 1 - t) for t in x_ticks])

        if save_indiv:
            plt.savefig(f'{opt_dir}/firing_rate_shifted_{d:03}.{_format}')
            plt.close(fig)
    if not save_indiv:
        plt.savefig(f'{opt_dir}/firing_rate_shifted.{_format}')
        plt.close()


def run_firing_rate_stretched(hiddens, save_dir, save_indiv=False, _format='pdf', options={}):
    """
    Standalone function (no shifting) that finds, per distance, the optimal X-stretch
    (sx, shared across neurons — time is physical) and per-neuron Y-stretch (sy) that
    jointly minimise MSE against a reference/consensus.

    Grid search over sx_range; for each sx the closed-form sy* = sum(r*c)/sum(c^2).

    Three alignment options, each saved to its own subdirectory:
      option1/ — each distance independently aligned to the longest-distance curve only.
      option2/ — sequential from longest to shortest: each distance aligns to the
                 consensus of all already-placed (longer) distances.
      option3/ — Procrustes (iterative): build consensus from all distances, re-optimise
                 each, repeat until sx assignments stop changing.

    For each option, three anchor versions are saved:
      onset/  — all curves anchored at t=0 (left edge).
      offset/ — all curves anchored at t=max_L (right edge).
      best/   — per-neuron, whichever anchor gave lower final MSE.
    """
    ignore_last = options.get('ignore_last', False)
    sigma       = options.get('smoothing_sigma', 0)
    truncate    = options.get('smoothing_truncate', 0)
    ref_dist    = options.get('ref_dist', None)
    sx_range    = options.get('sx_range', np.linspace(0.5, 4.0, 71))
    max_iter    = options.get('max_iter', 20)

    out_dir = os.path.join(save_dir, 'indiv_ignore_last_stretched' if ignore_last else 'indiv_stretched')
    if sigma > 0:
        out_dir += f'Sig{sigma}Truncate{truncate}'
    os.makedirs(out_dir, exist_ok=True)

    D = hiddens[0].shape[2]
    _hiddens = [e.cpu().numpy() if isinstance(e, torch.Tensor) else e for e in hiddens]
    if sigma > 0:
        _hiddens = [gaussian_filter1d(v, sigma=sigma, axis=0, truncate=truncate) for v in _hiddens]

    Ls       = np.asarray([len(e) for e in _hiddens])
    unique_L = np.unique(Ls)
    max_L    = int(max(Ls))

    colors     = ['#EC2526', '#BF2146', '#7D287D', '#4D429A', '#3C54A4']
    color_dict = {(l - 2 if ignore_last else l): c for l, c in zip(unique_L, colors)}

    # Mean firing rate per distance: {L: ndarray [T, D]}
    mean_rates = {}
    for l in unique_L:
        idxs  = np.nonzero(l == Ls)[0]
        stack = np.stack([_hiddens[i][:-2, 0, :] if ignore_last else _hiddens[i][:, 0, :] for i in idxs])
        mean_rates[l] = stack.mean(0)  # [T, D]

    ref_L = ref_dist if ref_dist is not None else unique_L[-1]

    # ------------------------------------------------------------------ helpers
    def _interpolate(c_all, T_c, sx):
        T_new = max(1, round(T_c * sx))
        if T_new == T_c:
            return c_all, T_new
        t_new  = np.linspace(0, T_c - 1, T_new)
        t_orig = np.arange(T_c, dtype=float)
        n_cols = c_all.shape[1]
        c_sx   = np.stack([np.interp(t_new, t_orig, c_all[:, i])
                           for i in range(n_cols)], axis=1)
        return c_sx, T_new

    def _place(c_sx, T_new, anchor_name):
        """Return (t0, t1, c_win) — the [0,max_L] window this curve occupies."""
        if anchor_name == 'offset':
            t0 = max(0, max_L - T_new)
            t1 = max_L
            return t0, t1, c_sx[T_new - (t1 - t0):]
        else:  # onset
            t0 = 0
            t1 = min(T_new, max_L)
            return t0, t1, c_sx[:t1 - t0]

    def _build_consensus(sxs, sys_, anchor_name):
        """Build consensus [max_L, D] and valid mask from current sx/sy assignments."""
        cons  = np.zeros((max_L, D))
        count = np.zeros(max_L, dtype=int)
        for l in unique_L:
            T_c = mean_rates[l].shape[0]
            c_sx, T_new = _interpolate(mean_rates[l], T_c, sxs[l])
            t0, t1, c_win = _place(c_sx, T_new, anchor_name)
            n = t1 - t0
            if n > 0:
                cons[t0:t1] += sys_[l] * c_win
                count[t0:t1] += 1
        valid = count > 0
        cons[valid] /= count[valid, None]
        return cons, valid

    def _best_sx_sy(l, cons, valid, anchor_name, prev_sx):
        """
        For distance l, find the sx in sx_range minimising total MSE (averaged over all
        neurons) against cons, then return (best_sx, best_sy_per_neuron).
        sy* = sum(cons*c) / sum(c^2)  per neuron [closed-form].
        """
        T_c = mean_rates[l].shape[0]
        best_sx        = prev_sx
        best_total_mse = np.inf

        for sx in sx_range:
            c_sx, T_new   = _interpolate(mean_rates[l], T_c, sx)
            t0, t1, c_win = _place(c_sx, T_new, anchor_name)
            if t1 - t0 <= 0 or not valid[t0:t1].any():
                continue
            vm       = valid[t0:t1]
            cons_seg = cons[t0:t1][vm]   # [n_v, D]
            c_seg    = c_win[vm]         # [n_v, D]
            num = (cons_seg * c_seg).sum(axis=0)
            den = (c_seg    * c_seg).sum(axis=0)
            sy  = np.where(den > 1e-12, num / den, 1.0)
            total_mse = np.mean((cons_seg - sy * c_seg) ** 2)
            if total_mse < best_total_mse:
                best_total_mse = total_mse
                best_sx        = sx

        # Compute per-neuron sy* for the chosen sx
        c_sx, T_new   = _interpolate(mean_rates[l], T_c, best_sx)
        t0, t1, c_win = _place(c_sx, T_new, anchor_name)
        if t1 - t0 > 0 and valid[t0:t1].any():
            vm       = valid[t0:t1]
            cons_seg = cons[t0:t1][vm]
            c_seg    = c_win[vm]
            num = (cons_seg * c_seg).sum(axis=0)
            den = (c_seg    * c_seg).sum(axis=0)
            best_sy = np.where(den > 1e-12, num / den, 1.0)
        else:
            best_sy = np.ones(D)

        return best_sx, best_sy

    def _final_mse(sxs, sys_, anchor_name):
        """Compute final per-neuron MSE for all distances against their joint consensus."""
        cons, valid = _build_consensus(sxs, sys_, anchor_name)
        out = {}
        for l in unique_L:
            T_c = mean_rates[l].shape[0]
            c_sx, T_new   = _interpolate(mean_rates[l], T_c, sxs[l])
            t0, t1, c_win = _place(c_sx, T_new, anchor_name)
            if t1 - t0 > 0 and valid[t0:t1].any():
                vm       = valid[t0:t1]
                cons_seg = cons[t0:t1][vm]
                c_seg    = c_win[vm]
                mse = np.mean((cons_seg - sys_[l] * c_seg) ** 2, axis=0)  # [D]
            else:
                mse = np.full(D, np.inf)
            out[l] = {'sx': sxs[l], 'sy': sys_[l], 'mse': mse}
        return out

    alignment = options.get('alignment', 'option3')  # 'option1' | 'option2' | 'option3'
    opt_dir   = os.path.join(out_dir, alignment)

    # ------------------------------------------------------------------ compute sx/sy
    # results[anchor][l] = {'sx': scalar, 'sy': [D], 'mse': [D]}
    results = {}
    print(f'[stretched {alignment}] {len(unique_L)} distances, {len(sx_range)} sx candidates, D={D}', flush=True)
    for anchor_fixed in ('onset', 'offset'):
        print(f'  anchor={anchor_fixed} ...', flush=True)
        sxs  = {l: 1.0        for l in unique_L}
        sys_ = {l: np.ones(D) for l in unique_L}

        if alignment == 'option1':
            # Each distance independently vs reference (longest) only.
            T_ref_c = mean_rates[ref_L].shape[0]
            c_ref, T_ref_new = _interpolate(mean_rates[ref_L], T_ref_c, 1.0)
            t0_r, t1_r, c_ref_win = _place(c_ref, T_ref_new, anchor_fixed)
            cons  = np.zeros((max_L, D))
            valid = np.zeros(max_L, dtype=bool)
            cons[t0_r:t1_r]  = c_ref_win
            valid[t0_r:t1_r] = True
            for l in unique_L[:-1]:
                print(f'    dist L={l}', flush=True)
                sx, sy = _best_sx_sy(l, cons, valid, anchor_fixed, sxs[l])
                sxs[l]  = sx
                sys_[l] = sy
                print(f'      best sx={sx:.3f}', flush=True)

        elif alignment == 'option2':
            # Sequential from longest to shortest.
            for l in sorted(unique_L[:-1], reverse=True):
                print(f'    dist L={l}', flush=True)
                cons  = np.zeros((max_L, D))
                count = np.zeros(max_L, dtype=int)
                for l2 in unique_L:
                    if l2 <= l:
                        continue
                    T_c2 = mean_rates[l2].shape[0]
                    c_sx2, T_new2   = _interpolate(mean_rates[l2], T_c2, sxs[l2])
                    t0_2, t1_2, c_win2 = _place(c_sx2, T_new2, anchor_fixed)
                    n2 = t1_2 - t0_2
                    if n2 > 0:
                        cons[t0_2:t1_2] += sys_[l2] * c_win2
                        count[t0_2:t1_2] += 1
                valid = count > 0
                if not valid.any():
                    print(f'      (longest — no reference, keep sx=1)', flush=True)
                    continue
                cons[valid] /= count[valid, None]
                sx, sy = _best_sx_sy(l, cons, valid, anchor_fixed, sxs[l])
                sxs[l]  = sx
                sys_[l] = sy
                print(f'      best sx={sx:.3f}', flush=True)

        else:  # option3: Procrustes iterative
            for it in range(max_iter):
                cons, valid = _build_consensus(sxs, sys_, anchor_fixed)
                new_sxs, new_sys_ = {}, {}
                for l in unique_L[:-1]:
                    sx, sy = _best_sx_sy(l, cons, valid, anchor_fixed, sxs[l])
                    new_sxs[l]  = sx
                    new_sys_[l] = sy
                new_sxs[unique_L[-1]]  = 1.0
                new_sys_[unique_L[-1]] = np.ones(D)
                converged = all(new_sxs[l] == sxs[l] for l in unique_L)
                sxs_str = ' '.join(f'L{l}:{new_sxs[l]:.2f}' for l in unique_L)
                print(f'    iter {it+1}: {sxs_str}', flush=True)
                sxs  = new_sxs
                sys_ = new_sys_
                if converged:
                    print(f'    converged at iter {it+1}', flush=True)
                    break

        results[anchor_fixed] = _final_mse(sxs, sys_, anchor_fixed)
        print(f'  anchor={anchor_fixed} done.', flush=True)

    print(f'[stretched {alignment}] saving plots ...', flush=True)

    # ------------------------------------------------------------------ plot & save
    def _plot_neuron(ax, l, d, res, anchor_name):
        c_orig = mean_rates[l][:, d:d+1]
        T_c    = len(c_orig)
        sx     = res[l]['sx']
        sy     = res[l]['sy'][d]
        c_sx, T_new   = _interpolate(c_orig, T_c, sx)
        t0, t1, c_win = _place(c_sx, T_new, anchor_name)
        X = np.arange(t0, t1)
        c_color = color_dict.get(l - 2 if ignore_last else l, 'black')
        lw = 1# 2.0 if l == ref_L else 1.0
        ax.plot(X, sy * c_win[:, 0], color=c_color, linewidth=lw)

    def _set_ticks(ax):
        x_ticks = list(range(0, max_L, 12))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([-e for e in reversed(range(0, len(x_ticks) * 12, 12))])

    XX = int(D ** 0.5)
    YY = int(np.ceil(D / XX))

    res_onset  = results['onset']
    res_offset = results['offset']

    for version in ('onset', 'offset', 'best'):
        version_dir = os.path.join(opt_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        if not save_indiv:
            fig       = plt.figure(figsize=(YY * 3, XX * 3))
            gs_layout = gridspec.GridSpec(XX, YY, figure=fig)

        for d in range(D):
            if save_indiv:
                fig = plt.figure()
                ax  = fig.gca()
            else:
                ax = fig.add_subplot(gs_layout[d // YY, d % YY])

            for l in unique_L:
                if version == 'best':
                    use_onset   = res_onset[l]['mse'][d] < res_offset[l]['mse'][d]
                    anchor_name = 'onset' if use_onset else 'offset'
                    res         = res_onset if use_onset else res_offset
                elif version == 'onset':
                    anchor_name, res = 'onset', res_onset
                else:
                    anchor_name, res = 'offset', res_offset
                _plot_neuron(ax, l, d, res, anchor_name)

            _set_ticks(ax)

            if save_indiv:
                plt.savefig(f'{version_dir}/firing_rate_stretched_{d:03}.{_format}')
                plt.close(fig)

        if not save_indiv:
            plt.savefig(f'{version_dir}/firing_rate_stretched.{_format}')
            plt.close()


