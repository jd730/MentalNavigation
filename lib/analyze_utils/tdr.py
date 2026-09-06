"""Targeted Dimensionality Reduction (Mante, Sussillo, Shenoy, Newsome 2013).

Fits a linear regression from hidden-state activations onto task variables
(start position, end position, distance, ...) and returns an orthogonalized
basis spanning the subspace where each task variable is encoded.
"""
import os
import pandas as pd

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats
import matplotlib.gridspec as gridspec


def visualize_TDR(proj, target, mask=None, names=None, average=True, color_name='viridis', filename='TDR', save_dir='./'):
    """
        Visualization of Targeted Dimensionality Reduction
        Input: proj \in (T, N, V)
            mask: torch.Tensor, np.ndarray: same shape as data (V, T, C)
    """
    
    cmap = cm.get_cmap(color_name)
    colors = cmap.colors
    T, N, V = proj.shape
#    nrow = int((V-1) ** 0.5)
#    ncol = math.ceil((V-1) / nrow)
    ncol = V-1
    nrow = 2
    fig = plt.figure(figsize=(8*ncol,6*nrow))
    axes = []
    pearson_axes = []
    for v in range(V): # for each condition
        if v == 0: # bias
            continue

        name = f'{v}th_projection' if names is None else f'{names[v-1]}'

        ax = fig.add_subplot(nrow, ncol, v)
        pearson_ax = fig.add_subplot(nrow, ncol, v + ncol)
        time = torch.arange(T)
#        if alignment == 'offset':
#            time = time.flip(0)
        if average:
            # Visualize average and standard error for each variable over the entire trajectories.
            unique_var = target[:,v].unique()
            color_interval = int(len(colors) / len(unique_var))
            xs = []
            ys = []
            # Pearson
            pearson_data = []
            for i, var in enumerate(unique_var):
                color = colors[color_interval * i]
                idx = target[:,v] == var
                y = proj[:, idx, v]
                m = mask[idx, :, 0]
                count = m.sum(0)
                valid = count > 0
                x = time[valid]
                y = y[valid]
                m = m[:,valid] # in case of trim_ends
                count = count[valid]
#                if alignment == 'offset':
#                    for j in range(y.shape[1]): # flip for each condition
#                        l = m[j] > 0
#                        y[l, j] = y[l, j].flip(0)
                
                avg_y = y.sum(-1) / (count + 1e-12)
                var_y = (y**2).sum(-1) / count - avg_y ** 2
                std_y = var_y ** 0.5
                ste_y = std_y / (count ** 0.5 + 1e-12)
                avg_y = avg_y
                ste_y = ste_y
                y_min = avg_y - ste_y
                y_max = avg_y + ste_y
                # Add pearson
                _var = torch.ones((len(avg_y)), device=avg_y.device) * var
                pearson_data.append(torch.stack((_var, x, avg_y)))

                label = f'{var.item()}'
                ax.plot(x, avg_y, marker='o', markersize=4, label=label, color=color)
                ax.fill_between(x, avg_y-ste_y, avg_y+ste_y, alpha=0.2, color=color)

            # Get Pearson Correlation for each time step.
            pearson_x = []
            pearson_p = []
            pearson_r = []
            pearson_data = torch.cat(pearson_data, 1).T # [variable, time, projection]

            for t in time:
                data_t = pearson_data[pearson_data[:, 1] == t]
                if len(data_t) <= 2:
                    continue
                r_value, p_value = stats.pearsonr(data_t[:,0], data_t[:, 2])
                pearson_x.append(t)
                pearson_p.append(p_value)
                pearson_r.append(r_value)
        else:
            color_interval = int(len(colors) / N)
            for n in range(N):
                color = colors[color_interval * n]
                if mask is not None:
                    x = time[mask[n, :, 0]==1]
                    y = proj[:, n, v][mask[n, :, 0]==1]
                else:
                    x = time
                    y = proj[:, n, v]
#                if alignment == 'offset':
#                    x = x.flip(0)
                ax.plot(x, y, marker='o', markersize=4, label=f'{target[n,v].item()}', color=color)

        ax.set_xlabel('Time')
        ax.set_ylabel('Projection')
        ax.set_title(name)
        axes.append(ax)

#            pearson_ax2 = pearson_ax.twinx()
#            pearson_ax2.plot(pearson_x, pearson_p, color='r')
#            pearson_ax2.set_ylabel('p-value')
        pearson_ax.plot(pearson_x, pearson_r,  color='b')
        pearson_ax.set_xlim(time.min(), time.max())
        pearson_ax.set_ylim(-1, 1)
        pearson_ax.set_ylabel('R')
        pearson_ax.set_title('Pearson Correlation Coefficient: ' + name)
        pearson_ax.set_xlabel('Time')

        pearson_axes.append(pearson_ax)
        ax.legend(loc=(1.04, 0.0))
#        handle, label = ax.get_legend_handles_labels()
    
    if False:
        ax = fig.add_subplot(nrow, ncol, V)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Add supertitle
    fig.suptitle(filename)
#    legend = ax.legend(handle, label, loc='center')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{filename}.pdf'))
    print(f"Save {save_dir}/{filename}.pdf")


def get_z_score(data, mask=None):
    """Z-score the rows of ``data`` (optionally over a masked subset).

    Args:
        data: 2D array of shape ``(n_samples, n_features)``; accepts
            ``np.ndarray`` or ``torch.Tensor``.
        mask: Optional boolean mask of length ``n_samples``. When given,
            mean/std are computed over the masked subset but applied to
            all samples.

    Returns:
        Same shape and type as ``data``, mean-centered and divided by
        the (unbiased) standard deviation per feature.
    """
    if type(data) is np.ndarray:
        if mask is not None:
            mask_sum = mask.sum((0, 1), keepdim=True).clip(min=1e-6)
            mean = data.sum((0, 1), keepdim=True) / mask_sum
            sq_mean = (data **2).sum((0, 1), keepdim=True) / mask_sum
            var = sq_mean - mean ** 2
        else:
            mean = data.sum((0, 1), keepdim=True)
            var = data.var((0, 1), keepdim=True)
        data = (data - mean) / (var**0.5 + 1e-12)
    elif type(data) is torch.Tensor:
        # convert to z-score
        if mask is not None:
            mask_sum = mask.sum((0, 1), keepdim=True).clip(min=1e-6)
            mean = data.sum((0, 1), keepdim=True) / mask_sum
            sq_mean = (data ** 2).sum((0, 1), keepdim=True) / mask_sum
            var = sq_mean - mean ** 2
        else:
            mean = data.sum((0, 1), keepdim=True)
            var = data.var((0, 1), keepdim=True)
        var[var < 0] *= -1 # numerical glitch
        data = (data - mean) / (var**0.5 + 1e-12)
    return data



def _TDR(data, target, mask=None, option='mask', alignment='offset', trim_ends=False, n_pca=0, reg_vec=None, target_step=None):
    if type(data) is pd.DataFrame:
        raise NotImplementedError
        
    average_first = True
    if average_first and len(target) > 1:
        target, indices = target.unique(dim=0, return_inverse=True)
        new_data = []
        new_mask = []
        for i in range(len(target)):
            new_data.append(data[indices == i].mean(0))
            new_mask.append(mask[indices == i][0])
        data = torch.stack(new_data)
        mask = torch.stack(new_mask)

    if trim_ends:
        # trim the first data
        data = data[:, 1:]
        mask = mask[:, 1:]

        # trim the last data
        for i in range(len(mask)):
            end = mask[i, :, 0].nonzero()
            if len(end) <= 1:
                return None
            end = end[-1]
            mask[i, end:] = 0
            data[i, end:] = 0
    
    original_data = data
    # Convert data to z-score
    data = get_z_score(data, mask) * mask

    if alignment == 'offset':
        # TODO align data
        for j in range(data.shape[1]): # flip for each condition
            l = mask[:, j, 0] > 0
            data[l, j] = data[l, j].flip(0)
        data = data.flip(1)
        mask = mask.flip(1)

    if target_step is not None:
        if alignment == 'offset':
            data = data[:, -(1+target_step):-target_step]
        else:
            data = data[:, target_step:target_step+1]
        mask = mask.max(dim=1)[0].unsqueeze(1)
#        mask = mask[:, target_step:target_step+1]

    if reg_vec is not None: # project to existing reg_vec.
        proj = data.permute(1, 0, 2) @ reg_vec.T
        T, N = target.shape
        if type(target) is np.ndarray:
            target = np.concatenate((np.ones((T, 1)), target), axis=-1)
        else:
            target = torch.cat((torch.ones((T, 1), device=target.device, dtype=target.dtype), target), -1)
        print(proj)
        return reg_vec, proj, target, mask
    
    if mask is not None:
        if option == 'trim_first': # deprecated
            length = mask.sum(1)[:,0].int()
            n = length.min()
            data = data[:, :n]
            mask = mask[:, :n]
        elif option == 'trim_last': # deprecated
            length = mask.sum(1)[:,0].int()
            min_length = length.min()
            data = [e[l-min_length:l] for e, l in zip(data, length)]
            if type(mask) is np.ndarray:
                data = np.stack(data)
                mask = np.ones_like(data)
            elif type(mask) is torch.Tensor:
                data = torch.stack(data)
                mask = torch.ones_like(data)
        elif option == 'extrapolate': # filling the missing cases.
            original_data = data
            original_mask = mask
            length = mask.sum(1)[:,0].int()
            n = length.min()
            # Trim by the minimum length of the trajectory.
            if n > 1:
                if alignment == 'onset':
                    data = data[:, :n]
                    mask = mask[:, :n]
                else: # offset
                    data = data[:, -n:]
                    mask = mask[:, -n:]

    if n_pca > 0:
        if type(data) is np.ndarray:
            raise NotImplementedError
        else:
            N, T, C = data.shape
            data = data.reshape(-1, C)
            pca_data = torch.pca_lowrank(data, niter=10, q=min(data.shape[0], data.shape[1]))
            U, S, V = pca_data
            data = U[:, :n_pca].view(N, T, -1)
            mask = mask[:,:,:n_pca]
            power = S[:n_pca] / S.sum()
            print(power, power.sum())
    
    T, N = target.shape
    if type(data) is np.ndarray:
        target = np.concatenate((np.ones((T, 1)), target), axis=-1)
        support = np.linalg.inv(target.T @ target) @ target.T
        coeff = support @ data.transpose(1, 0, 2)
        # TODO add numpy version
    elif type(data) is torch.Tensor:
        # concatenate the bias term
        target = torch.cat((torch.ones((T, 1), device=target.device, dtype=target.dtype), target), -1)
        # linear regression
        try:
            support = torch.inverse(target.T @ target) @ target.T # (V, N)
        except Exception:
            print("Singular")
            return None
        data = data.permute(1, 0, 2) # (N, T, C) -> (T, N, C)
        coeff = support @ data # (T, V, C)
        
        # Get max_norm coeff across te time.
        idx = coeff.norm(dim=2).argmax(0)
#        print(filename, alignment, trim_ends)
#        print(idx, coeff.norm(dim=2))
        reg_vec = torch.stack([coeff[e, i] for i, e in enumerate(idx)])
        
        # Project data into the regression vectors.
        if option == 'extrapolate':
            data = original_data.permute(1,0,2)
            mask = original_mask
        proj = data @ reg_vec.T # (T, N, C) x (C, V) -> (T, N, V)
        
    else:
        raise Exception("Type Error")
    
    return reg_vec, proj, target, mask, coeff



def TDR(data, target, mask=None, names=None, average=True, option='mask', alignment='offset', trim_ends=False, visualize=True, n_pca=0, filename='TDR', save_dir='./', reg_vec=None, target_step=None):
    """
        Targeted Dimensionality Reduction
        Input, filename=filename:
            data: torch.Tensor, np.ndarray: (N, T, C)
                - N: The number of trajectories (condition)
                - T: The number of timesteps
                - C: The number of neurons
            target: torch.Tensor, np.ndarray: (N, V)
                - V: The number of variables
            mask: torch.Tensor, np.ndarray: (N, T, C)
                - mask for valid data in case of different length.
            visualize: Bool,
            n_pca: int, the number of principal component.
            target_step: target timestep for Figure 3 (d) in neupane2023vector
        Output:
            Regression Vector 
    """
    print(data.shape)

    # boostrap
    N = 1000
    repeat = 25
    projs = []
    targets = []
    masks = []
    coeffs = []
    if len(data) > 1:
        for n in range(repeat):
            # augment data for N times
            indices = np.random.randint(0, len(data), size=N)
            _data = data[indices]
            _target = target[indices]
            _mask = mask[indices]
            out = _TDR(_data, _target, _mask, option, alignment, trim_ends, n_pca, reg_vec, target_step)
            if out is None:
                continue
            _reg_vec, _proj, _target, _mask, _coeff = out
            projs.append(_proj)
            targets.append(_target)
            masks.append(_mask)
            coeffs.append(_coeff)

    out = _TDR(data, target, mask, option, alignment, trim_ends, n_pca, reg_vec, target_step)
    if out is None:
        return None, None
    reg_vec, proj, target, mask, coeff = out
   
    if visualize:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        filename = f'{filename}_{alignment}_{option}'
        if average:
            filename += '_avg'
        if trim_ends:
            filename += '_trim_ends'
        
        visualize_neuronal_coeff(coeff, filename, save_dir, names)
        visualize_TDR(proj, target, mask, names, average=average, filename=filename, save_dir=save_dir)
        if target_step is not None and proj.shape[1] > 1:
            visualize_figure3d(proj, target, mask, filename, save_dir, projs, targets, masks)
    return reg_vec, coeff # (C, V)

def visualize_neuronal_coeff(coeff, filename, save_dir, names):
    """
        Visualize coefficient for each neuron (L, A, D).
    """
    # make 256 as array
    L, A, D = coeff.shape
    fig = plt.figure(figsize=(A * 4, 3))
    gs = gridspec.GridSpec(1, A, figure=fig)
    names = ['bias'] + names
    for a in range(A):
        ax = fig.add_subplot(gs[0, a])
        ax.axvline(L-13, color='red')
        name = names[a]
        for d in range(D):
            Y = coeff[:, a, d]
            X = np.arange(len(Y))
            ax.plot(X, Y, alpha=0.1, color='grey')
        ax.set_title(name)
    plt.savefig(os.path.join(save_dir, filename + f'_{L-13}coeff.pdf'))

def get_xy(proj, target, mask, return_error=False, N=12):
    """Per-condition mean of the projected activations along the target axis.

    Helper used by the TDR figure code to convert a per-sample projection
    onto a single axis into a per-condition trace ready for plotting.

    Args:
        proj: 1D array of per-sample projection scalars (shape ``(n_samples,)``).
        target: 1D array of per-sample target values; ``np.unique(target[mask])``
            defines the conditions binned over.
        mask: Boolean mask of length ``n_samples`` selecting valid samples.
        return_error: If True, also return per-condition standard error of
            the mean.
        N: Step scale; ``x`` is returned in units of ``unique_target / N``.

    Returns:
        Tuple ``(x, y)``: ``x`` are the unique target values divided by
        ``N``, ``y`` are the per-condition means of ``proj``. If
        ``return_error`` is True, returns ``(x, y, err)`` where ``err``
        is the per-condition SEM.
    """
    # filter based on mask.
    mask = mask.sum((1,2)) > 0
    proj = proj[0][mask]
    target = target[mask]
    # sort based on target.
    order = target[:,1].argsort()
    proj = proj[order]
    target = target[order]
    X = []
    Y = []
    Ste = []
    for t in target[:,1].unique():
        # aggregate
        indices = target == t
        _proj = proj[indices]
        mean = _proj.mean(0)
        std = _proj.std(0) 
        ste = std / (len(_proj) ** 0.5)
        X.append(t)
        Y.append(mean)
        Ste.append(ste)
    X = (np.asarray(X) /N).astype(int)
    Y = np.asarray(Y)
    if return_error:
        Ste = np.asarray(Ste)
    #    plt.plot(X, Y, marker='o', markersize=4)
        errorbar = np.stack((Y-Ste, Y+Ste))
        return X, Y, errorbar
    else:
        return X, Y

def visualize_figure3d(proj, target, mask, filename, save_dir, projs=[], targets=[], masks=[]):
    """Render a 3D TDR scatter (one trace) or paired 3D + 1D overlay (multi-trace).

    Args:
        proj: ``(n_samples, 3)`` projection onto three TDR axes for the
            primary trace.
        target: Per-sample target values that color the primary trace.
        mask: Boolean mask selecting which samples of the primary trace to plot.
        filename: Output filename stem (no extension).
        save_dir: Output directory.
        projs: Optional list of additional ``(n_samples_i, 3)`` projections,
            stacked into the same figure as overlays.
        targets: Per-overlay target arrays; must align with ``projs``.
        masks: Per-overlay boolean masks; must align with ``projs``.

    Returns:
        None. Saves ``<filename>.pdf`` into ``save_dir``.
    """
    fig = plt.figure()
    X, Y, errorbar = get_xy(proj, target, mask, True)
    for _proj, _target, _mask in zip(projs, targets, masks):
        _X, _Y = get_xy(_proj, _target, _mask)
#        plt.plot(_X, _Y, marker='o', markersize=1, color='gray', alpha=0.7)
        plt.plot(_X, _Y, color='gray', alpha=0.7)
    
    color = 'green' if 'offset' in filename else 'blue'
    plt.errorbar(X, Y, yerr=errorbar, marker='o', markersize=4, color=color)

    save_name = os.path.join(save_dir, filename + '_figure3d.pdf')
    print(Y)
    print(save_name)
    plt.savefig(save_name)



if __name__ == '__main__':

    data = torch.randn(5, 10, 3)
    target = torch.randn(10, 7)
    mask = torch.randn(5, 10, 3)
    mask[mask > 0] = 1
    mask[mask<=0] = 0
#    data = np.random.randn(5,10, 3)
#    target = np.random.randn(10, 7)
    TDR(data, target, mask)


# ---------------------------------------------------------------------------
# TDR analysis drivers
# ---------------------------------------------------------------------------

def _run_tdr(data, target, mask, names, average, option, alignment, trim_end, filename, save_dir, env_id=None, coeff_info=None, single=False, target_step=None):
    """
        Run TDR to the data from forward/backward direction separately or together.
        Input:
            data: hidden state
            target: variables
            mask: mask of data since the length of each data is different.
            average: plot averaged result for each variable.
            option: [trim_first, trim_last, extrapolate]
            alignment: [onset, offset]
            trim_end: ignore both ends since both are outlier.
            filename: filename to save the plot.
            save_dir: the location of directory to save.
            coeff_info: maximum coefficient from the previous environments at the same checkpoint.
        Output:
            forward coefficient, backward coefficient, total coefficient.
    """

    forward_coeffs = None
    backward_coeffs = None
    raw_forward_coeffs = None
    raw_backward_coeffs = None

    if target_step is not None:
        filename += str(target_step)
    # forward
    forward = target[:, -1] > 0
#    forward[1:] = False
    backward = torch.logical_not(forward)
    if single and False:
        if not os.path.exists(os.path.join(save_dir, 'indiv')):
            os.mkdir(os.path.join(save_dir, 'indiv'))
        for i in range(len(data)): 
            TDR(data[i:i+1], target[i:i+1][:,:-1], mask[i:i+1], names[:-1], average, option, alignment, trim_end, filename=filename+f'_{env_id}_{i}', save_dir=os.path.join(save_dir, 'indiv'), target_step=target_step)
    if coeff_info is not None:
        f_coeff = coeff_info['forward_coeffs']
        b_coeff = coeff_info['backward_coeffs']
        coeff = coeff_info['coeffs']
        try:
            TDR(data[forward], target[forward][:,:-1], mask[forward], names[:-1], average, option, alignment, trim_end, filename=filename+f'_{env_id}_fwd', save_dir=save_dir, reg_vec=f_coeff, target_step=target_step)
        except Exception:
            pass
        try:
            TDR(data[backward], target[backward][:,:-1], mask[backward], names[:-1], average, option, alignment, trim_end, filename=filename+f'_{env_id}_bwd', save_dir=save_dir, reg_vec=b_coeff, target_step=target_step)
        except Exception:
            pass
        return None

    coeffs, raw_coeffs = TDR(data, target, mask, names, average, option, alignment, trim_end, filename=filename, save_dir=save_dir, target_step=target_step)
    if forward.sum() > 0:
        forward_coeffs, raw_forawrd_coeffs = TDR(data[forward], target[forward][:,:-1], mask[forward], names[:-1], average, option, alignment, trim_end, filename=filename+'_fwd', save_dir=save_dir, target_step=target_step)
    if backward.sum() > 0:
        backward_coeffs, raw_backward_coeffs = TDR(data[backward], target[backward][:,:-1], mask[backward], names[:-1], average, option, alignment, trim_end, filename=filename+'_bwd', save_dir=save_dir, target_step=target_step)
    # both direction
#    coeffs = TDR(data[lengths], target[lengths], mask[lengths], names, average, option, alignment, trim_end, filename=filename+'_3rdmax_length', save_dir=save_dir)
    if forward_coeffs is None:
        print("Forward Fail")
    if backward_coeffs is None:
        print("Backward Fail")

    coeff_info = {
            'alignment': alignment,
            'option': option,
            'trim_end': trim_end,
            'forward_coeffs': forward_coeffs,
            'backward_coeffs': backward_coeffs,
            'coeffs': coeffs,
            'raw_forward_coeffs': raw_forward_coeffs,
            'raw_backward_coeffs': raw_backward_coeffs,
            'raw_coeffs': raw_coeffs
            }
    return coeff_info




def run_tdr(features, trajs, goals=None, average=True, filename='TDR', save_dir='TDR', tdr_info=None, target_step=None):
    """Run TDR (regression on task variables -> orthonormal subspace) for one env.

    Builds the per-sample target vector from ``trajs`` (start position,
    end position, signed distance, time within trial) and fits regression
    coefficients via the inner ``_run_tdr`` helper.

    Args:
        features: List of per-trial hidden-state arrays of shape
            ``(T_i, 1, n_units)``.
        trajs: Per-trial position sequences (list of lists of ints).
        goals: Optional per-trial goal positions, used for goal-conditioned
            regressors.
        average: If True, regression coefficients are averaged across
            timesteps before projection; otherwise per-timestep.
        filename: Output filename stem for the TDR figures.
        save_dir: Output directory.
        tdr_info: Optional list of prior-environment coefficient dicts;
            if given, the new env's projection is overlaid against them.
        target_step: Optional fixed timestep at which target features are
            taken (e.g. ``1`` -> condition-locked second step); ``None``
            uses the per-trial time axis.

    Returns:
        dict: Per-target coefficient bundle suitable for chaining into
        the next env via ``tdr_info``. Side effect: TDR figures (regressor
        importance heatmap, per-target projection scatter, neuronal
        coefficient maps) written into ``save_dir``.
    """
    N = len(features)
    L = max([len(e) for e in features])
    if type(features[0]) is torch.Tensor:
        data = torch.zeros([N, L] + list(tuple(features[0].shape[1:])))
    else:
        data = torch.zeros([N, L] + list(tuple(features[0].shape[1:])))
        features = [torch.from_numpy(e) for e in features]


    start_pos = np.asarray([e[0] for e in trajs])
    if goals is None or len(goals) == 0:
        end_pos = np.asarray([e[-1] for e in trajs])
    else:
        end_pos = np.asarray(goals)
    dists = np.asarray([len(e) - 1 for e in trajs])
    print(dists)
    st_indices = np.argsort(start_pos * 10 + end_pos)
    ed_indices = np.argsort(end_pos * 10 + start_pos)
    dist_indices = np.argsort(dists * 10 + start_pos)
    direction_indices = [-1 if st > ed else 1 for st, ed in zip(start_pos, end_pos)]
    indices = [start_pos, end_pos, dists, direction_indices]
#    indices = [dists, direction_indices]

    target = np.stack(indices)
    target = torch.from_numpy(target)
    mask = torch.zeros_like(data)
    for i, feature in enumerate(features):
        n = len(feature)
        data[i, :n] = feature
        mask[i, :n] = 1
    data = data.squeeze().float()
    mask = mask.squeeze().float()
    target = target.float().T
    names = ['st_idx', 'ed_idx', 'distance', 'direction']
#    names = ['distance', 'direction']
    options = ['extrapolate', 'mask', 'trim_first', 'trim_last']
    options = ['extrapolate']
    alignments = ['offset', 'onset']
    coeff_infos = []
    

    for alignment in alignments:
        for option in options:
#            for trim_end in [True, False]:
            for trim_end in [False]:
                coeff_info =  _run_tdr(data, target, mask, names, average, option, alignment, trim_end, filename, save_dir, single=alignment==alignments[0], target_step=target_step)
                coeff_infos.append(coeff_info)

                # Find the corresponding TDR maximum coefficient in the previous environments.
                if tdr_info is not None:
                    _tdr_info = []
                    for info in tdr_info:
                        _env_id = info['env_id']
                        coeff_info = info['coeff_info']
                        for _coeff_info in coeff_info:
                            # Check the condition
                            if _coeff_info['alignment'] == alignment and _coeff_info['option'] == option and _coeff_info['trim_end'] == trim_end:
                                # Project to the previous coefficients
                                _run_tdr(data, target, mask, names, average, option, alignment, trim_end, filename, save_dir, _env_id, _coeff_info, target_step=target_step)


    return coeff_infos



