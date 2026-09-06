# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:50:59 2022

@author: murra
"""
import numpy as np
import torch
from lib.grid_cells.assoc_utils_np import train_gcpc, pseudotrain_3d_iterative, pseudotrain_3d_iterative_step, module_wise_NN, pseudotrain_Wps, pseudotrain_Wsp, gen_gbook
from lib.grid_cells.assoc_utils_np_2D import module_wise_NN_2d, path_integration_Wgg_1d
from lib.grid_cells.sensory_remap import dynamics_surprisal_remap

epsilon=0.01


def _nonlin_p(x, use_torch):
    """Plain ReLU used in the Vector-HaSH ReLU variant (assoc_utils_np.py:nonlin,
    thresh=0). Applied wherever the sign-variant thresholded place-cell
    activity. Sensory reconstruction (Wsp @ p) still uses sign; only place-cell
    activations swap."""
    if use_torch:
        return torch.relu(x)
    import numpy as _np
    return _np.maximum(x, 0.0)


def _pgate_p(x, use_relu, use_torch):
    """Dispatch: sign (original VHA) or ReLU (VHA-ReLU) on place-cell activity."""
    if use_relu:
        return _nonlin_p(x, use_torch)
    if use_torch:
        return torch.sign(x)
    import numpy as _np
    return _np.sign(x)


def grid_cell_initial(Np, Ns, lambdas, cuda=False, grid_step_size=1, use_relu=False):
    """Initialise grid-cell connection weights and codebooks for the grid wrapper.

    Hardcoded to the 1D-track case (``dimension=1``); the 2D-maze paths
    were dropped along with the ``-dimension`` CLI flag. The inner
    ``run_grid_cells`` / ``reconstruct`` machinery still carries the 2D
    plumbing because internal ``_2d=True`` call sites exercise it
    regardless of the env dimension.

    Args:
        Np: Number of place-cell units.
        Ns: Number of sensory-input units.
        lambdas: Module periods (list of ints, one per grid module).
        cuda: If True, move all returned tensors to GPU.
        grid_step_size: Discretisation step in cells per environment unit.

    Returns:
        tuple: ``(Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes,
        theta_sp, theta_ps, allocated_grids)`` — the grid-cell connectivity
        matrices, per-module codebooks, sizes, learned thresholds, and the
        place-cell allocation table consumed by ``run_grid_cells``.
    """
    nruns = 1

    Ng = np.sum(lambdas)
    Npos = np.prod(lambdas)
    module_sizes = list(lambdas)

    if cuda:
        module_gbooks = [torch.eye(i).cuda() for i in module_sizes]
    else:
        module_gbooks = [np.eye(i) for i in module_sizes]

    print("Gen Gbook")
    gbook = gen_gbook(lambdas, Ng, Npos)

    Wggs = {}
    print("Path Integration")
    if type(grid_step_size) is list:
        grid_step_size = np.array(grid_step_size)
    for direction in [-1, 0, 1]:
        Wggs[(0, direction)] = path_integration_Wgg_1d(lambdas, Ng, 0, direction * grid_step_size)

    Wpg = np.random.randn(nruns, Np, Ng)
    print("Get PBook")
    if cuda:
        gbook = torch.Tensor(gbook).cuda()
        Wpg = torch.Tensor(Wpg).cuda()
        pbook = torch.einsum('ijk,km->ijm', Wpg, gbook)
        pbook = _pgate_p(pbook, use_relu, use_torch=True)
        torch.cuda.empty_cache()
    else:
        pbook = np.einsum('ijk,km->ijm', Wpg, gbook)  # (nruns, Np, Npos)
        pbook = _pgate_p(pbook, use_relu, use_torch=False)

    gbook_flattened = gbook.reshape(Ng, int(Npos))
    pbook_flattened = pbook.reshape(nruns, Np, int(Npos))
    print("Train GCPC")
    Wgp = train_gcpc(pbook_flattened, gbook_flattened, use_torch=cuda)

    if cuda:
        ginit = torch.unsqueeze(torch.unsqueeze(gbook[:, 0], 0), -1)
        sbook_obs = torch.zeros((Ns, 1)).cuda()
        pbook_obs = torch.zeros((Np, 1)).cuda()
        torch.cuda.empty_cache()
    else:
        ginit = np.expand_dims(np.expand_dims(gbook[:, 0], 0), -1)
        sbook_obs = np.zeros((Ns, 1))
        pbook_obs = np.zeros((Np, 1))

    pbook_train = pbook_flattened[:,:,0:1]
    pinit = pbook_train
    if cuda:
        pbook_train = pbook_train.cpu().numpy()

    sbook_train = np.zeros((1, Ns, 1)) # dummy
    
    print("PseudoTrain 3D Iterative")
    Wsp, theta_sp = pseudotrain_3d_iterative(pbook_train, sbook_train, epsilon=epsilon, init_only=True)
    Wps, theta_ps = pseudotrain_3d_iterative(sbook_train, pbook_train, epsilon=epsilon, init_only=True)
    
    # Wps, Wsp = np.zeros((nruns, Np, Ns)), np.zeros((nruns, Ns, Np))
    if cuda:
        Wps = torch.Tensor(Wps).cuda()
        Wsp = torch.Tensor(Wsp).cuda()
        Wggs = {k: torch.Tensor(v).cuda() for k, v in Wggs.items()}
        theta_sp = torch.Tensor(theta_sp).cuda()
        theta_ps = torch.Tensor(theta_ps).cuda()

        torch.cuda.empty_cache()

    return Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, ginit, sbook_obs, pbook_obs, theta_sp, theta_ps

def run_grid_cells(sensory, action, Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes,
                   prev_g, sbook_obs, pbook_obs, theta_sp, theta_ps, cuda=False, is_eval=False, force_prev=False, verbose=False, allocated_grids=None, gap=0, recon_only=False, internal_velocity=1, use_relu=False):
    """Step the grid-cell network one timestep: sensory + action -> updated grid state.

    Args:
        sensory: Current sensory input vector.
        action: Current action vector (controls path-integration direction).
        Wggs, Wgp, Wpg, Wsp, Wps: Connection matrices from
            ``grid_cell_initial``.
        module_gbooks: Per-module grid codebook tensors.
        module_sizes: Per-module size list (``lambda**dimension``).
        prev_g: Previous-timestep grid state.
        sbook_obs, pbook_obs: Cached sensory- and place-codebooks for the
            current environment.
        theta_sp, theta_ps: Sparsification thresholds for sensory->place
            and place->sensory readouts.
        cuda: If True, perform the step on GPU.
        is_eval: If True, disable any in-loop weight updates.
        force_prev: If True, force the previous grid state instead of the
            sensory-driven prediction (used during go-cue stages).
        verbose: If True, print intermediate state shapes.
        allocated_grids: Optional pre-allocated grid table (saves a
            re-allocation when running many trials back-to-back).
        gap: Optional time gap forwarded to the path-integration kernel.
        recon_only: If True, skip the predict path and run only the
            sensory reconstruction pathway.
        internal_velocity: Velocity multiplier for path integration.

    Returns:
        tuple: ``(g, p, s_recon, ...)`` — the updated grid state, the
        derived place-cell activation, the reconstructed sensory input,
        and any auxiliary tensors the caller needs to chain into the next step.
    """


    # get p and g from the current sensory input



    p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                       module_gbooks, module_sizes,
#                                       Wgg@prev_g, sensory, 1, Niter=1,
                                       prev_g, sensory, 1, Niter=1,
                                       continuous=True, _2d=True,
                                       binary=True, use_torch=cuda, use_relu=use_relu)

    if recon_only:
        return p_recon, g_recon, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps

    if verbose:
        cos_sim = (sensory * s).sum() / (s.norm() * sensory.norm() + 1e-12)
        cos_dist = 1 - cos_sim

    
    use_prev = force_prev


    if type(action) is int or len(action) == 1:
        action = (0, action)
    else:
        action = tuple(action)
    
    Wgg = Wggs[action]
    if use_prev:
        p = p_recon
        g = g_recon
        
        # To check whether a new sensory input is referred to already associated input.
        if allocated_grids is not None and not is_eval:
            compare = g_recon == allocated_grids
            compare = compare.all(1).nonzero()
            while len(compare) > 0:
                print(compare)
                idx = int(compare[-1,0])
                step = gap + len(allocated_grids) - idx
                Wgg = Wggs[1]
                new_g = prev_g
                step = 1 #Niter
                new_g = Wgg @ new_g
                print(f"Duplicate grid code, Shift {step} step")
                p_recon, g_recon, s = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                                   module_gbooks, module_sizes,
                                                   new_g, sensory, 0, Niter=1,
                                                   continuous=True, _2d=True,
                                                   binary=True, use_torch=cuda,
                                                   internal_velocity=internal_velocity,
                                                   use_relu=use_relu)
                compare = g_recon == allocated_grids
                compare = compare.all(1).nonzero()

            p = p_recon
            g = g_recon
        
        p = Wpg @ g
        p = _pgate_p(p, use_relu, use_torch=cuda)

        if verbose:
            print(force_prev)
            print("USE_PREV", cos_dist, sensory.view(-1), s.view(-1), (sensory-s).max(), g.view(-1).nonzero().cpu().tolist())
    else:
        surprisal = 0
        # when init
        p, g, s = dynamics_surprisal_remap(Wgp, Wpg, Wgg, Wsp, Wps,
                                           module_gbooks, module_sizes,
                                           prev_g, sensory, surprisal, Niter=1,
                                           continuous=True, _2d=True,
                                           binary=True, use_torch=cuda,
                                           internal_velocity=internal_velocity,
                                           use_relu=use_relu)
        
    if not is_eval: # and not torch.all(sensory == 0):
        Wsp, theta_sp = pseudotrain_3d_iterative_step(Wsp, theta_sp, p[:,:,0], sensory[:,:,0], cuda)
        Wps, theta_ps = pseudotrain_3d_iterative_step(Wps, theta_ps, sensory[:,:,0], p[:,:,0], cuda)

    if not use_prev:
        p_recon, g_recon, s = reconstruct(Wgp, Wpg, Wsp, Wps,
                                           module_gbooks, module_sizes,
                                           g, sensory, 0, Niter=1,
                                           continuous=True, _2d=True,
                                           binary=True, use_torch=cuda, use_relu=use_relu)
        if verbose:
            print("USE NEW", cos_dist, sensory.view(-1), s.view(-1), (sensory-s).max(), g.view(-1).nonzero().cpu().tolist())

    if cuda:
        sbook_obs = torch.cat((sbook_obs, sensory[0]), 1)
        pbook_obs = torch.cat((pbook_obs, p[0]), 1)
    else:
        sbook_obs = np.concatenate((sbook_obs, sensory[0]), axis=1)
        pbook_obs = np.concatenate((pbook_obs, p[0]), axis=1)

        
    return p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps



def reconstruct(Wgp, Wpg, Wsp, Wps,
                                gbooks, module_sizes,
                                ginit, sinit, surprisal, Niter,
                                continuous=False, _2d=False,
                                binary=True, alpha=0, beta=0, use_torch=False,
                                use_relu=False):
    """
    Runs network of grid/place/sensory (g, p, s) layers
    Dynamics for remapping model:
        1. path integration advances g
        2. surprisal inhibits Wpg
        3. compute p based on s and g
        4. compute reconstructed g, s from p
        (repeat 3-4 for *Niter* times)
    Handles multiple runs simultaneously

    Inputs:
        Wgp, Wpg, Wgg, Wsp, Wps - np.array, weight matrices (Wyx is weights from x to y)
            contains matrices for multiples runs (except for Wgg, which is shared acrosss runs)
        gbooks, module_sizes - grid codebook & grid periods for computing g from raw activitions
            if 2D, gbooks is a list of flattened gbooks for each module
            if 1D, gbooks is a single gbook containing all modules
            (see module_wise_NN(), module_wise_NN_2d())
        ginit - np.array, grid vector before current input, size (Ng, 1)
        sinit - np.array, current sensory input, size (nruns, Ns, 1)
        surprisal - np.array, surprisal of current sensory input, size (nruns, 1, 1)
        Niter - int, number of times to run s/p -> g -> s/p loop
        continuous - bool, whether to output continuous sensory vectors 
            or binary (1/-1, from sign function)
        _2d - bool, whether env is 2D (or 1D)
        binary, alpha, beta - how to weigh inputs from s, g to p
            binary - bool, if True then p either gets only input from p (if surprisal is low)
                or only input from s (if surprisal is high)
            alpha, beta - float, weights on input from p, g to p
                not used if binary=True

    Outputs:
        p, g, s - activity vectors for p, g, s layers, 
            size (nruns, # cells in layer, 1)
    """
    s = sinit
    g = ginit
    if use_torch:
        sign_fn = torch.sign
    else:
        sign_fn = np.sign
    for i in range(Niter):
        if binary:
            p = surprisal * Wps@s + (1-surprisal) * Wpg@g
        else:
            p = alpha * Wps@s + beta*(1-surprisal) * Wpg@g
        p = _pgate_p(p, use_relu, use_torch)
        gin = Wgp@p
        if _2d:
            g = module_wise_NN_2d(gin, gbooks, module_sizes, use_torch=use_torch)
        else:
            g = module_wise_NN(gin, gbooks[:,:module_sizes[-1]], module_sizes, use_torch=use_torch)
        s = Wsp@p
        if not continuous:
            s = sign_fn(s)
#            s[s<0] = 0
    return p, g, s


