"""Experiment runtime helpers for Mental Navigation training.

Provides the per-trial loss/forward functions, the inner training loop,
and the evaluation routine called from train.py.
"""
import numpy as np
import torch
import torch.nn.functional as F

from lib.rl_utils import reinforce, get_reward
from lib.utils import calculate_slope


def convert_velocity(velocity):
    """
    Convert velocity input to appropriate format.

    Args:
        velocity (torch.Tensor or None): Velocity tensor.

    Returns:
        int or torch.Tensor: Converted velocity.
    """
    if velocity is None:
        velocity = 1
    elif velocity.shape[1] == 1:
        velocity = int(velocity.round())
    else:
        velocity = velocity.argmax(-1) + 1 # Remove the velocity = 0
    return velocity


def get_action(logit):
    """Map a 3-class logit to an action in {-1, 0, +1}.

    Args:
        logit: Tensor with last-dim 3. We always run args.type='cls'
            with num_class=3, so the action is `argmax - 1`.

    Returns:
        int: Action in {-1, 0, +1}.
    """
    return (logit.argmax() - 1).item()


def get_loss(logits, gt, recon_states, states, velocities, args, loss_fn, loss_recon):
    """
    Calculate the loss for the model.

    Args:
        logits (torch.Tensor): Logit outputs from the model.
        gt (torch.Tensor): Ground truth actions.
        recon_states (torch.Tensor): Reconstructed states.
        states (torch.Tensor): Actual states.
        velocities (torch.Tensor): Predicted velocities.
        args (argparse.Namespace): Parsed command-line arguments.
        loss_fn (callable): Loss function for the task.
        loss_recon (callable): Loss function for reconstruction.

    Returns:
        tuple: Task loss, reconstruction loss, and velocity loss.
    """
    # 3-class action loss.
    task_loss = loss_fn(logits, gt)
    if args.decoder != 'none':
        # 'future' decoder (RNN_autoreg / RNN-D_autoreg) reconstructs the
        # next-step landmark image; the recon head only fires for those.
        recon_loss = loss_recon(recon_states, states)
    else:
        recon_loss = task_loss * 0
    if (not args.scaling_factor) or (type(velocities) == list and len(velocities) == 0):
        vel_loss = task_loss * 0
    else:
        # REINFORCE-style loss: -(return) * log p, with the corr+suc
        # reward filter that was always on when the scaling-factor head ran.
        reward = get_reward(logits, gt)
        prob = F.softmax(velocities, dim=-1)
        gamma = 0.9
        vel_loss = reinforce(prob, reward, gamma=gamma)
    return task_loss, recon_loss, vel_loss



def run_single_trial(args, model, env, use_cuda: bool = False, is_mental: bool = False, idx=None, is_log: bool = False):
    """
    Run a single trial in the environment.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        model: The model to be used.
        env: The environment instance.
        use_cuda (bool, optional): Flag to use CUDA.
        is_mental (bool, optional): Flag for mental trials.
        idx (int, optional): Index of the trial.
        is_log (bool, optional): Flag to log the results.

    Returns:
        tuple: Various trial outputs.
    """
    trial_length = args.trial_length
    decoder = args.decoder
    cat = args.cat_action
    use_internal_velocity = args.scaling_factor
    # TODO actual length
    LIMIT = 1000
    pred_st_tg_dist = None

    logits = []
    pred_st_tg_dists = []
    gt = []
    gt_st_tg_dists = []
    states = []
    actions = []
    recon_states = []
    hiddens = []
    base_hiddens = []
    fc_hiddens = []
    velocities = []
    values = []
    actual_velocities = []
    internal_velocities = []
    traj = []
    state, info = env.reset(idx=idx)
    
    # Test
    traj.append(info['current_pos'])
    model.reset()
    infos = []
    done = False
    begin_trial = True
    prev_action = torch.zeros((1,))
    # dist target variants (paper canonical: dist_scale=5, rev on, abs on).
    dist_scale = float(getattr(args, 'dist_scale', 5))
    use_abs = not bool(getattr(args, 'no_abs', False))
    use_rev = not bool(getattr(args, 'no_rev', False))

    if use_cuda:
        prev_action = prev_action.cuda()

#    gt_st_tg_dist = torch.LongTensor([info['dist']//12 + 6])
    gt_st_tg_dist = torch.FloatTensor([info['dist']]) #/ 12  #a/ args.num_images
    if use_cuda:
        gt_st_tg_dist = gt_st_tg_dist.cuda()
    initial_dist = gt_st_tg_dist
    ratio = dist_scale / abs(initial_dist)
    # `rev` on by default: gt = initial - current (starts at 0, ends at initial).
    # -no_rev keeps raw current distance so gt = current (starts at initial, ends at 0).
    if use_rev:
        gt_st_tg_dist = initial_dist - gt_st_tg_dist
    gt_st_tg_dist = gt_st_tg_dist * ratio
    # `abs` on by default (canonical rev+abs); -no_abs keeps signed target.
    if use_abs:
        gt_st_tg_dist = abs(gt_st_tg_dist)
    st_tg_loss = 0
    st_tg_loss_count = 0
    for step in range(trial_length):
        state = state.detach().clone()
        if use_cuda:
            state = state.cuda()
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if is_mental and decoder == 'future' and step > 0: # use imagined image
            state[:, :recon_states[-1].shape[1]] = recon_states[-1].detach()

        # go-cue is permanently disabled (args.go was always 0).
        obs = state

        if cat:
            prev_action = torch.ones((1,1)).to(obs.device) * prev_action
            obs = torch.cat((obs, prev_action), -1)
        
        logit, model_info = model(obs, is_log=is_log)
#        print(obs, logit.argmax(-1).item()-1, info['best_action'], info['current_pos'], info['target_pos'])
        if 'st_tg_dist' in model_info:
            pred_st_tg_dist = model_info['st_tg_dist']
            st_tg_loss += F.mse_loss(pred_st_tg_dist.view(-1), gt_st_tg_dist)  # dist_coeff was always 1
#            print(pred_st_tg_dist, gt_st_tg_dist, st_tg_loss)
            st_tg_loss_count += 1
        if (type(logit) is list or type(logit) is tuple):
            logit, pred_st_tg_dist = logit


        # convert action.
        action = get_action(logit)
        
        if is_log:
            logit = model_info['logit']
            hidden = model_info['hidden']
            fc_hidden = model_info['fc_hidden']
            velocity = model_info.get('velocity', None)
            value = model_info.get('value', None)
            recon_state = model_info.get('recon', None)
            hidden_base = model_info.get('hidden_base', None)
            hiddens.append(hidden.detach())
            fc_hiddens.append(fc_hidden.detach())
            if hidden_base is not None:
                base_hiddens.append(hidden_base.detach())
            if velocity is not None:
                velocities.append(velocity)
            if value is not None:
                values.append(value)
            if recon_state is not None:
                recon_states.append(recon_state)

            if decoder == 'current':
                states.append(info['next_img'][1].to(state.device).unsqueeze(0))
            elif decoder == 'future':
                states.append(info['next_img'][action + 1].to(state.device).unsqueeze(0))
            else:
                states.append(state)
        else:
            velocity = None
            # END LOG
        

        logits.append(logit)
        actions.append(action)
        gt.append(info['best_action'])
        gt_st_tg_dists.append(gt_st_tg_dist)

        # check dimensionality
        if use_internal_velocity:
            internal_velocity = convert_velocity(velocity)
            velocity = 1
        else:
            velocity = convert_velocity(velocity)
            internal_velocity = 1  # was args.default_internal_vel, always 1
        
        for i in range(velocity):
            if i == LIMIT: # manual limit for the sake of time
                break
            next_state, reward, done, info = env.step(action, internal_velocity)
            state = next_state

        actual_velocities.append(velocity)
        internal_velocities.append(internal_velocity.item() if type(internal_velocity) is not int else internal_velocity)
        infos.append(info)
        prev_action = action
        if done or step == trial_length - 1:
            break
        traj.append(info['current_pos'])

        gt_st_tg_dist = torch.FloatTensor([info['dist']]) # / 12  #a/ args.num_images
        if use_cuda:
            gt_st_tg_dist = gt_st_tg_dist.cuda()
        
        # Same transformation as t=0: rev on unless -no_rev, abs on unless -no_abs.
        if use_rev:
            gt_st_tg_dist = initial_dist - gt_st_tg_dist
        gt_st_tg_dist = gt_st_tg_dist * ratio
        if use_abs:
            gt_st_tg_dist = abs(gt_st_tg_dist)
        # END FOR LOOP
    
    logits = torch.cat(logits)
    gt = torch.Tensor(gt).to(logits.device).long() + 1
#    gt = torch.Tensor(gt_st_tg_dists).to(logits.device).float()

    if is_log:
        hiddens = torch.cat(hiddens)
        fc_hiddens = torch.cat(fc_hiddens)
        if len(base_hiddens) > 0:
            base_hiddens = torch.cat(base_hiddens)
    
        if decoder != 'none':
            states = torch.cat(states)
            # only predict current image part
            recon_states = torch.cat(recon_states)
        if len(velocities) > 0:
            velocities = torch.cat(velocities)
        if len(values) > 0:
            values = torch.cat(values)

    info['success'] = info['success'] and len(np.unique(actions)) == 2 # only move one direction and stop.
    if st_tg_loss_count > 0:
        st_tg_loss /= st_tg_loss_count
    return {
        'logits':              logits,
        'gt':                  gt,
        'states':              states,
        'recon_states':        recon_states,
        'velocities':          velocities,
        'internal_velocities': internal_velocities,
        'values':              values,
        'info':                info,
        'hiddens':             hiddens,
        'traj':                traj,
        'infos':               infos,
        'fc_hiddens':          fc_hiddens,
        'base_hiddens':        base_hiddens,
        'st_tg_loss':          st_tg_loss,
    }


def evaluate(model, env, envs, args, loss_fn, loss_recon, step_size: int, use_cuda: bool):
    """
    Evaluate the model on the given environments.

    Args:
        model: The model to be evaluated.
        env: The main environment instance.
        envs (list): List of additional environments.
        args (argparse.Namespace): Parsed command-line arguments.
        loss_fn (callable): Loss function for the task.
        loss_recon (callable): Loss function for reconstruction.
        step_size (int): Step size for the environment.
        use_cuda (bool): Flag to use CUDA.

    Returns:
        tuple: Run information, logs, task environment, and step count.
    """
    k = 0
    run_info = {}
    logs = {}
    splits = ['train', 'test']
    modes = ['visual', 'mental'] #[:1] # TODO change
    for split in splits:
        for mode in modes:
            name = mode + '/' + ('seen/' if split == 'train' else 'unseen/')
            name += f'{len(envs)}/'
            print(name, env.target_found)
            with torch.no_grad():
                _logs, _info = run_epoch(split, model, env, args, loss_fn, loss_recon, step_size, name, use_cuda, is_mental=mode=='mental')
            logs.update(_logs)
            run_info.update(_info)

    if args.grid_cells:
        task_env, k = env.get_env()
    else:
        task_env = env

    for e, _env in enumerate(envs):
        if args.grid_cells:
            env.set_env(_env, k=1) # 1 is just for preventing remapping
        else:
            env = _env
        for split in splits:
            for mode in modes:
                name = mode + '/' + ('seen/' if split == 'train' else 'unseen/')
                name += f'{e}/'
                print(name, env.target_found)
                with torch.no_grad():
                    _logs, _info = run_epoch(split, model, env, args, loss_fn, loss_recon, step_size, name, use_cuda, is_mental=mode=='mental')
                logs.update(_logs)
                run_info.update(_info)
    return run_info, logs, task_env, k



def run_epoch(mode: str, model, env, args, loss_fn, loss_recon, step_size: int, prefix: str, use_cuda: bool, is_mental: bool = False):
    """
    Run an epoch for training or evaluation.

    Args:
        mode (str): Mode of operation ('train' or 'test').
        model: The model to be used.
        env: The environment instance.
        args (argparse.Namespace): Parsed command-line arguments.
        loss_fn (callable): Loss function for the task.
        loss_recon (callable): Loss function for reconstruction.
        step_size (int): Step size for the environment.
        prefix (str): Prefix for logging.
        use_cuda (bool): Flag to use CUDA.
        is_mental (bool, optional): Flag for mental trials.

    Returns:
        tuple: Logs and information from the epoch.
    """
    lengths = []
    dists = []
    task_losses = []
    recon_losses = []
    vel_losses = []
    optimal_steps = []
    success = []

    trajs = []
    actions = []
    goals = []
    hiddens = []
    base_hiddens = []
    fc_hiddens = []
    unidirection = []
    logitss = []
    internal_velocities = []
    values = []

    ps = []
    gs = []
    imgs = []

    env.set_mental(is_mental)
    env.set_mode(mode)

    for i in range(len(env)):
        out = run_single_trial(args, model, env, use_cuda, idx=i, is_log=True)
        logits       = out['logits']
        gt           = out['gt']
        states       = out['states']
        recon_states = out['recon_states']
        velocities   = out['velocities']
        internal_vel = out['internal_velocities']
        value        = out['values']
        info         = out['info']
        hidden       = out['hiddens']
        traj         = out['traj']
        infos        = out['infos']
        fc_hidden    = out['fc_hiddens']
        base_hidden  = out['base_hiddens']
        task_loss, recon_loss, vel_loss = get_loss(logits, gt, recon_states, states, velocities, args, loss_fn, loss_recon)
        dists.append(info['dist'])
        lengths.append(len(logits))

        task_losses.append(task_loss.item() / len(logits))
        recon_losses.append(recon_loss.item() / len(logits))
        vel_losses.append(vel_loss.item() / len(logits))
        values += value
        internal_velocities += internal_vel
#        print(info['success'])
        # TODO fix this
        optimal_step = abs(info['start_pos'] - info['target_pos']) / step_size # this can be wrong in random_matrix environment
        l = lengths[-1]
        action = logits.argmax(dim=-1) -1
        if action[-1] == 0:
            l -= 1

        optimal_step = l / optimal_step
        optimal_steps.append(optimal_step)
        suc = info['success'] #and optimal_step == 1
        success.append(float(suc)) #info['success'])

        traj = np.divide(traj, step_size).astype(int)
        trajs.append(traj)
        goals.append(np.divide(info['target_pos'], step_size).astype(int))
        actions.append(action.cpu().numpy())
        unidirection.append(len(np.unique(actions[-1])) == 2 and actions[-1].prod() == 0)
        hiddens.append(hidden.cpu())
        fc_hiddens.append(fc_hidden.cpu())
        base_hiddens.append(base_hidden)
        logitss.append(logits.cpu())

        _optimal_step = abs(info['start_pos'] - info['target_pos']) / step_size # this can be wrong in random_matrix environment

        if len(infos) > 0:
            if 'current_img' in infos[0]:
                current_img = np.concatenate([e['current_img'] for e in infos])
                target_img = np.concatenate([e['target_img'] for e in infos])
                imgs.append(np.concatenate((current_img, target_img), axis=-1))
            if 'current_p' in infos[0]:
                current_p = np.concatenate([e['current_p'] for e in infos])
                current_g = np.concatenate([e['current_g'] for e in infos])
                if 'target_p' in infos[0]:
                    target_p = np.concatenate([e['target_p'] for e in infos])
                    ps.append(np.concatenate((current_p, target_p), axis=-1))
                    target_g = np.concatenate([e['target_g'] for e in infos])
                    gs.append(np.concatenate((current_g, target_g), axis=-1))
                else:
                    ps.append(current_p)
                    gs.append(current_g)
    
    mean_dist = np.mean(dists)
    mean_length = np.mean(lengths)
    mean_task_loss = np.mean(task_losses)
    mean_recon_loss = np.mean(recon_losses)
    mean_vel_loss = np.mean(vel_losses)
    mean_optimal_step = np.mean(optimal_steps)
    mean_acc = np.mean(success)
    mean_internal_vel = np.mean(internal_velocities)
    var_internal_vel = np.var(internal_velocities)

    coeff, intercept, r2 = calculate_slope(trajs, goals)


    logs = {
            f'{prefix}task_loss': mean_task_loss,
            f'{prefix}recon_loss': mean_recon_loss,
            f'{prefix}velocity_loss': mean_vel_loss,
            f'{prefix}length': mean_length,
            f'{prefix}dist': mean_dist,
            f'{prefix}optimal_step': mean_optimal_step,
            f'{prefix}accuracy': mean_acc, 
            f'{prefix}internal_vel': mean_internal_vel,
            f'{prefix}var_internal_vel': var_internal_vel,
            f'{prefix}slope_coeff': coeff,
            f'{prefix}slope_intercept': intercept,
            f'{prefix}slope_r2': r2,
            }
    mean_dist = '{:.3}'.format(mean_dist)
    mean_length = '{:.3}'.format(mean_length)
    mean_task_loss = '{:.3}'.format(mean_task_loss)
    mean_recon_loss = '{:.3}'.format(mean_recon_loss)
    mean_vel_loss = '{:.3}'.format(mean_vel_loss)
    mean_optimal_step = '{:.3}'.format(mean_optimal_step)
    success_rate = '{:.3}'.format(mean_acc)
    mean_internal_vel = '{:.3}'.format(mean_internal_vel)

 
    print(f'\t[{prefix}] Task Loss: {task_losses[-1]} ({mean_task_loss})\t' \
            f'Velocity Loss: {vel_losses[-1]} ({mean_vel_loss})\t' \
            f'Recon Loss: {recon_losses[-1]} ({mean_recon_loss})')
    print(f'\tStart {info["start_pos"]} Current {info["current_pos"]}, Target {info["target_pos"]}\t' \
            f'\tSUC: {success[-1]} ({success_rate})\t' \
            f'Length: {len(logits)} ({mean_length})\tLength / Optimal {optimal_step:.3f} ({mean_optimal_step})\tDist: {info["dist"]} ({mean_dist})\t' \
            f'V_internal {mean_internal_vel} (var: {var_internal_vel})')
    info = {f'{prefix}hidden': hiddens,
            f'{prefix}fc_hidden': fc_hiddens,
            f'{prefix}base_hidden': base_hiddens,
            f'{prefix}logits': logitss,
            f'{prefix}suc': success,
            f'{prefix}actions': actions,
            f'{prefix}unidirection': unidirection,
            f'{prefix}trajs': trajs,
            f'{prefix}goals': goals,
            f'{prefix}imgs': imgs,
            f'{prefix}gs': gs,
            f'{prefix}ps': ps,
            }
    if len(velocities) > 0:
        info[f'{prefix}velocities'] = velocities


    return logs, info


