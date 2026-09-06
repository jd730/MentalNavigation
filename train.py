"""Train one of the Mental Navigation baselines (see lib.models.MODELS)."""
import os
import glob
import time
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

import wandb


from lib.models import MODELS
from lib.envs import *
from lib.arguments import get_args, get_name
from lib.utils import set_seed, get_free_gpu
from lib.exp import run_single_trial, get_loss, evaluate



def run_exp(model, optimizer, env, use_cuda, args, loss_fn, loss_recon, prev_epoch=0, num_epochs=0, envs=[], save_name='', is_log=True):
    """
    Run the experiment for a given number of epochs.

    Args:
        model: The model to be trained.
        optimizer: Optimizer for training.
        env: The environment instance.
        use_cuda (bool): Flag to use CUDA.
        args (argparse.Namespace): Parsed command-line arguments.
        loss_fn (callable): Loss function for the task.
        loss_recon (callable): Loss function for reconstruction.
        prev_epoch (int, optional): Previous epoch count.
        num_epochs (int, optional): Number of epochs to run.
        envs (list, optional): List of environments.
        save_name (str, optional): Name for saving the results.
        is_log (bool, optional): Flag to log the results.

    Returns:
        tuple: Model, accuracy, run information, and logs.
    """
    use_wandb = not args.debug
    num_epochs =  prev_epoch + num_epochs
    log_interval = args.log_interval

    lengths = []
    dists = []
    task_losses = []
    recon_losses = []
    vel_losses = []
    st_tg_losses = []
    optimal_steps = []
    success = []
    values = []
    
    verbose = []
    internal_velocities = []

    max_acc = -1
    step_size = args.step_size

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # explore from the leftmost image to the rightmost image.
    if hasattr(env, 'explore_sequence'):
        env.explore_sequence()
        
    local_logs = []
    if args.eval_only:
        # Read the directory and eval all.
        target_epoch = prev_epoch = num_epochs - 1
        fnames = glob.glob(os.path.join(args.save_dir, save_name, f'*{target_epoch}*.pth'))
        logs = {}
        print(fnames)
        for path in fnames:
            if 'run_info' in path:
                continue
            # torch load
            state_dict = torch.load(path)['model']
            model.load_state_dict(state_dict)
            print(f'Load {path}')
            run_info, _logs, task_env, k = evaluate(model, env, envs, args, loss_fn, loss_recon, step_size, use_cuda)
            run_info_paths = path.split('/')
            run_info_paths[-1] = 'run_info_v2_' + run_info_paths[-1]
            run_info_path = '/'.join(run_info_paths)
            torch.save(run_info, run_info_path)
            print(run_info_path)
        return model, 0, run_info, logs
    
    st = time.time()

    model.reset()
    for epoch in range(prev_epoch, num_epochs):
        model.train()
        env.set_mode('train')
        env.set_mental(False)
        optimizer.zero_grad()
        # sample starting and target images
        out = run_single_trial(args, model, env, use_cuda, is_log=is_log)
        logits       = out['logits']
        gt           = out['gt']
        states       = out['states']
        recon_states = out['recon_states']
        velocities   = out['velocities']
        internal_vel = out['internal_velocities']
        value        = out['values']
        info         = out['info']
        st_tg_loss   = out['st_tg_loss']

        if hasattr(model, 'action_coeff'):
            action_coeff = model.action_coeff
        else:
            action_coeff = torch.tensor(1.0)
        if hasattr(model, 'scaling_coeff') and args.resolution != 1:
            scaling_coeff = model.scaling_coeff
        else:
            scaling_coeff = torch.tensor(1.0)

        if hasattr(model, 'dist_coeff'):
            dist_coeff = model.dist_coeff
        else:
            dist_coeff = torch.tensor(1.0)
        task_loss, recon_loss, vel_loss = get_loss(logits, gt, recon_states, states, velocities, args, loss_fn, loss_recon)
        loss = task_loss + recon_loss + vel_loss + st_tg_loss
        loss.backward()
        optimizer.step()
        dists.append(info['dist'])
        lengths.append(len(logits))
        task_losses.append(loss.item() / len(logits))
        recon_losses.append(recon_loss.item() / len(logits))
        vel_losses.append(vel_loss.item() / len(logits))
        if type(st_tg_loss) == torch.Tensor:
            st_tg_losses.append(st_tg_loss.item() / len(logits))
        else:
            st_tg_losses.append(0)
        internal_velocities += internal_vel
        values += value
        success.append(info['success'])
        optimal_step = abs(info['start_pos'] - info['target_pos']) / step_size
        l = lengths[-1]
        optimal_step = l / optimal_step
        optimal_steps.append(optimal_step)

        if (epoch+1) == num_epochs or (epoch+1) % log_interval == 0 or epoch == prev_epoch:
            if dist_coeff * scaling_coeff * action_coeff != 1:
                print(f'Epoch {epoch} Action Coeff: {action_coeff} Scaling Coeff: {scaling_coeff} Dist Coeff: {dist_coeff}')
            ed = time.time()
            print(ed - st)
            mean_dist = np.mean(dists)
            mean_length = np.mean(lengths)
            mean_task_loss = np.mean(task_losses)
            mean_recon_loss = np.mean(recon_losses)
            mean_vel_loss = np.mean(vel_losses)
            mean_st_tg_loss = np.mean(st_tg_losses)
            mean_optimal_step = np.mean(optimal_steps)
            mean_acc = np.mean(success)
            mean_internal_vel = np.mean(internal_velocities)
            var_internal_vel = np.var(internal_velocities)
            logs = {
                    'task_loss': mean_task_loss,
                    'st_tg_loss': mean_st_tg_loss,
                    'recon_loss': mean_recon_loss,
                    'velocity_loss': mean_vel_loss,
                    'length': mean_length,
                    'dist': mean_dist,
                    'optimal_step': mean_optimal_step,
                    'accuracy': mean_acc, 
                    'epoch': epoch,
                    'internal_vel': mean_internal_vel,
                    'var_internal_vel': var_internal_vel
                    }
            
            mean_dist = '{:.3}'.format(mean_dist)
            mean_length = '{:.3}'.format(mean_length)
            mean_task_loss = '{:.3}'.format(mean_task_loss)
            mean_recon_loss = '{:.3}'.format(mean_recon_loss)
            mean_vel_loss = '{:.3}'.format(mean_vel_loss)
            mean_optimal_step = '{:.3}'.format(mean_optimal_step)
            success_rate = '{:.3}'.format(mean_acc)
            mean_internal_vel = '{:.3}'.format(mean_internal_vel)

            if max_acc <= mean_acc:
                print('Save', epoch, max_acc, args.save_dir, args.name)
                torch.save({'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            }
                            , os.path.join(args.save_dir, args.name, 'model_best.pth')
                           )
                max_acc = mean_acc
    
            print(f'Epoch [{epoch}] Task Loss: {task_losses[-1]} ({mean_task_loss})\t' \
                    f'Vel Loss: {vel_losses[-1]} ({mean_vel_loss})\t' \
                    f'St_Tg Loss: {st_tg_losses[-1]} ({mean_st_tg_loss})\t' \
                    f'Recon Loss: {recon_losses[-1]} ({mean_recon_loss})')
            print(f'\tStart {info["start_pos"]} Current {info["current_pos"]} Target {info["target_pos"]}\t' \
                    f'SUC: {success[-1]} ({success_rate})\t' \
                    f'Length: {len(logits)} ({mean_length})\tLength / Optimal {optimal_step:.3f} ({mean_optimal_step})\tDist: {info["dist"]} ({mean_dist})\t' \
                    f'V_internal {mean_internal_vel} (var: {var_internal_vel})')


            model.eval()
            env.eval()
            run_info, _logs, task_env, k = evaluate(model, env, envs, args, loss_fn, loss_recon, step_size, use_cuda)
            

            logs.update(_logs)
            local_logs.append(logs)
            if use_wandb:
                wandb.log(logs, step=int(epoch))

            if args.grid_cells:
                env.set_env(task_env, k)
            else:
                env = task_env
            
            lengths = []
            dists = []
            recon_losses = []
            task_losses = []
            vel_losses = []
            optimal_steps = []
            success = []
            internal_velocities = []

    save_dir = args.save_dir
    save_dir = os.path.join(save_dir, save_name)


    epoch = '{:06}'.format(epoch)
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
                , f'{save_dir}/{epoch}.pth')

    torch.save(run_info, f'{save_dir}/run_info_{epoch}.pth')
    checkpoint_name = f'{save_dir}/run_info_{epoch}.pth'

    return model, max_acc, run_info, local_logs

def get_env(args, env_cl, use_cuda=False):
    """
    Get the environment instance based on the arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        env_cl: Environment class to instantiate.
        use_cuda (bool, optional): Flag to use CUDA.

    Returns:
        tuple: Environment instance and input dimension.
    """
    input_dim = args.input_dim
    step_size = args.step_size
    # interval=True, seen_ratio=0.8, done='action' were always the
    # script-passed values; the CLI flags are gone and these are
    # hardcoded in the env_cl call.
    env = env_cl(args.num_images, input_dim, step_size, 0.8, True, 'action', use_cuda=use_cuda)
    return env, input_dim



def main():
    """CLI entry: train the Mental RNN end-to-end for one seed.

    Reads CLI flags via ``lib.arguments.get_args``, sets up the environment,
    instantiates ``Mental`` from ``lib.model``, runs the training loop
    via ``lib.exp.run_single_trial`` / ``lib.exp.evaluate``, and
    periodically dumps per-checkpoint ``run_info_*.pth`` snapshots into
    ``<save_dir>/<run_name>/`` for downstream analysis.

    Side effects:
        - Writes ``arguments.json``, ``log.csv``, ``run_info_*.pth``, and
          ``verbose_info_*.pkl`` under the run directory.
        - Optionally logs metrics to Weights & Biases.

    Returns:
        None.
    """
    args = get_args()
    # 3-class output: stop / left / right.
    num_class = 3

    # grid_cells is implicit in the model name: VHA / VHA-D use the
    # grid-cell input wrapper, all the RNN* baselines do not.
    args.grid_cells = args.model.startswith('VHA')

    input_dim = args.input_dim

    # Environment is always random_vector now.
    env_cl = RandomVecEnv

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(get_free_gpu())
    print(use_cuda)

    # Set random seed for reproduce
    set_seed(args.seed)

    # RNG-state preservation (for reproducibility): this first get_env
    # builds one RandomVecEnv that is immediately discarded on L308
    # (`original_env = envs[0]`). It is kept intact because removing it
    # shifts the global torch RNG state and breaks byte-identical
    # reproduction of paper runs at seed 43. See
    # lib/envs/random_vector.py:generate_data + divide, which consume
    # `torch.randn` and `torch.randperm` respectively.
    original_env, input_dim = get_env(args, env_cl, use_cuda)

    # Add parallel environments. (avoid_data collision-avoidance was
    # scaffolded but never implemented for RandomVecEnv -- the param
    # was threaded through and silently discarded.)
    envs = []
    for i in range(args.num_envs):
        new_env, _ = get_env(args, env_cl, use_cuda)
        envs.append(new_env)

    Ns = input_dim
    Np = min(args.np, np.prod(args.lambdas))

    # The original_env from L293 is now overwritten; only its RNG side
    # effect during construction is load-bearing for seed reproducibility.
    original_env = envs[0]
    env = original_env
    envs = envs[1:]

    name = get_name(args)
    name = name.replace('/','_')
    
    # Change the input dimension based on the running option
    target_dim = input_dim

    match = False
    for lamb in args.lambdas:
        if lamb/args.resolution == (args.input_dim + args.interval_dim) / args.step_size:
            match = True
            break
    args.period_match = match

    if args.grid_cells:
        # Canonical runs are 1D, so e**1 = e; the dimension knob is gone.
        Ng = int(sum(args.lambdas))
        # VHA-ReLU / VHA-ReLU-D flip the place-cell nonlinearity from sign to
        # plain ReLU, matching Vector-HaSH's assoc_utils_np.py:nonlin(...,thresh=0).
        use_relu = args.model in ('VHA-ReLU', 'VHA-ReLU-D')
        arg_gcpc = {'Ng': Ng, 'Np': Np, 'Ns': Ns, 'lambdas': args.lambdas,
                    'use_relu': use_relu}
        print(arg_gcpc)
        env = GridWrapper(original_env, arg_gcpc, use_cuda=use_cuda, resolution=args.resolution)
        input_dim = len(args.lambdas)

    if args.grid_cells:
        target_dim = input_dim
    # Architectural sub-flags derived from args.model so callers don't
    # have to set them in addition to -model.
    args.cat_action = (args.model == 'RNN_action')
    args.decoder = 'future' if args.model in ('RNN_autoreg', 'RNN-D_autoreg') else 'none'
    if args.cat_action:
        input_dim += 1
    # Adjust the input dimension of the Model
    input_dim = input_dim + target_dim
    periods = torch.Tensor(args.lambdas).view(1, -1)
    # Per-baseline class dispatch (see lib.models.MODELS); each class
    # hardcodes its own is_double / encoder_dim / recon choices.
    # grid_cells=True turns on the canonical periodic-subtract input
    # transform (was subtract_mode='positive' + conv_int=True).
    model = MODELS[args.model](
        input_dim=input_dim, hidden_dim=args.feature_dim, num_class=num_class,
        alpha=args.alpha, grid_cells=args.grid_cells, periods=periods,
        scaling_factor=args.scaling_factor,
    )

    print(model)
    if use_cuda:
        model = model.cuda()
    
    # Set parameters for optimizer
    lr = args.lr
    momentum = 0.9
    T = 25
    eta_min = 1e-5

    # Log
    use_wandb = not args.debug
    
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    print(os.listdir(save_dir))
    if os.path.exists(os.path.join(save_dir, 'log.csv')):
        # Already run
        print(save_dir, "Already Ran this setting")
        exit()

    config = {
            'input_dim_actual': input_dim,
            'momentum':momentum,
            'name': name
            }
            
    config.update(vars(args))
    print(config)
    if use_wandb:
        wandb.init(
            name=name,
#            project='mental_navigation_go_period', #'mental_navigation_analysis_visualize',
            project=args.project_name,
            config=config,
            dir=save_dir
            )

    args.name = name
    # Store the current argument into json file in local directory.
    with open(os.path.join(save_dir, "arguments.json"), "w") as f:
        json.dump(config, f)

    params = list(model.parameters())

    # 3-class action classification.
    loss_fn = nn.CrossEntropyLoss(reduction='sum')
    loss_recon = nn.MSELoss(reduction='mean')


    # Define Optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    else:
        raise NotImplementedError
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T, eta_min=eta_min)
#    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    # Run Experiment.
    logs = []
    print("START EXPERIMENT")
    
    model, acc, run_info, _logs = run_exp(model, optimizer, env, use_cuda, args, loss_fn, loss_recon, num_epochs=args.epochs, save_name=name, is_log=True)
    logs += _logs

    # More Environments for testing generalization across the environment
    epochs = int(args.epochs)
    adapt_epochs = int(args.epochs)

    envs = [original_env] + envs
    for i, new_env in enumerate(envs):
        if i == 0:
            continue
        print(f"Adapt {i}-th Environment, Starting from {epochs}")
        if args.grid_cells:
            env.set_env(new_env, k=1)
        else:
            env = new_env
        model, _acc, _run_info, _logs = run_exp(model, optimizer, env, use_cuda, args, loss_fn, loss_recon, prev_epoch=epochs, num_epochs=adapt_epochs, envs=envs[:i], save_name=name, is_log=True)
        epochs += adapt_epochs
        logs += _logs
    
    dfs = pd.DataFrame(logs)
    dfs.to_csv(os.path.join(save_dir, 'log.csv'))


# When the program start from this file, call main function.
if __name__ == '__main__':
    main()

