import torch
import copy

from lib.gridUtils import grid_cell_initial, run_grid_cells 
from .base_wrapper import BaseWrapper



class GridWrapper(BaseWrapper):
    def __init__(self, env, arg_gcpc, use_cuda=True, resolution=1):
        """
        Initialize the GridWrapper.

        Args:
            env: The environment instance.
            arg_gcpc (dict): Arguments for grid cell place cells.
            use_cuda (bool): Flag to determine if CUDA should be used.
            resolution (int, optional): Internal-velocity resolution multiplier.
        """
        super().__init__(env)
        self.use_cuda = use_cuda
        self.current_internal_loc = None
        self.init_gcpc(arg_gcpc)
        self.sanity_mode = False
        self.verbose = False
        self.k = 0
        self.prev_action_in_zero = 0
        self.gps = None
        self.last_g = None
        self.allocated_grids = None
        self.visited = set()
        self.retrieve = []
        self.jaed = {}
        self.jaed2 = {}
        self.resolution = resolution
    

    @property
    def target_found(self):
        """Check if the target is found in the environment."""
        return self.env.target_found

    def init_gcpc(self, arg_gcpc):
        """
        Initialize grid cell place cells (GCPC).

        Args:
            arg_gcpc (dict): Arguments for grid cell place cells.
        """
        self.Np = arg_gcpc['Np']
        self.Ns = arg_gcpc['Ns']
        self.Ng = arg_gcpc['Ng']
        self.lambdas = arg_gcpc['lambdas']
        self.use_relu = arg_gcpc.get('use_relu', False)
        print('init')
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps = grid_cell_initial(self.Np, self.Ns, self.lambdas, self.use_cuda, use_relu=self.use_relu)
        torch.cuda.empty_cache()
        self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps]


    def explore_sequence(self, verbose=False):
        """
        Explore the sequence of actions and observations in the environment.

        Args:
            verbose (bool, optional): Flag for verbose mode.
        """
        self.sanity_mode = True
        self.verbose = verbose
        target_idx = self.num_images - 1

        current_pos = self.current_pos
        current_internal_loc = self.current_internal_loc
        target_found = self.env.target_found

        if self.last_g is not None:
            Wgg = self.grid_info[0][(0, 1)]
            g = self.last_g
            step = self.resolution
            for _ in range(step):
                g = Wgg @ g
            print(self.last_g.nonzero()[:,0], g.nonzero()[:,0])
            self.grid_info[7] = g
            self.last_g = None
        prev_g = self.grid_info[7]
#        self.k += 1
        grid_info = copy.deepcopy(self.grid_info)
        (obs, p, g, s, sinit), info = self.reset(start_idx=0, target_idx=target_idx)

        self.grid_info = grid_info


        c_left = 0
        c_right = 0

        # Find the leftmost Observation
        while (self.current_pos != current_pos or self.current_internal_loc != current_internal_loc):
            current_internal_loc = self.current_internal_loc
            current_pos = self.current_pos
            obs, _, done, info = self.env.step(-1)
            self.k += 1
            self.current_pos = info['current_pos']
            self.current_internal_loc = info.get('current_internal_loc', None)
            c_left += 1
        
        sensories = []
        places = []
        grids = []
        self.train()
        current_pos = -1
        current_internal_loc = None

        # moving to the right
        while (self.current_pos != current_pos or self.current_internal_loc != current_internal_loc):
            current_internal_loc = self.current_internal_loc
            current_pos = self.current_pos
            (obs, p, g, s, sinit), _, done, info = self.step(1, internal_velocity=self.resolution)

            sensories.append(sinit)
            places.append(p)
            grids.append(g)

            c_right += 1

        print(c_left, c_right)
            
        # check whether it can start from any location
        self.eval()
        sensories = torch.cat(sensories)
        places = torch.cat(places)
        grids = torch.cat(grids)

        self.gps = [grids, places, sensories]
        int_grids = [e.nonzero()[:,0] for e in grids]
        
        self.association_test(grids, places, sensories)

        self.verbose = False
        self.sanity_mode = False

        self.current_pos = -1
        self.current_internal_loc = None

        
        grids = grids.unique(dim=0, sorted=False)
        grids = torch.flip(grids, dims=(0,)) # order is reversed.

        if self.allocated_grids is None:
            self.allocated_grids = grids
        else:
            self.allocated_grids = torch.cat((self.allocated_grids, grids))
        
        self.last_g = self.gps[0][self.gps[0].nonzero()[-1][0]]

        self.env.target_found = target_found

    def association_test(self, grids, places, sensories):
        """
        Perform an association test on grids, places, and sensories.

        Args:
            grids (torch.Tensor): Grids tensor.
            places (torch.Tensor): Places tensor.
            sensories (torch.Tensor): Sensories tensor.
        """
        sanity_mode = self.sanity_mode
        self.sanity_mode = True
        target_idx = 0

        for i in range(self.num_images):
            (obs, p, g, s, sinit), info = self.reset(start_idx=i, target_idx=target_idx)
            correct_grid = grids[(sensories == sinit).all(1)[:,0]][0:1]
            correct_place = places[(sensories == sinit).all(1)[:,0]][0:1]
            check_g = ((g == correct_grid).all())
            check_p = ((p == correct_place).all())
#            print(g.nonzero(), correct_grid.nonzero(), i) 
            print(f"Starting from {i}-th Image Check g: {check_g}, p: {check_p}")
            if not check_g:
                compare = (g==correct_grid).all(1)
                print(g.nonzero()[:,1], correct_grid.nonzero()[:,1])
                print("Reconstruction is failed")
        self.sanity_mode = sanity_mode


    def reset(self, idx=None, start_idx=None, target_idx=None):
        """
        Reset the environment.

        Args:
            idx (int, optional): Index for reset.
            start_idx (int, optional): Start index for reset.
            target_idx (int, optional): Target index for reset.

        Returns:
            tuple: Observation and information.
        """
        self.prev_known = False
        self.retrieve = []
        obs, info = self.env.reset(idx, start_idx, target_idx)
        obs = self.convert_obs(obs, force_prev=self.k > 0, internal_velocity=self.resolution)
        if not self.sanity_mode:
            obs, obs_info = obs
            info.update(obs_info)
        self.current_pos = info['current_pos']
        self.current_internal_loc = info.get('current_internal_loc', None)

        return obs, info


    def convert_obs(self, obs, prev_action=0, force_prev=False, internal_velocity=1):
        """
        Convert the observation.

        Args:
            obs (torch.Tensor): Observation tensor.
            prev_action (int, optional): Previous action taken.
            force_prev (bool, optional): Flag to force previous action.
            internal_velocity (int, optional): Internal velocity value.

        Returns:
            tuple: Converted observation and observation information.
        """
        target = obs[len(obs)//2:]
        obs = obs[:len(obs)//2] # extract current observation ignoring target obsa
        sinit = obs.unsqueeze(-1).unsqueeze(0)
        
        if not self.target_found:
            target[:] = 0 # or random vec.
            target = torch.randn_like(target)

        flag = not self.is_eval and torch.all(sinit == 0) # do not move grid code during interval
        if self.prev_action_in_zero != 0 and prev_action != self.prev_action_in_zero: # correction due to empty vector.
            prev_action = 0
        self.prev_action_in_zero = 0 # init.
        if self.use_cuda:
            sinit = sinit.cuda()
        else:
            sinit = sinit.numpy()
        obs = self.grid_step(sinit, prev_action, force_prev, internal_velocity=internal_velocity)

        tinit = target.unsqueeze(-1).unsqueeze(0)
        if self.use_cuda:
            tinit = tinit.cuda()
        else:
            tinit = tinit.numpy()
        target = self.grid_step(tinit, recon_only=True)

        if self.sanity_mode:
            obs, p, g, s, sinit = obs
            target = target[0]
            obs = torch.cat((obs, target), 1)
            return obs, p, g, s, sinit

        obs, obs_info = obs
        obs_info = {'current_'+k: v for k, v in obs_info.items()}
        target, target_info = target
        obs_info.update({'target_'+k: v for k, v in target_info.items()})
        if flag:
            obs *= 0
        obs = torch.cat((obs, target), 1)
        return obs, obs_info


    def set_env(self, env, k=0):
        """
        Set the environment.

        Args:
            env: The environment instance.
            k (int, optional): Counter for steps taken.
        """
        # move grid cell to the initial position.
        env.current = self.env.current
        print(self.env, self.env.target_found, env, env.target_found)
        self.env.jaed = self.jaed
        self.env.jaed2 = self.jaed2
        if hasattr(env, 'jaed'):
            print("Recall Jaed and Jaed2")
            self.jaed = env.jaed
            self.jaed2 = env.jaed2
        self.env = env
        self.k = k

    def get_env(self):
        """
        Get the current environment and step counter.

        Returns:
            tuple: Environment instance and step counter.
        """
        return self.env, self.k

    def store_env(self):
        """
            Store the current environmental information for simulation and recall.
        """
        self.env.store_env()
        self.fixed_pos = self.current_pos
        self.fixed_grid_info = copy.deepcopy(self.grid_info)

    def recall_env(self):
        """
        Recall the stored environmental information.
        """
        self.env.recall_env()
        self.current_pos = self.fixed_pos
        self.grid_info = self.fixed_grid_info
        del self.fixed_pos
        del self.fixed_grid_info

    def step(self, action, internal_velocity=1):
        """
        Perform a step in the environment.

        Args:
            action (int or list): Action to be taken.
            internal_velocity (int, optional): Internal velocity value.

        Returns:
            tuple: Observation, reward, done flag, and information.
        """
        obs, reward, done, info = self.env.step(action if type(action) is not list else action[-1])
        prev_pos = self.current_pos
        prev_internal_loc = self.current_internal_loc
        self.current_pos = info['current_pos']
        self.current_internal_loc = info.get('current_internal_loc', None)
        self.k += 1
        if prev_pos == self.current_pos and self.current_internal_loc == prev_internal_loc:
            prev_action = 0
        elif info['prev_action'] != action:
            raise RuntimeError(
                f"GridWrapper.step: env recorded prev_action="
                f"{info['prev_action']!r} but agent emitted action={action!r}; "
                "this should never happen if the wrapped env honours its action.")
        else:
            prev_action = action

        obs = self.convert_obs(obs, prev_action, internal_velocity=internal_velocity)
        if not self.sanity_mode:
            obs, obs_info = obs
            info.update(obs_info)
        return obs, reward, done, info


    def run_grid_cells(self, sinit, disp, force_prev, recon_only, internal_velocity=1):
        """
        Run grid cells for the given inputs.

        Args:
            sinit (torch.Tensor): Initial sensory input.
            disp (int): Displacement value.
            force_prev (bool): Flag to force previous action.
            recon_only (bool): Flag to indicate reconstruction only.
            internal_velocity (int, optional): Internal velocity value.

        Returns:
            tuple: Place, grid, and sensory outputs.
        """
        gap = 3 + (getattr(self.env, 'mat_size', 0) // 2)
        Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps = self.grid_info
        p, g, s, sbook_obs, pbook_obs, Wsp, Wps, theta_sp, theta_ps = run_grid_cells(sinit, disp, Wggs, Wgp, Wpg,
                                                                                 Wsp, Wps, module_gbooks, module_sizes, g,
                                                                                 sbook_obs, pbook_obs, theta_sp, theta_ps,
                                                                                 self.use_cuda, self.is_eval, force_prev,
                                                                                 verbose=self.verbose, allocated_grids=self.allocated_grids,
                                                                                 gap=gap, recon_only=recon_only,
                                                                                 internal_velocity=internal_velocity,
                                                                                 use_relu=self.use_relu)
        if not recon_only:
            self.grid_info = [Wggs, Wgp, Wpg, Wsp, Wps, module_gbooks, module_sizes, g, sbook_obs, pbook_obs, theta_sp, theta_ps]
        return p, g, s

    def grid_step(self, sinit, disp=0, force_prev=False, recon_only=False, internal_velocity=1):
        """
        Perform a grid step.

        Args:
            sinit (torch.Tensor): Initial sensory input.
            disp (int, optional): Displacement value.
            force_prev (bool, optional): Flag to force previous action.
                force_prev is only True when the second or later episode is reset.
            recon_only (bool, optional): Flag to indicate reconstruction only.
                recon_only is only used for converting target to grid code.
            internal_velocity (int, optional): Internal velocity value.

        Returns:
            tuple: Observation and observation information.
        """

        is_known = True
        p, g, s = self.run_grid_cells(sinit, disp, force_prev, recon_only, internal_velocity=internal_velocity)

        if type(g) is not torch.Tensor:
            g = torch.Tensor(g).float()
            p = torch.Tensor(p).float()

        # mode is always 'g' and conv_int is always True: p/s outputs and
        # the non-conv_int path were both dead in the canonical paper runs.
        _g = g.nonzero()[:,1]
        for i,  lamb in enumerate(self.lambdas[:-1]): #changed 05.07.2024
            for j in range(i+1, len(self.lambdas)):
                _g[j] = _g[j] - lamb
        _g = _g.unsqueeze(0)
        obs = _g.float()
        obs_info = {'p': p.cpu().numpy(), 'img': sinit.cpu().numpy(), 'g': _g.cpu().numpy()}

        if self.sanity_mode:
            return obs, p, g, s, sinit
        obs_info['is_known'] = is_known
        return obs, obs_info

    def _update_g_only(self, g):
        """
        Update grid information only.

        Args:
            g (torch.Tensor): Grid tensor.
        """
        self.grid_info[7] = g

    def compare_grid(self, g1, g2):
        """
        Compare two grids.

        Args:
            g1 (torch.Tensor): First grid tensor.
            g2 (torch.Tensor): Second grid tensor.

        Returns:
            bool: True if grids are equal, False otherwise.
        """
        return g1[0]==g2[0] and g1[1] == g2[1] and g1[2] == g2[2]

