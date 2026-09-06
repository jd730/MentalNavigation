import numpy as np
import torch


class RandomVecEnv:
    def __init__(self, num_images, input_size, step_size=1, seen_ratio=0.8, interval=False, done='correct', use_cuda=False):
        self.use_cuda = use_cuda
        self.num_images = num_images
        self.input_size = input_size
        self.step_size = step_size
        self.current = -1
        self.ratio = seen_ratio
        self.interval = interval
        self.done = done
        self.is_mental = False
        self.mode = 'all'
        self.is_eval = False
        # search_target was always False in canonical runs; target is
        # known up-front, so target_found starts True.
        self.target_found = True
        self.clip_ends = True

        self.generate_data()
        self.divide()

    def eval(self):
        self.is_eval = True

    def train(self):
        self.is_eval = False

    def generate_data(self):
        images = torch.randn(self.num_images, self.input_size)
        zeros = torch.zeros_like(images)
        if self.interval:
            self.data = torch.cat((images, zeros), -1).view(-1)[:-self.input_size]
        else:
            self.data = images.view(-1)
        if self.use_cuda:
            self.data = self.data.cuda()

    def get_mode(self):
        return self.mode

    def set_mode(self, mode):
        self.mode = mode


    def divide(self):
        pair = torch.ones((self.num_images, self.num_images)) - torch.eye(self.num_images)
        pair = pair.nonzero() # N x 2
        # shuffle
        order = torch.randperm(len(pair))
        pair = pair[order]
        N = int(len(pair) * self.ratio)
        self.pair = pair
        self.train_pair = pair[:N]
        self.test_pair = pair[N:]


    def get_seq_len(self):
        return len(self.data) // self.step_size + 2

    def __len__(self):
        if self.mode == 'all':
            return len(self.pair)
        elif self.mode == 'train':
            return len(self.train_pair)
        else:
            return len(self.test_pair)

    def set_mental(self, val=False):
        self.is_mental = val

    def reset(self, idx=None, start_idx=None, target_idx=None):
        prev_loc = self.current
        if self.mode == 'all':
            pair = self.pair
        elif self.mode == 'train':
            pair = self.train_pair
        else:
            pair = self.test_pair
        
        if idx is None:
            idx = torch.randint(len(pair), (1,))
            if start_idx is None:
                start_idx = pair[idx, 0]
            if target_idx is None:
                target_idx = pair[idx, 1]
        else:
            start_idx = pair[idx, 0]
            target_idx = pair[idx, 1]

        target_idx = target_idx * self.input_size
        start_idx = start_idx * self.input_size
        if self.interval:
            target_idx *= 2
            start_idx *= 2
        self.target_img = self.data[target_idx:target_idx+self.input_size]
        self.target_idx = target_idx.item() if type(target_idx) is not int else target_idx
        self.start_idx = start_idx.item() if type(start_idx) is not int else start_idx
        start_img = self.data[start_idx:start_idx+self.input_size]
        state = torch.cat((start_img, self.target_img))

        self.current = self.start_idx
        info = self.info
        if self.current >= 0:
            info['reset_dist'] = self.current - prev_loc
        else:
            info['reset_dist'] = 0
        return state, info


    def get_current_img(self, idx, imagine=False):
        if (self.is_mental and not imagine) or idx < -self.input_size or idx >= len(self.data):
            current_img = torch.zeros((self.input_size), device=self.data.device)
        elif idx < 0:
            padding = torch.zeros((-idx), device=self.data.device)
            img = self.data[:idx+self.input_size]
            current_img = torch.cat((padding, img))
        elif idx + self.input_size > len(self.data):
            padding = torch.zeros((idx+self.input_size-len(self.data)), device=self.data.device)
            img = self.data[idx:]
            current_img = torch.cat((img, padding))
        else:
            current_img = self.data[idx:idx+self.input_size]

        return current_img 

    @property
    def info(self):
#        interval = 2 if self.interval else 1

        next_img = [self.get_current_img(self.current-self.step_size, True),
                    self.get_current_img(self.current, True),
                    self.get_current_img(self.current+self.step_size, True)]

        dist = (self.target_idx - self.current) // self.step_size
        info = {'best_action': self.best_action,
                'current_pos': self.current,
                'start_pos': self.start_idx,
                'target_pos': self.target_idx,
                'next_img': next_img,
                'dist': dist,
                'success': int(dist == 0),
                'prev_action': 0}
        return info

    def store_env(self):
        """
            Store the current environmental information for simulation and recall.
        """
        self.fixed_current = self.current

    def recall_env(self):
        """
            Recall the stored environmental information.
        """
        self.current = self.fixed_current
        del self.fixed_current

    def step(self, action):
        self.current = self.current + action * self.step_size
        if self.clip_ends:
            self.current = np.clip(self.current, -self.input_size, len(self.data))

        current_img = self.get_current_img(self.current)

        state = torch.cat((current_img, self.target_img))
        if self.done == 'correct':
            done = self.current == self.target_idx and action == 0
        elif self.done == 'action':
            done = action == 0
        else:
            raise NotImplementedError

        self.target_found = self.target_found or (self.current == self.target_idx)
        reward = 0
        info = self.info
        info['prev_action'] = action
        info['success'] = int(info['success'] and action == 0)
        return state, reward, done, info


    @property
    def best_action(self):
        if self.target_idx > self.current:
            return +1
        elif self.target_idx < self.current:
            return -1
        else:
            return 0


if __name__ == '__main__':

    env = RandomVecEnv(4, 5, 1)
    print(env.reset())
    for i in range(30):
        print(env.best_action)
        print(env.step(1))
        breakpoint()
