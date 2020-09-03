import torch
import numpy as np
import os
import random

from env.neural_augs.utils import call_augfn_torch_batched
from env.neural_augs.Res2Net import Res2Net

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


class ReplayBuffer(object):
    """Buffer to store environment transitions"""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, 
        neural_aug_type, neural_aug_skip_prob, neural_aug_average_over, 
        neural_aug_start_iter, neural_aug_warmup_iters, 
        save_augpics, save_augpics_freq, save_augpics_dir
    ):
        self.capacity = capacity
        self.batch_size = batch_size

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        # Neural Augmentations
        self.neural_aug_type = neural_aug_type
        self.neural_aug_skip_prob = neural_aug_skip_prob
        self.neural_aug_average_over = neural_aug_average_over
        assert self.capacity % self.neural_aug_average_over == 0
        self.neural_aug_start_iter = neural_aug_start_iter
        self.neural_aug_warmup_iters = neural_aug_warmup_iters

        self.save_augpics = save_augpics
        self.save_augpics_freq = save_augpics_freq 
        self.save_augpics_dir = save_augpics_dir

        self.noise2net_MAX_EPS = 0.15 # TODO: Add command line arg
        self.noise2net = Res2Net(epsilon=self.noise2net_MAX_EPS, batch_size=3).train().cuda() # Multiply by 3 because frame_stack

        self.idx = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
        
        if self.neural_aug_average_over != 1:
            print("WARNING. This might screw things up")

        if self.neural_aug_type not in {'noise2net', 'none'}:
            raise NotImplementedError()
        
        # Default Values
        obses = [obs for _ in range(self.neural_aug_average_over)]
        next_obses = [next_obs for _ in range(self.neural_aug_average_over)]

        if self.idx > self.neural_aug_start_iter:
            if self.neural_aug_type == 'noise2net' and random.random() > self.neural_aug_skip_prob:
                # Apply noise2net
                self.noise2net.reload_parameters()
                
                # Linear epsilon warmup
                warmup_frac = (self.idx - self.neural_aug_start_iter)/self.neural_aug_warmup_iters
                if warmup_frac < 1.0:
                    eps = self.noise2net_MAX_EPS * warmup_frac
                else:
                    eps = random.uniform(0, self.noise2net_MAX_EPS)
                self.noise2net.set_epsilon(eps)

                obses = (call_augfn_torch_batched(
                    torch.as_tensor(obs).float().cuda().unsqueeze(0) / 255.0, 
                    self.noise2net.forward, 
                    copies=self.neural_aug_average_over
                ).cpu().numpy() * 255.0).astype(np.uint8)
                next_obses = (call_augfn_torch_batched(
                    torch.as_tensor(next_obs).float().cuda().unsqueeze(0) / 255.0, 
                    self.noise2net.forward, 
                    copies=self.neural_aug_average_over
                ).cpu().numpy() * 255.0).astype(np.uint8)

        # Fix dimensions of actions, rewards, and dones
        actions = [action for _ in range(self.neural_aug_average_over)]
        rewards = [reward for _ in range(self.neural_aug_average_over)]
        dones = [done for _ in range(self.neural_aug_average_over)]

        if self.save_augpics or self.idx % self.save_augpics_freq == 0:
            # Save and exit
            import torchvision
            O = torch.as_tensor(obses).float().detach().cpu().reshape((3, 3, 100, 100))
            N = torch.as_tensor(next_obses).float().detach().cpu().reshape((3, 3, 100, 100))
            torchvision.utils.save_image(O / 255.0, os.path.join(self.save_augpics_dir, f"augmented_obs_{self.idx}.png"))
            torchvision.utils.save_image(N / 255.0, os.path.join(self.save_augpics_dir, f"augmented_next_obs_{self.idx}.png"))

            if self.save_augpics:
                # Legacy flag
                print(actions)
                exit()

        for i, (O, A, R, N, D) in enumerate(zip(obses, actions, rewards, next_obses, dones)):
            np.copyto(self.obses[self.idx + i], O)
            np.copyto(self.actions[self.idx + i], A)
            np.copyto(self.rewards[self.idx + i], R)
            np.copyto(self.next_obses[self.idx + i], N)
            np.copyto(self.not_dones[self.idx + i], not D)

        self.idx = (self.idx + self.neural_aug_average_over) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)

        return obses, actions, rewards, next_obses, not_dones

    def sample_curl(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs]).float().cuda()
        actions = torch.as_tensor(self.actions[idxs]).cuda()
        rewards = torch.as_tensor(self.rewards[idxs]).cuda()
        next_obses = torch.as_tensor(self.next_obses[idxs]).float().cuda()
        not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

        pos = obses.clone()

        obses = random_crop(obses)
        next_obses = random_crop(next_obses)
        pos = random_crop(pos)
        
        curl_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, curl_kwargs


def get_curl_pos_neg(obs, replay_buffer):
	"""Returns one positive pair + batch of negative samples from buffer"""
	obs = torch.as_tensor(obs).cuda().float().unsqueeze(0)
	pos = obs.clone()

	obs = random_crop(obs)
	pos = random_crop(pos)

	# Sample negatives and insert positive sample
	obs_pos = replay_buffer.sample_curl()[-1]['obs_pos']	
	obs_pos[0] = pos

	return obs, obs_pos


def batch_from_obs(obs, batch_size=32):
	"""Converts a pixel obs (C,H,W) to a batch (B,C,H,W) of given size"""
	if isinstance(obs, torch.Tensor):
		if len(obs.shape)==3:
			obs = obs.unsqueeze(0)
		return obs.repeat(batch_size, 1, 1, 1)

	if len(obs.shape)==3:
		obs = np.expand_dims(obs, axis=0)
	return np.repeat(obs, repeats=batch_size, axis=0)


def _rotate_single_with_label(x, label):
	"""Rotate an image"""
	if label == 1:
		return x.flip(2).transpose(1, 2)
	elif label == 2:
		return x.flip(2).flip(1)
	elif label == 3:
		return x.transpose(1, 2).flip(2)
	return x


def rotate(x):
	"""Randomly rotate a batch of images and return labels"""
	images = []
	labels = torch.randint(4, (x.size(0),), dtype=torch.long).to(x.device)
	for img, label in zip(x, labels):
		img = _rotate_single_with_label(img, label)
		images.append(img.unsqueeze(0))

	return torch.cat(images), labels


def random_crop_cuda(x, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized CUDA implementation of random crop"""
	assert isinstance(x, torch.Tensor) and x.is_cuda, \
		'input must be CUDA tensor'
	
	n = x.shape[0]
	img_size = x.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return x, None, None
		return x

	x = x.permute(0, 2, 3, 1)

	if w1 is None:
		w1 = torch.LongTensor(n).random_(0, crop_max)
		h1 = torch.LongTensor(n).random_(0, crop_max)

	windows = view_as_windows_cuda(x, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[torch.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped


def view_as_windows_cuda(x, window_shape):
	"""PyTorch CUDA-enabled implementation of view_as_windows"""
	assert isinstance(window_shape, tuple) and len(window_shape) == len(x.shape), \
		'window_shape must be a tuple with same number of dimensions as x'
	
	slices = tuple(slice(None, None, st) for st in torch.ones(4).long())
	win_indices_shape = [
		x.size(0),
		x.size(1)-int(window_shape[1]),
		x.size(2)-int(window_shape[2]),
		x.size(3)    
	]

	new_shape = tuple(list(win_indices_shape) + list(window_shape))
	strides = tuple(list(x[slices].stride()) + list(x.stride()))

	return x.as_strided(new_shape, strides)


def random_crop(imgs, size=84, w1=None, h1=None, return_w1_h1=False):
	"""Vectorized random crop, imgs: (B,C,H,W), size: output size"""
	assert (w1 is None and h1 is None) or (w1 is not None and h1 is not None), \
		'must either specify both w1 and h1 or neither of them'

	is_tensor = isinstance(imgs, torch.Tensor)
	if is_tensor:
		assert imgs.is_cuda, 'input images are tensors but not cuda!'
		return random_crop_cuda(imgs, size=size, w1=w1, h1=h1, return_w1_h1=return_w1_h1)
		
	n = imgs.shape[0]
	img_size = imgs.shape[-1]
	crop_max = img_size - size

	if crop_max <= 0:
		if return_w1_h1:
			return imgs, None, None
		return imgs

	imgs = np.transpose(imgs, (0, 2, 3, 1))
	if w1 is None:
		w1 = np.random.randint(0, crop_max, n)
		h1 = np.random.randint(0, crop_max, n)

	windows = view_as_windows(imgs, (1, size, size, 1))[..., 0,:,:, 0]
	cropped = windows[np.arange(n), w1, h1]

	if return_w1_h1:
		return cropped, w1, h1

	return cropped
