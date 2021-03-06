# -*- coding: utf-8 -*-

"""
Given a bunch of commands to run, check the available GPUs and run them on the GPUs in separate tmux sessions.
Usage: Just modify the settings in the Config class and then run python3 gpu_run.py
"""

import GPUtil
import subprocess
import sys
import time

class Config:
    """
    Global class that houses all configurations
    """
    
    # Shared args to put onto all of the JOBS
    SHARED_ARGS = ""

    HEADER = "conda activate rl-pad-3; "

    # Specifies tasks to run. It maps tmux session name to the command to run in that session.
    JOBS = {
        
        #####################################################################################
        #### SHADOWFAX
        #####################################################################################

        # Finished Running

        # "cartpole_swingup_noSS_noise2net_startIter0_warmupIter50k_seed3" : "python3 src/train.py \
        #     --domain_name cartpole \
        #     --task_name swingup \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 8 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 1 \
        #     --work_dir logs/cartpole_swingup/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed1 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "ball_in_cup_catch_noSS_noise2net_startIter0_warmupIter50k_seed3" : "python3 src/train.py \
        #     --domain_name ball_in_cup \
        #     --task_name catch \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 4 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 1 \
        #     --work_dir logs/ball_in_cup_catch/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed1 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "cartpole_swingup_noSS_noise2net_startIter0_warmupIter50k_seed2" : "python3 src/train.py \
        #     --domain_name cartpole \
        #     --task_name swingup \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 8 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 2 \
        #     --work_dir logs/cartpole_swingup/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed2 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "ball_in_cup_catch_noSS_noise2net_startIter0_warmupIter50k_seed2" : "python3 src/train.py \
        #     --domain_name ball_in_cup \
        #     --task_name catch \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 4 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 2 \
        #     --work_dir logs/ball_in_cup_catch/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed2 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "cartpole_swingup_noSS_noise2net_startIter0_warmupIter50k_seed3_REAL" : "python3 src/train.py \
        #     --domain_name cartpole \
        #     --task_name swingup \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 8 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 3 \
        #     --work_dir logs/cartpole_swingup/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed3 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "ball_in_cup_catch_noSS_noise2net_startIter0_warmupIter50k_seed3_REAL" : "python3 src/train.py \
        #     --domain_name ball_in_cup \
        #     --task_name catch \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 4 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 3 \
        #     --work_dir logs/ball_in_cup_catch/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter50k_seed3 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 50000 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # neural_aug_warmup_iters 0 and max_eps = 0.20

        # "ball_in_cup_catch_noSS_noise2net_startIter0_warmupIter0_seed3_REAL" : "python3 src/train.py \
        #     --domain_name ball_in_cup \
        #     --task_name catch \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 4 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 3 \
        #     --work_dir logs/ball_in_cup_catch/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter0_seed3 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 0 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # "cartpole_swingup_noSS_noise2net_startIter0_warmupIter0_seed3_REAL" : "python3 src/train.py \
        #     --domain_name cartpole \
        #     --task_name swingup \
        #     --replaybuffer_size 200000 \
        #     --action_repeat 8 \
        #     --mode train \
        #     --num_shared_layers 8 \
        #     --seed 3 \
        #     --work_dir logs/cartpole_swingup/noSS/replaybuffer_size_200000_noise2net_startIter0_warmupIter0_seed3 \
        #     --save_model \
        #     --neural_aug_start_iter 0 \
        #     --neural_aug_warmup_iters 0 \
        #     --save_augpics_freq 10101 \
        #     --neural_aug_type noise2net",

        # Currently Running

        "ball_in_cup_catch_noSS_noise2net_maxEps25e-2_startIter0_warmupIter0_seed3" : "python3 src/train.py \
            --domain_name ball_in_cup \
            --task_name catch \
            --replaybuffer_size 200000 \
            --action_repeat 4 \
            --mode train \
            --num_shared_layers 8 \
            --seed 3 \
            --work_dir logs/ball_in_cup_catch/noSS/replaybuffer_size_200000_noise2net_maxEps25e-2_startIter0_warmupIter0_seed3 \
            --save_model \
            --neural_aug_start_iter 0 \
            --neural_aug_warmup_iters 0 \
            --save_augpics_freq 10101 \
            --neural_aug_max_eps 0.25 \
            --neural_aug_type noise2net",

        "cartpole_swingup_noSS_noise2net_maxEps25e-2_startIter0_warmupIter0_seed3" : "python3 src/train.py \
            --domain_name cartpole \
            --task_name swingup \
            --replaybuffer_size 200000 \
            --action_repeat 8 \
            --mode train \
            --num_shared_layers 8 \
            --seed 3 \
            --work_dir logs/cartpole_swingup/noSS/replaybuffer_size_200000_noise2net_maxEps25e-2_startIter0_warmupIter0_seed3 \
            --save_model \
            --neural_aug_start_iter 0 \
            --neural_aug_warmup_iters 0 \
            --save_augpics_freq 10101 \
            --neural_aug_max_eps 0.25 \
            --neural_aug_type noise2net",

        #####################################################################################
        #### SMAUG
        #####################################################################################
        
    }

    # Time to wait between putting jobs on GPUs (in seconds). This is useful because it might take time 
    # for a process to actually load the network onto the GPU, so we wait until that is done before 
    # selecting the GPU for the next process.
    SLEEP_TIME = 30

    # Minimum memory required on a GPU to consider putting a job on it (MiB).
    MIN_MEMORY_REQUIRED = 4000


# Stick the shared args onto each JOB
for key, value in Config.JOBS.items():
    new_value = value + " " + Config.SHARED_ARGS
    Config.JOBS[key] = new_value

def select_gpu(GPUs):
    """
    Select the next best available GPU to run on. If nothing exists, return None
    """
    GPUs = list(filter(lambda gpu: gpu.memoryFree > Config.MIN_MEMORY_REQUIRED, GPUs))
    if len(GPUs) == 0:
        return None
    GPUs = sorted(GPUs, key=lambda gpu: gpu.memoryFree)
    return GPUs[-1]

for index, (tmux_session_name, command) in enumerate(Config.JOBS.items()):
    # Get the best available GPU
    print("Finding GPU for command \"{0}\"".format(command))
    curr_gpu = select_gpu(GPUtil.getGPUs())

    if curr_gpu == None:
        print("No available GPUs found. Exiting.")
        sys.exit(1)

    print("SUCCESS! Found GPU id = {0} which has {1} MiB free memory".format(curr_gpu.id, curr_gpu.memoryFree))

    result = subprocess.run("tmux new-session -d -s {0}".format(tmux_session_name), shell=True)        
    if result.returncode != 0:
        print("Failed to create new tmux session called {0}".format(tmux_session_name))
        sys.exit(result.returncode)

    result = subprocess.run("tmux send-keys '{2} CUDA_VISIBLE_DEVICES={0} {1}' C-m".format(
        curr_gpu.id, command, Config.HEADER
    ), shell=True)
    if result.returncode != 0:
        print("Failed to run {0} in tmux session".format(command, tmux_session_name))
        sys.exit(result.returncode)

    print("---------------------------------------------------------------")

    if index < len(Config.JOBS) - 1:
        time.sleep(Config.SLEEP_TIME)
