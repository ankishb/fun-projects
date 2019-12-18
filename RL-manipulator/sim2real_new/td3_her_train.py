
import argparse
import os
import sys

import copy
import gym
import numpy as np
import torch

import TD3
import utils

import envs
import noise

f_save = open("save_reward.txt", "a")


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=2):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        count=0
        while not done:
            st = obs['state']
            gt = obs['goal']
            sjt = obs['joint']
            sgt = np.hstack((st, gt, sjt))
            action = policy.select_action(np.array(sgt))
        
            obs_, reward, done, _ = env.step(action, sjt)
            avg_reward += reward
            # print(action, reward)
            count += 1
            obs = obs_


    avg_reward /= eval_episodes

    f_save.write(f"Evaluation over {eval_episodes} episodes: {avg_reward} \n")
    print("---------------------------------------")
    print("Evaluation over {} episodes: {}".format(eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename_suffix", type=str,default = "sinx") # custom file name suffix
    parser.add_argument("--exp_name", type=str,default="ur10")  # Place to store log
    parser.add_argument("--env_name", default="ur_Env-v0")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=1e4,
                        type=int)  # How many time steps purely random policy is run for
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=2e6, type=int)  # Max time steps to run environment for
    parser.add_argument("--save_models", action="store_true",default=True)  # Whether or not models are saved
    parser.add_argument("--expl_noise", default=0.4, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=128, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2, type=float)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5, type=float)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--check_collision", action="store_true",default=True)  # whether to check for collision or not

    # if check collision is true, then whether to end episode on collision
    parser.add_argument("--stop_on_hit", action="store_true",default=True)
    # if check_collision is true, then what reward should be added to current reward  
    parser.add_argument("--collision_reward", default=-10, type=int)  
    # while training, whether to sample valid goals or not
    parser.add_argument("--valid_goals", action="store_true",default=True)
    # reward to provide on successfully reaching the goal
    parser.add_argument("--reaching_reward", default=1, type=int)
    # what coeff factor to apply on action norm while calculating the reward  
    parser.add_argument("--action_reward_coeff", default=0, type=float)
    # whether want to visualize in GUI mode  
    parser.add_argument("--renders",type=str,default=True)  


    args = parser.parse_args()
    if args.filename_suffix is None:
        file_name = "{}_{}".format(args.env_name, str(args.seed))
    else:
        file_name = "{}_{}".format(args.env_name, args.filename_suffix)

    root_path = '/'.join(i for i in os.path.abspath(__file__).split('/')[:-1])
    sys.path.insert(0, root_path)
 
    base_path = os.path.join(root_path, "log", args.exp_name)
    if not os.path.exists(os.path.join(base_path, "results")):
        os.makedirs(os.path.join(base_path, "results"))
    if args.save_models and not os.path.exists(os.path.join(base_path, "pytorch_models")):
        os.makedirs(os.path.join(base_path, "pytorch_models"))

    env = gym.make(args.env_name)
    env._renders          = args.renders
    env._reaching_rew     = args.reaching_reward
    env._action_rew_coeff = args.action_reward_coeff
    
    if args.check_collision:
        env._collision_check  = args.check_collision
        env._collision_reward = args.collision_reward
        
        if args.stop_on_hit:    
            env._stop_on_hit = True
    
    if args.valid_goals:    
        env._valid_goal = True

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(f"state_dim : {state_dim}, action_dim : {action_dim}")
    # Initialize policy
    policy = TD3.TD3(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    # evaluations = [evaluate_policy(policy)]

    total_timesteps      = 0
    timesteps_since_eval = 0
    
    done = True
    episode_num = 0


    #############################################
    #########        Modification By       ######
    #########         Ankish Bansal        ######
    #########       to your same logic     ######
    #############################################
    """
    for i in range(args.max_timesteps):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        

        for j in range(MAX_EP_STEPS):
            total_timesteps += 1
        
            # Select action randomly or according to policy
            if total_timesteps > args.start_timesteps:
                action = env.action_space.sample()
            else:
                action = policy.select_action(np.array(obs))
                action = np.random.normal(action, args.expl_noise)
                action = np.clip(env.action_space.low, env.action_space.high)

            
            new_obs, reward, done, _ = env.step(action)

            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_reward += reward

            # Store data in replay buffer
            replay_buffer.add((obs, new_obs, action, reward, done_bool))

            obs = new_obs

            episode_timesteps += 1
            total_timesteps += 1
            timesteps_since_eval += 1


        if replay_buffer.pointer >= replay_buffer.max_size:
            print("Total T: {} Episode Num: {} Episode T: {} Reward: {}".format(
                total_timesteps, episode_num, episode_timesteps, episode_reward
            ))
            policy.train(
                replay_buffer, episode_timesteps, 
                args.batch_size, args.discount, 
                args.tau, args.policy_noise, 
                args.noise_clip, args.policy_freq
            )
        
        # Evaluate episode
        if done or timesteps_since_eval >= args.eval_freq:
            timesteps_since_eval %= args.eval_freq
            evaluations.append(evaluate_policy(policy))

            if args.save_models and max(evaluations) == evaluations[-1]: 
                policy.save(file_name,directory=base_path + "/pytorch_models")
            np.save(os.path.join(base_path, "results", file_name), evaluations)
    
    """
    #############################################
    #############################################
    #############################################
    """
    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    if args.save_models and max(evaluations) == evaluations[-1]: 
        policy.save(file_name, directory=base_path + "/pytorch_models")
    np.save(os.path.join(base_path, "results", file_name), evaluations)
    """


    #############################################
    ################   DDPG-HER   ###############
    #############################################
    """
    for ep in range(total_episode):
        sample goal g and initial state s0
        for t in range(total_steps):
            at = policy(st||g)
            stt = env.step(st, at)
            rt = reward(st, at, g)
            store_transition(st||g, at, rt, stt||g)

            G = sample_goals
            for g_new in G:
                r_new = reward(st, at, g_new)
                store_transition(st||g_new, at, r_new, stt||g_new)

            if memory.pointer > memory.capacity:
                sample batch 
                train model 
# 
    """
    #############################################
    #############################################
    #############################################







    #############################################
    #########        Ankish Bansal         ######
    #########          TD3 + HER           ######
    #############################################
    """
        obs is dictionary:  
            1. ss:  states
            2. sg: goal position

        n_goal_sample: hyperparameter
        r_high       : env max reward
        n_train      : number of training steps
    
    """
    evaluations = []
    n_goal_sample = 10
    r_high = env._reaching_rew
    n_train = 1
    MAX_EP_STEPS = 100
    args.max_timesteps = 1e6
    for i in range(int(args.max_timesteps)):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_timesteps = 0
        collect_ = []

        # st, sg, sjt = obs['state'], obs['goal'], obs['joint']
        for j in range(MAX_EP_STEPS):
            print("="*50)
            print(f"ep {i} timestep {j}")
            st, sg, sjt = obs['state'], obs['goal'], obs['joint']
            sst = np.hstack((st, sg, sjt))
            at = policy.select_action(sst)
            # print(sst,at) 
            args.expl_noise = 0.1
            at = np.random.normal(at, args.expl_noise)
            # at = np.clip(at, env.action_space.low, env.action_space.high)
            at = np.clip(at, -1, 1)

            obss = copy.deepcopy(obs)
            # make a step with current state and action
            new_obs, rt, done,_ = env.step(at, sst)
            stt = new_obs['state']
            sjtt = new_obs['joint']

            # compute flag: done
            done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
            episode_timesteps += 1
            
            # print(f"ep: {j}, \n state: {st} \n next_state : {stt} \n")
            # collect everything in bucket
            collect_.append((st, sjt, at, rt, stt, sjtt, done_bool, sg))

            # f_save.write(f"ep: {j}, \n obs : {np.round(obss['state'],2)}, \n new_obs {np.round(new_obs['state'],2)}, \n action: {np.round(at,2)} \n")
            print(f"ep: {j}, \n obs : {np.round(obss['state'],2)}, \n new_obs {np.round(new_obs['state'],2)}, \n action: {np.round(at,2)} \n")
            print(f"joint: {np.round(obss['joint'],2)}, \n new joint: {np.round(new_obs['joint'],2)} \n ")
            # collect reward
            episode_reward += rt
            obs = new_obs

            # if done, break loop
            if done:
                break


        for j in range(MAX_EP_STEPS):
            st, sjt, at, rt, stt, sjtt, done_bool, gt = collect_[j]
            print("*"*50)
            print(f"ep: {j} \n state: {st} \n new-state: {stt} \n joint: {sjt} \n new-joint: {sjtt} \n")
            # Store original sample in replay buffer
            sgt  = np.hstack((st, sg, sjt))  # (current state, goal)
            sgtt = np.hstack((stt, sg, sjtt)) # (next state, goal)
            replay_buffer.add((sgt, sgtt, at, rt, done_bool))
            # replay_buffer.add((obs, new_obs, action, reward, done_bool))

            # now we sample goal(future position) and store in buffer
            # G = env.sample_goals()
            size = len(collect_)
            for k in range(n_goal_sample):
                sample_idx = np.random.randint(low=j, high=size)
                _, _, _, _, _, _, _, g_new = collect_[j]
                g_new = collect_[j][-1]

                # Store data in replay buffer
                sgt  = np.hstack((st, g_new, sjt))  # (current state, goal)
                sgtt = np.hstack((stt, g_new, sjtt)) # (next state, goal)
                replay_buffer.add((sgt, sgtt, at, r_high, done_bool))


        for j in range(n_train):
            policy.train(
                replay_buffer, 
                1, 
                args.batch_size, 
                args.discount, 
                args.tau, 
                args.policy_noise, 
                args.noise_clip, 
                args.policy_freq
            )

        if (i+1)%10 == 0:
            evaluations.append(evaluate_policy(policy))
            # if (j+1)%2 == 0:
            #     # Evaluate episode
            #     evaluations.append(evaluate_policy(policy))

            # save weights

    #############################################
    #############################################
    #############################################

    f_save.close()