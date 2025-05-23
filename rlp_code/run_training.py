#!/usr/bin/env python

import os
import fnmatch
import json

from collections import defaultdict

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
import pygame

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import QRDQN, RecurrentPPO, MaskablePPO, TRPO

import rlp
from muzero.muzero import MuZero
from dreamerv3.example import main as dreamerv3_main


class SaveOnBestEpisodeLengthCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self, check_freq: int, log_dir: str, puzzle_name: str, verbose: int = 1
    ):
        super(SaveOnBestEpisodeLengthCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, f"best_model_{puzzle_name}")
        self.best_mean_length = np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            # compute exact episode lengths
            x[1:] = x[1:] - x[0:-1]
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_length = np.mean(x[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean length: {self.best_mean_length:.2f} - Last mean length per episode: {mean_length:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_length < self.best_mean_length:
                    self.best_mean_length = mean_length
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

class PuzzleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_state = None
        self.last_laserno = None
        self.last_nguesses = None
        self.steps_without_progress = 0
        self.episode_count = 0
        self.action_counts = defaultdict(int)  # Track action frequencies

    def step(self, action):
        self.action_counts[action] += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        puzzle_state = info.get("puzzle_state")
        if puzzle_state is None:
            raise ValueError("Missing puzzle state")
        
        # Print full state every 100 steps for debugging
        if self.episode_count > 100 and self.episode_count % 100 == 0:
            print(f"\nFull State:\n{puzzle_state}\n")
        
        custom_reward = self.calculate_reward(puzzle_state)
        
        self.last_laserno = puzzle_state['laserno']
        self.last_nguesses = puzzle_state['nguesses']
        self.previous_state = puzzle_state.copy()
        
        if terminated or truncated:
            self.episode_count += 1
            print(f"\nEpisode {self.episode_count} Summary:")
            print(f"Actions taken: {dict(self.action_counts)}")
            self.action_counts.clear()
            
        return obs, custom_reward, terminated, truncated, info

    '''
    def calculate_reward(self, puzzle_state):
        reward = 0
        debug_msg = []
        
        # Large reward for solving
        if puzzle_state['done']:
            reward += 20.0
            debug_msg.append("SOLVED (+20.0)")
            print("\n".join(debug_msg))
            return reward
        
        # Correct guesses
        if puzzle_state['nright'] > 0:
            r = puzzle_state['nright'] * 2.0
            reward += r
            debug_msg.append(f"CORRECT (+{r})")
        
        # Wrong guesses (should trigger if nwrong increases)
        if puzzle_state['nwrong'] > 0:
            if self.previous_state and puzzle_state['nwrong'] > self.previous_state['nwrong']:
                penalty = 3.0
                reward -= penalty
                debug_msg.append(f"WRONG (-{penalty})")
        
        # Missed balls (should trigger if nmissed increases)
        if puzzle_state['nmissed'] > 0:
            if self.previous_state and puzzle_state['nmissed'] > self.previous_state['nmissed']:
                penalty = 3.0
                reward -= penalty
                debug_msg.append(f"MISSED (-{penalty})")
        
        # Efficiency penalty
        if (self.last_laserno is not None and 
            self.last_nguesses is not None and
            puzzle_state['laserno'] == self.last_laserno and 
            puzzle_state['nguesses'] == self.last_nguesses):
            penalty = 0.5
            reward -= penalty
            #debug_msg.append(f"REPEATED ACTION (-{penalty})")
        
        # Step penalty
        reward -= 0.2
        #debug_msg.append(f"STEP (-0.2)")
        
        # Progress tracking
        if self.check_progress(puzzle_state):
            reward += 0.3
            debug_msg.append("PROGRESS (+0.3)")
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1
            if self.steps_without_progress > 10:
                penalty = 0.5
                reward -= penalty
                #debug_msg.append(f"NO PROGRESS (-{penalty})")
        
        # Print all debug messages
        if debug_msg:
            print(" | ".join(debug_msg))
            
        return reward
        '''

    def check_progress(self, puzzle_state):
        if not self.previous_state:
            return False
            
        return (puzzle_state['nright'] > self.previous_state['nright'] or
                puzzle_state['nwrong'] < self.previous_state['nwrong'] or
                puzzle_state['nmissed'] < self.previous_state['nmissed'])

# def make_reward_env(timelimit: int = 15000, **kwargs: dict) -> gym.Env:
#     # Pass timelimit as a keyword argument if the environment supports it
#     env = gym.make("rlp/Puzzle-v0", timelimit=timelimit, **kwargs)
#     if kwargs.get("obs_type") == "puzzle_state":
#         env = FlattenObservation(env)

#     return env
from gymnasium.wrappers import TimeLimit

def make_reward_env(timelimit: int = 15000, **kwargs: dict) -> gym.Env:
    # Create the environment without the timelimit argument
    env = gym.make("rlp/Puzzle-v0", **kwargs)

    # Apply the TimeLimit wrapper to enforce a maximum number of timesteps
    env = TimeLimit(env, max_episode_steps=timelimit)
    #env = PuzzleRewardWrapper(env)
    if kwargs.get("obs_type") == "puzzle_state":
        env = FlattenObservation(env)

    return env


def run_training_muzero(args):
    config = {
        "puzzle": args.puzzle,
        "render_mode": "rgb_array",
        "params": args.arg,
        "num_workers": 32,
        "obs_type": args.obs_type,
        "seed": args.seed,
        "results_path": f"./{args.algorithm}/results/{args.puzzle}_{args.arg}_{args.obs_type}/",
    }

    # Train
    muzero = MuZero("rlp", config, write_config=True)
    muzero.train()

    # Test
    muzero = MuZero("rlp", config, write_config=False)
    muzero.load_model(os.path.join(config["results_path"], "model.checkpoint"))
    muzero.test_multihread(
        render=False,
        num_tests=1000,
        num_workers=config["num_workers"],
        results_dir=config["results_path"],
    )


def run_training_dreamerv3(args):
    sys.argv = [
        "example.py",
        args.puzzle,
        args.arg,
        str(args.seed),
        f"./results/{args.puzzle}_{args.arg}_{args.obs_type}/",
        args.timelimit,
        args.max_state_repeats,
    ]
    dreamerv3_main()

    sys.argv = [
        "example.py",
        "test",
        f"./results/{args.puzzle}_{args.arg}_{args.obs_type}/",
        args.puzzle,
        args.arg,
        str(args.seed),
    ]
    dreamerv3_main()


def run_stable_baseline3(args):
    data_dir = f"./results/"

    if args.allowundo:
        undo_prefix = "undo"
    else:
        undo_prefix = "noundo"

    log_dir = f"{data_dir}monitor/{args.algorithm}_{args.timesteps}/{args.puzzle}_{args.arg}_{undo_prefix}_{args.obs_type}/"
    model_dir = f"{data_dir}models/{args.algorithm}_{args.timesteps}/{args.puzzle}_{args.arg}_{args.obs_type}_reward/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"log_dir = {log_dir}")
    print(f"model_dir = {model_dir}")

    render_mode = "human" if not args.headless else "rgb_array"

    if args.algorithm == "RecurrentPPO":
        policy_type = (
            "MlpLstmPolicy"
            if args.obs_type == "puzzle_state"
            else "MultiInputLstmPolicy"
        )
    else:
        policy_type = (
            "MlpPolicy" if args.obs_type == "puzzle_state" else "MultiInputPolicy"
        )

    env = make_vec_env(
        make_reward_env,
        env_kwargs=dict(
            puzzle=args.puzzle,
            render_mode=render_mode,
            params=args.arg,
            allow_undo=False,
            include_cursor_in_state_info=True,
            obs_type=args.obs_type,
            max_state_repeats=args.max_state_repeats,
        ),
        n_envs=1,
        monitor_dir=log_dir,
    )

    

    # Logs will be saved in log_dir/monitor.csv

    callback = SaveOnBestEpisodeLengthCallback(
        check_freq=1000, log_dir=log_dir, puzzle_name=args.puzzle
    )

    buffer_size = 1000000  # for off-policy algorithms

    model: PPO | DQN | A2C | QRDQN | RecurrentPPO | TRPO | MaskablePPO
    if args.algorithm == "PPO":
        model = PPO(policy_type, env, verbose=1)
    elif args.algorithm == "DQN":
        model = DQN(policy_type, env, verbose=1, buffer_size=buffer_size)
    elif args.algorithm == "A2C":
        model = A2C(policy_type, env, verbose=1)
    elif args.algorithm == "QRDQN":
        model = QRDQN(policy_type, env, verbose=1, buffer_size=buffer_size)
    elif args.algorithm == "RecurrentPPO":
        model = RecurrentPPO(policy_type, env, verbose=1)
    elif args.algorithm == "TRPO":
        model = TRPO(policy_type, env, verbose=1)
    elif args.algorithm == "MaskablePPO":
        model = MaskablePPO(policy_type, env, verbose=1)
    else:
        raise Exception("No supported RL algorithm chosen")

    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    print("Training complete. Proceeding to save and evaluate the model...")
    # Save the final agent
    model.save(model_dir)
    print("model saved")
    # Load the best agent
    model = model.load(os.path.join(log_dir, f"best_model_{args.puzzle}"))
    print("model loaded")
    res = evaluate_policy(
        model,
        env,
        n_eval_episodes=3,
        render=False,
        return_episode_rewards=True,
        deterministic=True,
    )
    print("evaluation complete")

    average_reward = np.mean(res[0])
    average_episode_length = np.mean(res[1])
    average_success_episode_length = np.mean(res[1][res[0] > 0])
    std_dev_episode_length = np.std(res[1])
    success_rate = np.where(res[0] > 0, 1.0, 0.0).mean()

    results_data = {
        "Average Reward": average_reward,
        "Average Episode Length": average_episode_length,
        "Average Success Episode Length": average_success_episode_length,
        "Std Dev Episode Length": std_dev_episode_length,
        "Success Rate": success_rate,
    }
    results_file_path = os.path.join(log_dir, "evaluation_results.json")
    with open(results_file_path, "w") as file:
        json.dump(results_data, file, indent=4)
    pygame.quit()


if __name__ == "__main__":
    parser = rlp.puzzle.make_puzzle_parser()
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Number of timesteps during training"
    )
    parser.add_argument(
        "-tl", "--timelimit", type=int, help="Max timesteps per episode", default=10000
    )
    parser.add_argument(
        "-alg",
        "--algorithm",
        type=str,
        help="Choice of RL Algorithm used for training",
        choices=[
            "PPO",
            "DQN",
            "A2C",
            "HER",
            "ARS",
            "QRDQN",
            "RecurrentPPO",
            "TRPO",
            "MaskablePPO",
            "MuZero",
            "DreamerV3",
        ],
        default="PPO",
    )
    parser.add_argument(
        "-ot",
        "--obs-type",
        type=str,
        help="Type of observation",
        choices=["rgb", "puzzle_state"],
        default="puzzle_state",
    )
    parser.add_argument("--seed", "-seed", type=int, help="Random seed", default=0)
    parser.add_argument(
        "-sr",
        "--max_state_repeats",
        type=int,
        help="Max number of state repeats",
        default=1000000,
    )

    args = parser.parse_args()
    args.size = (128, 128)
    print(f"Args: {args}")

    if args.algorithm == "MuZero":
        run_training_muzero(args)
    elif args.algorithm == "DreamerV3":
        run_training_dreamerv3(args)
    else:
        run_stable_baseline3(args)
