# #!/usr/bin/env python
# import gymnasium as gym

# import rlp


# class PuzzleRewardWrapper(gym.Wrapper):
#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)

#         # not a meaningful example, but shows how to use
#         # the internal puzzle state for intermediate rewards
#         #reward -= info["puzzle_state"]["w"]

#         puzzle_state = info.get("puzzle_state")
        
#         if puzzle_state is None:
#             raise ValueError("Puzzle state information is missing from the environment.")
        
#         # Calculate progress toward solving the puzzle
#         progress = self.calculate_progress(puzzle_state)
        
#         # Check if no progress was made in the last step
#         if self.previous_state is not None and self.check_no_progress(puzzle_state, self.previous_state):
#             reward -= 0.1  # Penalize for redundant or inefficient actions
        
#         # Update previous state to current state for next step
#         self.previous_state = puzzle_state

#         # Apply the progress-based reward shaping
#         reward += progress


#         return obs, reward, terminated, truncated, info

# def calculate_progress(self, puzzle_state):
#         """
#         Calculate how close the agent is to solving the Blackbox puzzle.
#         You can use a heuristic like counting how many elements are in their correct places.
#         """
#         correct_positions = sum(1 for cell in puzzle_state if self.is_correct(cell))  # Example logic
#         total_positions = len(puzzle_state)  # Total number of cells in puzzle_state
#         progress = correct_positions / total_positions
#         return progress

# def is_correct(self, cell):
#         """
#         Define logic to check if a given cell is in the correct state.
#         This could be a check against the target solution or some condition.
#         """
#         return cell['correct']  # Example: if the cell is in the correct position

# def check_no_progress(self, puzzle_state, previous_state):
#         """
#         Check if the agent made no progress by comparing the current state with the previous state.
#         """
#         return puzzle_state == previous_state  # If the state is identical, there's no progress


# if __name__ == "__main__":
#     env = gym.make("rlp/Puzzle-v0", puzzle="fifteen", render_mode="human", params="2x2")
#     wrapped_env = PuzzleRewardWrapper(env)

#     observation, info = wrapped_env.reset(seed=42)
#     for step in range(10000):
#         action = env.action_space.sample()  # a random policy
#         observation, reward, terminated, truncated, info = wrapped_env.step(action)
#         print(f"Step {step}: Action {action}, Reward: {reward}")

#         if terminated or truncated:
#             observation, info = wrapped_env.reset()
#             print(f"Resetting")
# wrapped_env.close()

#!/usr/bin/env python
import gymnasium as gym
import rlp


class PuzzleRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.previous_state = None  # To keep track of the previous puzzle state

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        puzzle_state = info.get("puzzle_state")
        
        if puzzle_state is None:
            raise ValueError("Puzzle state information is missing from the environment.")
        
        # Calculate progress toward solving the puzzle
        progress = self.calculate_progress(puzzle_state)
        
        # Check if no progress was made in the last step
        if self.previous_state is not None and self.check_no_progress(puzzle_state, self.previous_state):
            reward -= 0.1  # Penalize for redundant or inefficient actions
        
        # Update previous state to current state for next step
        self.previous_state = puzzle_state

        # Apply the progress-based reward shaping
        reward += progress
        
        return obs, reward, terminated, truncated, info

    def calculate_progress(self, puzzle_state):
        """
        Calculate how close the agent is to solving the Blackbox puzzle.
        You can use a heuristic like counting how many elements are in their correct places.
        """
        correct_positions = sum(1 for cell in puzzle_state if self.is_correct(cell))  # Example logic
        total_positions = len(puzzle_state)  # Total number of cells in puzzle_state
        progress = correct_positions / total_positions
        return progress

    def is_correct(self, cell):
        """
        Define logic to check if a given cell is in the correct state.
        This could be a check against the target solution or some condition.
        """
        return cell['correct']  # Example: if the cell is in the correct position

    def check_no_progress(self, puzzle_state, previous_state):
        """
        Check if the agent made no progress by comparing the current state with the previous state.
        """
        return puzzle_state == previous_state  # If the state is identical, there's no progress


if __name__ == "__main__":
    env = gym.make("rlp/Puzzle-v0", puzzle="fifteen", render_mode="human", params="2x2")
    wrapped_env = PuzzleRewardWrapper(env)

    observation, info = wrapped_env.reset(seed=42)
    for step in range(10000):
        action = env.action_space.sample()  # A random policy
        observation, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {step}: Action {action}, Reward: {reward}")

        if terminated or truncated:
            observation, info = wrapped_env.reset()
            print(f"Resetting")

    wrapped_env.close()
