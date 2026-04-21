"""
Self-play training script for the AI Football RL agent.
The learning policy plays from the left side against a frozen opponent snapshot.
"""

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RL_Environment import FootballEnvironment
from training_simulation import TrainingMatchSimulator


class SelfPlayEnvironment(gym.Env):
    """Self-play environment backed by the offline training simulator."""

    def __init__(self, opponent_model=None):
        super().__init__()
        self.opponent_model = opponent_model
        self.simulator = TrainingMatchSimulator()
        self.env_team1 = FootballEnvironment(lambda: self.simulator.get_state('left'), player_index=0)
        self.env_team2 = FootballEnvironment(lambda: self.simulator.get_state('right'), player_index=0)
        self.action_space = self.env_team1.action_space
        self.observation_space = self.env_team1.observation_space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.simulator.reset()
        obs, _ = self.env_team1.reset(seed=seed)
        self.env_team2.reset(seed=seed)
        return obs, {}

    def step(self, action):
        opponent_state = self.simulator.get_state('right')

        if self.opponent_model is not None:
            opponent_obs = self.env_team2.observation_from_state(opponent_state)
            opponent_action, _ = self.opponent_model.predict(opponent_obs, deterministic=True)
        else:
            opponent_action = self.simulator.baseline_action(opponent_state)

        self.simulator.step(action, opponent_action)
        team1_state = self.simulator.get_state('left')

        obs = self.env_team1.observation_from_state(team1_state)
        reward = self.env_team1._calculate_reward(
            team1_state['our_team'],
            team1_state['their_team'],
            team1_state['ball'],
            team1_state['our_score'],
            team1_state['their_score'],
            team1_state['time_left'],
            team1_state['total_time'],
            team1_state['your_side'],
            team1_state['half'],
        )
        terminated = team1_state['time_left'] <= 0
        truncated = False
        return obs, reward, terminated, truncated, {}


def train_self_play_agent(total_timesteps=200000, model_path="./models/rl_football_agent"):
    """
    Train an RL agent using self-play.

    Args:
        total_timesteps: Total training steps
        model_path: Path to an optional base model used as the frozen opponent
    """

    print("=" * 60)
    print("AI FOOTBALL - SELF-PLAY RL TRAINING")
    print("=" * 60)

    try:
        opponent_model = PPO.load(model_path)
        print(f"Loaded existing model from {model_path}")
    except Exception:
        opponent_model = None
        print("No existing model found, starting from a fresh policy.")

    env = SelfPlayEnvironment(opponent_model=opponent_model)

    if opponent_model is not None:
        model = PPO.load(model_path, env=env)
    else:
        model = PPO('MlpPolicy', env, verbose=1)

    os.makedirs("./models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./models/",
        name_prefix="selfplay_rl_model",
    )

    print(f"\nTraining with self-play for {total_timesteps} timesteps...")
    print("The learner controls the left side against a frozen opponent snapshot.")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
        )
        print("\nSelf-play training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return None

    final_model_path = "./models/selfplay_rl_football_agent"
    model.save(final_model_path)
    print(f"Self-play model saved to {final_model_path}.zip")

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train RL agent with self-play')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--model', type=str, default="./models/rl_football_agent", help='Path to base model')

    args = parser.parse_args()

    model = train_self_play_agent(
        total_timesteps=args.timesteps,
        model_path=args.model,
    )

    if model:
        print("\n" + "=" * 60)
        print("Self-play training completed!")
        print("Both teams now use the same RL model family.")
        print("=" * 60)
