"""
Training script for RL agent in AI Football.
Run this to train your RL agent, then use the model in Manager.py.
"""

import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RL_Environment import FootballEnvironment
from training_simulation import TrainingMatchSimulator


def train_agent(total_timesteps=100000, learning_rate=3e-4, n_steps=2048):
    """
    Train the RL agent.

    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps before update
    """

    print("=" * 60)
    print("AI FOOTBALL - RL AGENT TRAINING")
    print("=" * 60)

    simulator = TrainingMatchSimulator()

    def get_state():
        return simulator.get_state('left')

    env = FootballEnvironment(get_state, player_index=0)
    original_reset = env.reset
    original_step = env.step

    def reset_with_simulation(seed=None):
        simulator.reset()
        return original_reset(seed=seed)

    def step_with_simulation(action):
        simulator.step(action)
        return original_step(action)

    env.reset = reset_with_simulation
    env.step = step_with_simulation

    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=512,
        n_epochs=20,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        verbose=1,
        tensorboard_log=None,
        device='auto',
    )

    os.makedirs("./models", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="rl_model",
    )

    print(f"\nTraining for {total_timesteps} timesteps...")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps per update: {n_steps}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
        )
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        return None

    model_path = "./models/rl_football_agent"
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    return model


def load_trained_agent(model_path="./models/rl_football_agent"):
    """Load a trained agent."""
    return PPO.load(model_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train RL agent for AI Football')
    parser.add_argument('--timesteps', type=int, default=100000, help='Total training timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--n_steps', type=int, default=2048, help='Steps per update')

    args = parser.parse_args()

    model = train_agent(
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
    )

    if model:
        print("\n" + "=" * 60)
        print("Agent trained and saved!")
        print("You can now use it in Manager.py")
        print("=" * 60)
