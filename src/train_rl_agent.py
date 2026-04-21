"""
Training script for RL agent in AI Football
Run this to train your RL agent, then use the model in Manager.py
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from RL_Environment import FootballEnvironment


class SimpleGameWrapper:
    """Wrapper to integrate with actual game loop"""
    
    def __init__(self, headless=True):
        """
        Initialize game wrapper
        headless: If True, runs without pygame display
        """
        self.headless = headless
        self.current_state = None
        self.game_counter = 0
        
    def set_state(self, our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
        """Called by the game to update state"""
        self.current_state = {
            'our_team': our_team,
            'their_team': their_team,
            'ball': ball,
            'your_side': your_side,
            'half': half,
            'time_left': time_left,
            'our_score': our_score,
            'their_score': their_score,
            'total_time': 45 * 60
        }
        return self.current_state
    
    def get_state(self):
        """Returns current game state"""
        return self.current_state


def train_agent(total_timesteps=100000, learning_rate=3e-4, n_steps=2048):
    """
    Train the RL agent
    
    Args:
        total_timesteps: Total training steps
        learning_rate: Learning rate for PPO
        n_steps: Number of steps before update
    """
    
    print("=" * 60)
    print("AI FOOTBALL - RL AGENT TRAINING")
    print("=" * 60)
    
    # Create game wrapper
    game_wrapper = SimpleGameWrapper(headless=True)
    
    # Create environment
    def get_state():
        return game_wrapper.get_state()
    
    env = FootballEnvironment(get_state, player_index=0)
    
    # Create agent with HIGH RESOURCE USAGE for FASTER TRAINING
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,           # Large batch size
        batch_size=512,            # Larger batch processing
        n_epochs=20,               # More optimization passes per update
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,             # Entropy bonus for exploration
        vf_coef=0.5,               # Value function coefficient
        max_grad_norm=0.5,         # Gradient clipping
        use_sde=False,             # Disable state-dependent exploration for speed
        sde_sample_freq=-1,
        target_kl=None,            # No KL divergence limit for faster learning
        verbose=1,
        tensorboard_log=None,      # Disable logging for speed
        device='auto',             # Use GPU if available
    )
    
    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="rl_model"
    )
    
    print(f"\nTraining for {total_timesteps} timesteps...")
    print(f"Learning rate: {learning_rate}")
    print(f"Steps per update: {n_steps}\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
        )
        print("\n✓ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None
    
    # Save final model
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/rl_football_agent"
    model.save(model_path)
    print(f"✓ Model saved to {model_path}.zip")
    
    return model


def load_trained_agent(model_path="./models/rl_football_agent"):
    """Load a trained agent"""
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
        n_steps=args.n_steps
    )
    
    if model:
        print("\n" + "=" * 60)
        print("Agent trained and saved!")
        print("You can now use it in Manager.py")
        print("=" * 60)
