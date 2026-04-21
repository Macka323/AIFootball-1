"""
Self-play training script for RL agent in AI Football
The agent trains by playing against itself - both teams use the same RL model
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


class SelfPlayEnvironment(gym.Env):
    """
    Self-play environment where both teams use the same RL model
    """
    
    def __init__(self, model_path="./models/rl_football_agent"):
        super().__init__()
        
        # Load the trained model
        self.model = PPO.load(model_path)
        
        # Create two environments - one for each team
        self.env_team1 = FootballEnvironment(lambda: self.get_team_state(0), player_index=0)
        self.env_team2 = FootballEnvironment(lambda: self.get_team_state(1), player_index=0)
        
        # Action and observation spaces (same as single environment)
        self.action_space = self.env_team1.action_space
        self.observation_space = self.env_team1.observation_space
        
        self.current_team = 0  # 0 for team1, 1 for team2
        self.team1_state = None
        self.team2_state = None
        self.step_count = 0
        
    def get_team_state(self, team_idx):
        """Get state for specific team"""
        return self.team1_state if team_idx == 0 else self.team2_state
    
    def reset(self, seed=None):
        """Reset both teams"""
        super().reset(seed=seed)
        self.step_count = 0
        self.current_team = 0
        
        # Reset both environments with dummy states initially
        obs1, _ = self.env_team1.reset()
        obs2, _ = self.env_team2.reset()
        
        return obs1, {}
    
    def step(self, action):
        """Execute step for current team, then switch to other team"""
        self.step_count += 1
        
        # Get current game state (would be provided by actual game)
        # For now, we'll simulate with dummy data
        game_state = self._simulate_game_state()
        
        # Update both team states
        self.team1_state = game_state.copy()
        self.team2_state = game_state.copy()
        
        # Current team takes action
        if self.current_team == 0:
            obs, reward, terminated, truncated, info = self.env_team1.step(action)
        else:
            obs, reward, terminated, truncated, info = self.env_team2.step(action)
        
        # Switch teams
        self.current_team = 1 - self.current_team
        
        # If it's team2's turn now, get its observation
        if self.current_team == 1 and not terminated:
            obs, _, _, _, _ = self.env_team2.step(np.zeros(2))  # Dummy action for observation
        
        return obs, reward, terminated, truncated, info
    
    def _simulate_game_state(self):
        """Simulate a basic game state for training"""
        return {
            'our_team': [
                {'x': 200, 'y': 300, 'alpha': 0, 'a_max': 100, 'v_max': 100, 'radius': 20, 'weight': 75, 'shot_power_max': 50},
                {'x': 200, 'y': 400, 'alpha': 0, 'a_max': 80, 'v_max': 80, 'radius': 22, 'weight': 80, 'shot_power_max': 60},
                {'x': 200, 'y': 500, 'alpha': 0, 'a_max': 70, 'v_max': 85, 'radius': 24, 'weight': 85, 'shot_power_max': 40}
            ],
            'their_team': [
                {'x': 1000, 'y': 300, 'alpha': np.pi, 'a_max': 100, 'v_max': 100, 'radius': 20, 'weight': 75, 'shot_power_max': 50},
                {'x': 1000, 'y': 400, 'alpha': np.pi, 'a_max': 80, 'v_max': 80, 'radius': 22, 'weight': 80, 'shot_power_max': 60},
                {'x': 1000, 'y': 500, 'alpha': np.pi, 'a_max': 70, 'v_max': 85, 'radius': 24, 'weight': 85, 'shot_power_max': 40}
            ],
            'ball': {'x': 600, 'y': 384, 'vx': 0, 'vy': 0},
            'your_side': 'left' if self.current_team == 0 else 'right',
            'half': 1,
            'time_left': 45 * 60 - self.step_count,
            'our_score': 0,
            'their_score': 0,
            'total_time': 45 * 60
        }


def train_self_play_agent(total_timesteps=200000, model_path="./models/rl_football_agent"):
    """
    Train RL agent using self-play
    
    Args:
        total_timesteps: Total training steps
        model_path: Path to existing model (will create new one if doesn't exist)
    """
    
    print("=" * 60)
    print("AI FOOTBALL - SELF-PLAY RL TRAINING")
    print("=" * 60)
    
    # Try to load existing model, or create new one
    try:
        model = PPO.load(model_path)
        print(f"✓ Loaded existing model from {model_path}")
    except:
        print("⚠ No existing model found, creating new one...")
        # Create dummy environment for initial model
        dummy_env = FootballEnvironment(lambda: None, player_index=0)
        model = PPO('MlpPolicy', dummy_env, verbose=1)
    
    # Create self-play environment
    env = SelfPlayEnvironment(model_path)
    
    # Update model to use self-play environment
    model.set_env(env)
    
    # Callback to save checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./models/",
        name_prefix="selfplay_rl_model"
    )
    
    print(f"\nTraining with self-play for {total_timesteps} timesteps...")
    print("Both teams will use the same RL model")
    
    # Train
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_callback,
        )
        print("\n✓ Self-play training completed successfully!")
    except KeyboardInterrupt:
        print("\n✗ Training interrupted by user")
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return None
    
    # Save final model
    os.makedirs("./models", exist_ok=True)
    final_model_path = "./models/selfplay_rl_football_agent"
    model.save(final_model_path)
    print(f"✓ Self-play model saved to {final_model_path}.zip")
    
    return model


def create_self_play_manager():
    """Create a manager that uses the same RL model for both teams"""
    
    manager_code = '''
import numpy as np
import os
from stable_baselines3 import PPO

# ============================================================================
# SELF-PLAY RL CONFIGURATION
# ============================================================================
rl_model = None
rl_ready = False

def load_rl_model():
    """Load trained self-play RL model"""
    global rl_model, rl_ready
    
    model_path = os.path.join(os.path.dirname(__file__), '../../models/selfplay_rl_football_agent')
    
    try:
        if os.path.exists(model_path + '.zip'):
            rl_model = PPO.load(model_path)
            rl_ready = True
            print("✓ Self-play RL Model loaded successfully!")
            return True
        else:
            rl_ready = False
            print("⚠ Self-play RL model not found. Train it first with: python train_selfplay_rl_agent.py")
            return False
    except Exception as e:
        rl_ready = False
        print(f"⚠ Failed to load RL model: {e}")
        return False

def get_rl_observation(player, ball, other_players, score_diff, time_left):
    """Convert game state to RL observation vector (19 features)"""
    
    # Calculate distance and angle to ball
    dx = ball['x'] - player['x']
    dy = ball['y'] - player['y']
    distance_to_ball = np.sqrt(dx**2 + dy**2)
    angle_to_ball = np.arctan2(dy, dx)
    
    # Strength comparison - use shot_power_max or shot_power
    player_shot = player.get('shot_power_max', player.get('shot_power', 50))
    other_shot = other_players[0].get('shot_power_max', other_players[0].get('shot_power', 50))
    strength_diff = (player_shot - other_shot) / 100.0
    
    obs = np.array([
        # Player state (10 features)
        player['x'] / 1366.0,
        player['y'] / 768.0,
        player.get('vx', 0) / 100.0,
        player.get('vy', 0) / 100.0,
        player['alpha'] / (2 * np.pi),
        player.get('a_max', 100) / 100.0,
        player.get('v_max', 100) / 100.0,
        player.get('radius', 20) / 50.0,
        player.get('weight', 75) / 100.0,
        player_shot / 100.0,
        
        # Ball state (4 features)
        ball['x'] / 1366.0,
        ball['y'] / 768.0,
        ball.get('vx', 0) / 50.0,
        ball.get('vy', 0) / 50.0,
        
        # Game state (5 features)
        distance_to_ball / 1000.0,
        (angle_to_ball + np.pi) / (2 * np.pi),
        time_left / (45 * 60),
        strength_diff,
        0.0,
    ], dtype=np.float32)
    
    return obs

def action_to_decision(action, player):
    """Convert RL action (normalized 0-1) to decision dict"""
    try:
        # Ensure action is numpy array and handle NaN/inf
        action = np.array(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.5, posinf=1.0, neginf=0.0)
        action = np.clip(action, 0.0, 1.0)
        
        alpha = float(action[0] * 2 * np.pi)
        force = float(action[1] * player['a_max'])
        
        # Validate outputs
        if not np.isfinite(alpha):
            alpha = player['alpha']
        if not np.isfinite(force):
            force = 0.5 * player['a_max']
        
        decision = {
            'alpha': alpha,
            'force': force,
            'shot_request': False,
            'shot_power': 50,
        }
        
        return decision
    except Exception as e:
        print(f"⚠ Action conversion failed: {e}")
        return {
            'alpha': player['alpha'],
            'force': 0.5 * player['a_max'],
            'shot_request': False,
            'shot_power': 50,
        }

# Choose names for your players and team
def team_properties():
    properties = dict()
    player_names = ["Пандевалдо", "Панчевалдо", "Елмасалдо"]
    properties['team_name'] = "Мак Челзи"
    properties['player_names'] = player_names
    properties['image_name'] = 'Red.png'
    properties['weight_points'] = (9, 10, 15)
    properties['radius_points'] = (5, 10, 20)
    properties['max_acceleration_points'] = (40, 10, 15)
    properties['max_speed_points'] = (40, 10, 25)
    properties['shot_power_points'] = (18, 20, 13)
    return properties

# This function gathers game information and controls each one of your three players
def decision(our_team, their_team, ball, your_side, half, time_left, our_score, their_score):
    
    # Load RL model on first call
    global rl_model, rl_ready
    if rl_model is None and not rl_ready:
        load_rl_model()
    
    manager_decision = [dict(), dict(), dict()]
    score_diff = our_score - their_score
    
    for i in range(3):
        player = our_team[i]
        
        if rl_ready:
            # Use RL agent for this player
            try:
                obs = get_rl_observation(player, ball, their_team, score_diff, time_left)
                action, _ = rl_model.predict(obs, deterministic=True)
                manager_decision[i] = action_to_decision(action, player)
            except Exception as e:
                # Fallback to simple strategy if RL fails
                print(f"⚠ RL prediction failed for player {i}: {e}")
                manager_decision[i]['alpha'] = player['alpha']
                manager_decision[i]['force'] = 0.5 * player['a_max']
                manager_decision[i]['shot_request'] = False
                manager_decision[i]['shot_power'] = 50
        else:
            # Simple baseline strategy
            manager_decision[i]['alpha'] = player['alpha']
            manager_decision[i]['force'] = 0.7 * player['a_max']
            manager_decision[i]['shot_request'] = False
            manager_decision[i]['shot_power'] = 50
    
    return manager_decision
'''
    
    # Save to both team directories
    os.makedirs("./src/Team_name", exist_ok=True)
    os.makedirs("./src/Test_team", exist_ok=True)
    
    with open("./src/Team_name/Manager.py", "w") as f:
        f.write(manager_code)
    
    with open("./src/Test_team/Manager.py", "w") as f:
        f.write(manager_code)
    
    print("✓ Created self-play managers for both teams")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RL agent with self-play')
    parser.add_argument('--timesteps', type=int, default=200000, help='Total training timesteps')
    parser.add_argument('--model', type=str, default="./models/rl_football_agent", help='Path to base model')
    
    args = parser.parse_args()
    
    # Train the agent
    model = train_self_play_agent(
        total_timesteps=args.timesteps,
        model_path=args.model
    )
    
    if model:
        print("\n" + "=" * 60)
        print("Self-play training completed!")
        print("Creating self-play managers...")
        create_self_play_manager()
        print("Both teams now use the same RL model!")
        print("=" * 60)
