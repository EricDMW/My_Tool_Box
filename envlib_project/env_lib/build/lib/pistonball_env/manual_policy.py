"""
Manual Policy for Pistonball Environment

This module provides a manual policy for controlling the pistonball environment
using keyboard input. It allows human players to control the pistons directly.
"""

import numpy as np
import pygame


class ManualPolicy:
    """
    Manual policy for controlling pistonball environment with keyboard input.
    
    Controls:
    - W/S: Move selected piston up/down
    - A/D: Select previous/next piston
    - ESC: Exit
    - BACKSPACE: Reset environment
    """
    
    def __init__(self, env, agent_id: int = 0, show_obs: bool = False):
        """
        Initialize the manual policy.
        
        Args:
            env: The pistonball environment
            agent_id: Initial piston to control (default: 0)
            show_obs: Whether to show observations (not implemented)
        """
        self.env = env
        self.agent_id = agent_id
        self.show_obs = show_obs

        # Action mappings
        self.default_action = np.zeros(env.n_pistons)
        self.action_mapping = {
            pygame.K_w: 1.0,    # Move up
            pygame.K_s: -1.0,   # Move down
        }
        
        # Piston selection
        self.selected_piston = agent_id

    def __call__(self, observation, agent=None):
        """
        Get action based on keyboard input.
        
        Args:
            observation: Current observation (not used in manual mode)
            agent: Agent identifier (not used in manual mode)
            
        Returns:
            Action array for all pistons
        """
        # Set default action (no movement)
        action = self.default_action.copy()

        # Process keyboard events
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    # Escape to end
                    exit()
                elif event.key == pygame.K_BACKSPACE:
                    # Backspace to reset
                    self.env.reset()
                elif event.key == pygame.K_a:
                    # Select previous piston
                    self.selected_piston = (self.selected_piston - 1) % self.env.n_pistons
                    print(f"Selected piston: {self.selected_piston}")
                elif event.key == pygame.K_d:
                    # Select next piston
                    self.selected_piston = (self.selected_piston + 1) % self.env.n_pistons
                    print(f"Selected piston: {self.selected_piston}")
                elif event.key in self.action_mapping:
                    # Move selected piston
                    action[self.selected_piston] = self.action_mapping[event.key]

        return action

    @property
    def available_agents(self):
        """Return available agents."""
        return self.env.agents


def test_manual_policy():
    """Test the manual policy with the environment."""
    import gymnasium as gym
    
    # Register the environment if not already registered
    try:
        from envs.pistonball_new import PistonballEnv
    except ImportError:
        print("Please make sure the environment is properly installed")
        return
    
    # Create environment
    env = PistonballEnv(
        n_pistons=10,  # Fewer pistons for easier testing
        render_mode="human",
        continuous=True
    )
    
    # Create manual policy
    manual_policy = ManualPolicy(env)
    
    # Reset environment
    obs, info = env.reset()
    
    print("Manual Control Instructions:")
    print("- W/S: Move selected piston up/down")
    print("- A/D: Select previous/next piston")
    print("- ESC: Exit")
    print("- BACKSPACE: Reset environment")
    print(f"Starting with piston {manual_policy.selected_piston} selected")
    
    # Main loop
    clock = pygame.time.Clock()
    
    while True:
        clock.tick(env.metadata["render_fps"])
        
        # Get action from manual policy
        action = manual_policy(obs)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        # Check if episode is done
        if terminated or truncated:
            print(f"Episode finished! Reward: {reward}")
            obs, info = env.reset()
    
    env.close()


if __name__ == "__main__":
    test_manual_policy() 