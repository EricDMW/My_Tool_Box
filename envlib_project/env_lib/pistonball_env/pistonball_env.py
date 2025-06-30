"""
Pistonball Environment - A standard gym environment for multi-agent pistonball game.

This environment simulates a physics-based cooperative game where multiple pistons
work together to move a ball to the left wall. Each piston can move up or down
to create a path for the ball.

Parameters:
- n_pistons: Number of piston agents (default: 20)
- time_penalty: Reward penalty per time step (default: -0.1)
- continuous: Whether to use continuous actions (default: True)
- random_drop: Whether to randomly drop the ball (default: True)
- random_rotate: Whether to randomly rotate the ball (default: True)
- ball_mass: Mass of the ball (default: 0.75)
- ball_friction: Friction coefficient of the ball (default: 0.3)
- ball_elasticity: Elasticity of the ball (default: 1.5)
- max_cycles: Maximum number of steps per episode (default: 125)
- render_mode: Rendering mode ('human', 'rgb_array', None)
- movement_penalty: Penalty for piston movement (default: 0.0)
- movement_penalty_threshold: Minimum movement to trigger penalty (default: 0.01)
- kappa: Number of hops for observation range (default: 1)
"""

import math
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from gymnasium import Env
from gymnasium.spaces import Box, Discrete, Dict
from gymnasium.utils import EzPickle, seeding
import os

FPS = 20


def get_image(path):
    """Load and return a pygame surface from image path."""
    cwd = os.path.dirname(__file__)
    image = pygame.image.load(os.path.join(cwd, path))
    sfc = pygame.Surface(image.get_size(), flags=pygame.SRCALPHA)
    sfc.blit(image, (0, 0))
    return sfc


def validate_ball_position(x, y, screen_width, screen_height, ball_radius):
    """
    Validate ball position to ensure it stays within bounds.
    
    Args:
        x, y: Ball coordinates
        screen_width, screen_height: Screen dimensions
        ball_radius: Ball radius
        
    Returns:
        Validated coordinates
    """
    # Ensure ball stays within screen bounds
    x = np.clip(x, ball_radius, screen_width - ball_radius)
    y = np.clip(y, ball_radius, screen_height - ball_radius)
    
    return x, y


def safe_physics_step(space, dt, max_velocity=1000.0):
    """
    Safe physics step with velocity limiting to prevent instability.
    
    Args:
        space: Pymunk space
        dt: Time step
        max_velocity: Maximum allowed velocity
        
    Returns:
        Number of steps taken
    """
    try:
        # Limit velocities of all bodies to prevent instability
        for body in space.bodies:
            if hasattr(body, 'velocity') and body.velocity.length > max_velocity:
                body.velocity = body.velocity.normalized() * max_velocity
        
        # Step physics
        space.step(dt)
        return 1
        
    except Exception as e:
        print(f"Warning: Physics step failed: {e}")
        return 0


class PistonballEnv(Env, EzPickle):
    """
    Pistonball Environment - Standard gym environment for multi-agent pistonball game.
    
    This environment simulates a physics-based cooperative game where multiple pistons
    work together to move a ball to the left wall. Each piston can move up or down
    to create a path for the ball.
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "Pistonball-v0",
        "render_fps": FPS,
    }

    def __init__(
        self,
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
        render_mode=None,
        movement_penalty=0.0,
        movement_penalty_threshold=0.01,
        kappa=1,
    ):
        """
        Initialize the Pistonball environment.
        
        Args:
            n_pistons: Number of piston agents
            time_penalty: Reward penalty per time step
            continuous: Whether to use continuous actions
            random_drop: Whether to randomly drop the ball
            random_rotate: Whether to randomly rotate the ball
            ball_mass: Mass of the ball
            ball_friction: Friction coefficient of the ball
            ball_elasticity: Elasticity of the ball
            max_cycles: Maximum number of steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', None)
            movement_penalty: Penalty for piston movement (negative value)
            movement_penalty_threshold: Minimum movement to trigger penalty
            kappa: Number of hops for observation range (default: 1)
        """
        EzPickle.__init__(
            self,
            n_pistons,
            time_penalty,
            continuous,
            random_drop,
            random_rotate,
            ball_mass,
            ball_friction,
            ball_elasticity,
            max_cycles,
            render_mode,
            movement_penalty,
            movement_penalty_threshold,
            kappa,
        )
        
        # Environment parameters
        self.n_pistons = n_pistons
        self.time_penalty = time_penalty
        self.continuous = continuous
        self.random_drop = random_drop
        self.random_rotate = random_rotate
        self.ball_mass = ball_mass
        self.ball_friction = ball_friction
        self.ball_elasticity = ball_elasticity
        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.movement_penalty = movement_penalty
        self.movement_penalty_threshold = movement_penalty_threshold
        self.kappa = kappa
        
        # Physics parameters
        self.dt = 1.0 / FPS
        self.piston_head_height = 11
        self.piston_width = 40
        self.piston_height = 40
        self.piston_body_height = 23
        self.piston_radius = 5
        self.wall_width = 40 * 2
        self.ball_radius = 40
        self.screen_width = (2 * self.wall_width) + (self.piston_width * self.n_pistons)
        self.screen_height = 560
        self.maximum_piston_y = (
            self.screen_height
            - self.wall_width
            - (self.piston_height - self.piston_head_height)
        )
        
        # Piston movement parameters
        self.pixels_per_position = 4
        self.n_piston_positions = 16
        self.piston_y_half_range = 0.5 * self.pixels_per_position * self.n_piston_positions
        self.mid_piston_y = self.maximum_piston_y - self.piston_y_half_range
        
        # Agent setup
        self.agents = [f"piston_{i}" for i in range(self.n_pistons)]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.n_pistons))))
        
        # Action and observation spaces
        if self.continuous:
            # Action space: sequence of actions for all pistons
            # Each action value controls one piston (-1: down, 0: stay, 1: up)
            self.action_space = Box(low=-1, high=1, shape=(self.n_pistons,), dtype=np.float32)
        else:
            # Discrete mode: sequence of discrete actions for each piston
            # Each piston has 3 actions: 0 (down), 1 (stay), 2 (up)
            self.action_space = Box(low=0, high=2, shape=(self.n_pistons,), dtype=np.int32)
        
        # Observation space: each piston gets its position and ball information
        obs_dim = 7  # piston_pos_y + piston_pos_x + ball_x + ball_y + ball_vx + ball_vy + ball_angular_vel
        self.observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_pistons, obs_dim), 
            dtype=np.float32
        )
        
        # State space for rendering
        self.state_space = Box(
            low=0,
            high=255,
            shape=(self.screen_height, self.screen_width, 3),
            dtype=np.uint8,
        )
        
        # Initialize pygame and pymunk
        pygame.init()
        pymunk.pygame_util.positive_y_is_up = False
        
        # Rendering setup
        self.renderOn = False
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        
        # Load sprites
        self.piston_sprite = get_image("piston.png")
        self.piston_body_sprite = get_image("piston_body.png")
        self.background = get_image("background.png")
        
        # Game state
        self.pistonList = []
        self.piston_pos_y = np.full(self.n_pistons, self.mid_piston_y)
        self.ball = None
        self.space = None
        self.lastX = 0
        self.frames = 0
        self.terminate = False
        self.truncate = False
        
        # Rendering rectangles
        self.render_rect = pygame.Rect(
            self.wall_width,
            self.wall_width,
            self.screen_width - (2 * self.wall_width),
            self.screen_height - (2 * self.wall_width) - self.piston_body_height,
        )
        
        self.valid_ball_position_rect = pygame.Rect(
            self.render_rect.left + self.ball_radius,
            self.render_rect.top + self.ball_radius,
            self.render_rect.width - (2 * self.ball_radius),
            self.render_rect.height - (2 * self.ball_radius),
        )
        
        self.closed = False
        self.seed()
        
        # State tracking for local rewards
        self.last_ball_x = None
        self.last_ball_positions = None  # Will store ball positions for each piston

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_walls(self):
        """Add walls to the physics space."""
        if self.space is None:
            return
            
        top_left = (self.wall_width, self.wall_width)
        top_right = (self.screen_width - self.wall_width, self.wall_width)
        bot_left = (self.wall_width, self.screen_height - self.wall_width)
        bot_right = (
            self.screen_width - self.wall_width,
            self.screen_height - self.wall_width,
        )
        walls = [
            pymunk.Segment(self.space.static_body, top_left, top_right, 1),
            pymunk.Segment(self.space.static_body, top_left, bot_left, 1),
            pymunk.Segment(self.space.static_body, bot_left, bot_right, 1),
            pymunk.Segment(self.space.static_body, top_right, bot_right, 1),
        ]
        for wall in walls:
            wall.friction = 0.64
            self.space.add(wall)

    def add_ball(self, x, y):
        """Add a ball to the physics space."""
        # Validate ball position
        x, y = validate_ball_position(x, y, self.screen_width, self.screen_height, self.ball_radius)
        
        mass = self.ball_mass
        radius = self.ball_radius
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        
        if self.random_rotate:
            body.angular_velocity = self.np_random.uniform(-6 * math.pi, 6 * math.pi)
        
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.friction = self.ball_friction
        shape.elasticity = self.ball_elasticity
        self.space.add(body, shape)
        return body

    def add_piston(self, space, x, y):
        """Add piston to the physics space."""
        piston = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        piston.position = x, y
        segment = pymunk.Segment(
            piston,
            (0, 0),
            (self.piston_width - (2 * self.piston_radius), 0),
            self.piston_radius,
        )
        segment.friction = 0.64
        segment.color = pygame.color.THECOLORS["blue"]
        space.add(piston, segment)
        return piston

    def move_piston(self, piston, v):
        """Move piston by velocity v."""
        def cap(y):
            maximum_piston_y = (
                self.screen_height
                - self.wall_width
                - (self.piston_height - self.piston_head_height)
            )
            if y > maximum_piston_y:
                y = maximum_piston_y
            elif y < maximum_piston_y - (
                self.n_piston_positions * self.pixels_per_position
            ):
                y = maximum_piston_y - (
                    self.n_piston_positions * self.pixels_per_position
                )
            return y

        piston.position = (
            piston.position[0],
            cap(piston.position[1] - v * self.pixels_per_position),
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        if seed is not None:
            self.seed(seed)
            
        # Initialize physics space
        self.space = pymunk.Space(threaded=False)
        self.add_walls()
        self.space.gravity = (0.0, 750.0)
        self.space.collision_bias = 0.0001
        self.space.iterations = 10
        
        # Reset pistons
        self.pistonList = []
        self.piston_pos_y = np.full(self.n_pistons, self.mid_piston_y)
        
        for i in range(self.n_pistons):
            possible_y_displacements = np.arange(
                0,
                0.5 * self.pixels_per_position * self.n_piston_positions,
                self.pixels_per_position,
            )
            piston = self.add_piston(
                self.space,
                self.wall_width + self.piston_radius + self.piston_width * i,
                self.maximum_piston_y - self.np_random.choice(possible_y_displacements),
            )
            piston.velocity = (0, 0)
            self.pistonList.append(piston)
            self.piston_pos_y[i] = piston.position[1]

        # Add ball
        horizontal_offset_range = 30
        vertical_offset_range = 15
        horizontal_offset = 0
        vertical_offset = 0
        
        if self.random_drop:
            vertical_offset = self.np_random.integers(
                -vertical_offset_range, vertical_offset_range + 1
            )
            horizontal_offset = self.np_random.integers(
                -horizontal_offset_range, horizontal_offset_range + 1
            )
            
        ball_x = (
            self.screen_width
            - self.wall_width
            - self.ball_radius
            - horizontal_offset_range
            + horizontal_offset
        )
        ball_y = (
            self.screen_height
            - self.wall_width
            - self.piston_body_height
            - self.ball_radius
            - (0.5 * self.pixels_per_position * self.n_piston_positions)
            - vertical_offset_range
            + vertical_offset
        )

        ball_x = max(ball_x, self.wall_width + self.ball_radius + 1)
        self.ball = self.add_ball(ball_x, ball_y)
        
        if self.ball is not None:
            self.ball.angle = 0
            self.ball.velocity = (0, 0)
            
            if self.random_rotate:
                self.ball.angular_velocity = self.np_random.uniform(
                    -6 * math.pi, 6 * math.pi
                )

            self.lastX = int(self.ball.position[0] - self.ball_radius)
        
        self.frames = 0
        self.terminate = False
        self.truncate = False
        
        # Initialize ball position tracking for local rewards
        if self.ball is not None:
            self.last_ball_x = int(self.ball.position[0] - self.ball_radius)
            # Initialize last ball positions for each piston
            self.last_ball_positions = np.full(self.n_pistons, self.last_ball_x)
        else:
            self.last_ball_x = None
            self.last_ball_positions = None
        
        # Draw initial state
        self.draw_background()
        self.draw()
        
        return self._get_obs(), {}

    def _can_observe_ball(self, piston_index, ball_x):
        """
        Determine if a piston can observe the ball based on kappa-hop neighbor system.
        
        Args:
            piston_index: Index of the piston (0 to n_pistons-1)
            ball_x: X position of the ball
            
        Returns:
            bool: True if the piston can observe the ball, False otherwise
        """
        if self.ball is None:
            return False
            
        # Calculate piston's x position
        piston_x = self.wall_width + self.piston_radius + self.piston_width * piston_index
        
        # Calculate the range of pistons that can observe the ball
        # A piston can observe the ball if it's within kappa hops of the piston closest to the ball
        ball_piston_index = int((ball_x - self.wall_width - self.piston_radius) / self.piston_width)
        ball_piston_index = max(0, min(ball_piston_index, self.n_pistons - 1))
        
        # Check if current piston is within kappa hops of the ball's piston
        distance = abs(piston_index - ball_piston_index)
        return distance <= self.kappa

    def _get_obs(self):
        """Get observations for all agents with limited observation range."""
        obs = np.zeros((self.n_pistons, 7))
        
        if self.ball is None:
            return obs
            
        for i in range(self.n_pistons):
            # Piston position (normalized) - always observable
            piston_pos_y = (self.piston_pos_y[i] - self.mid_piston_y) / self.piston_y_half_range
            piston_pos_x = (self.wall_width + self.piston_radius + self.piston_width * i - self.wall_width) / (self.screen_width - 2 * self.wall_width)
            obs[i, 0] = piston_pos_y
            obs[i, 1] = piston_pos_x
            
            # Check if this piston can observe the ball
            if self._can_observe_ball(i, self.ball.position[0]):
                # Ball information - only if within observation range
                obs[i, 2] = (self.ball.position[0] - self.wall_width) / (self.screen_width - 2 * self.wall_width)  # normalized x
                obs[i, 3] = (self.ball.position[1] - self.wall_width) / (self.screen_height - 2 * self.wall_width)  # normalized y
                obs[i, 4] = self.ball.velocity[0] / 15  # normalized velocity x
                obs[i, 5] = self.ball.velocity[1] / 8   # normalized velocity y
                obs[i, 6] = self.ball.angular_velocity / 8  # normalized angular velocity
            else:
                # Set ball parameters to zero if cannot observe
                obs[i, 2:7] = 0.0
            
        return obs

    def step(self, action):
        """Take a step in the environment."""
        if self.space is None or self.ball is None:
            return self._get_obs(), 0, True, True, {}
            
        # Store previous piston positions for movement penalty calculation
        prev_piston_positions = np.array(self.piston_pos_y.copy())
        
        # Handle discrete actions
        if not self.continuous:
            # Convert discrete actions to continuous actions for each piston
            action_array = np.zeros(self.n_pistons)
            for i in range(self.n_pistons):
                # Convert discrete action (0, 1, 2) to continuous action (-1, 0, 1)
                action_array[i] = action[i] - 1
        else:
            # Ensure action is a numpy array with correct shape
            action_array = np.array(action, dtype=np.float32)
            if action_array.shape != (self.n_pistons,):
                raise ValueError(f"Action shape {action_array.shape} does not match expected shape ({self.n_pistons},)")
            
        # Apply actions to all pistons simultaneously
        for i, action_val in enumerate(action_array):
            # Clip action to valid range
            action_val = np.clip(action_val, -1, 1)
            self.move_piston(self.pistonList[i], action_val)
            self.piston_pos_y[i] = self.pistonList[i].position[1]

        # Step physics safely
        safe_physics_step(self.space, self.dt)
        
        # Validate ball position after physics step
        if self.ball is not None:
            x, y = validate_ball_position(
                self.ball.position[0], 
                self.ball.position[1], 
                self.screen_width, 
                self.screen_height, 
                self.ball_radius
            )
            self.ball.position = x, y
        
        # Check termination conditions
        ball_min_x = int(self.ball.position[0] - self.ball_radius)
        ball_next_x = (
            self.ball.position[0]
            - self.ball_radius
            + self.ball.velocity[0] * self.dt
        )
        
        if ball_next_x <= self.wall_width + 1:
            self.terminate = True
            
        # Calculate local rewards for each piston
        local_rewards = np.zeros(self.n_pistons)
        
        if self.ball is not None and self.last_ball_positions is not None:
            for i in range(self.n_pistons):
                # Get previous ball position for this piston
                prev_ball_x = self.last_ball_positions[i]
                curr_ball_x = ball_min_x
                
                # Calculate local reward for this piston
                local_rewards[i] = self.get_local_reward_for_piston(i, prev_ball_x, curr_ball_x)
                
                # Update last ball position for this piston
                self.last_ball_positions[i] = curr_ball_x
        else:
            # If no ball or no previous positions, all pistons get time penalty only
            local_rewards.fill(self.time_penalty)
            
        # Calculate movement penalties for each piston
        movement_penalties = np.zeros(self.n_pistons)
        if self.movement_penalty != 0.0:
            current_piston_positions = np.array(self.piston_pos_y)
            piston_movements = np.abs(current_piston_positions - prev_piston_positions)
            
            # Apply penalty only if movement exceeds threshold
            for i, movement in enumerate(piston_movements):
                if movement > self.movement_penalty_threshold:
                    movement_penalties[i] = self.movement_penalty * (movement / self.pixels_per_position)
        
        # Add movement penalties to local rewards
        local_rewards += movement_penalties
        
        # Calculate total reward (sum of all local rewards)
        total_reward = np.sum(local_rewards)
        
        # Update state
        self.last_ball_x = ball_min_x
        self.frames += 1
        self.truncate = self.frames >= self.max_cycles
        
        # Draw
        self.draw()
        
        # Return observations, local rewards list, total reward, termination flags, and info
        return self._get_obs(), local_rewards, total_reward, self.terminate, self.truncate, {}

    def get_local_reward_for_piston(self, piston_index, prev_ball_x, curr_ball_x):
        """
        Calculate local reward for a specific piston based on its observation.
        
        Args:
            piston_index: Index of the piston
            prev_ball_x: Previous ball x position
            curr_ball_x: Current ball x position
            
        Returns:
            float: Local reward for this piston
        """
        # Check if this piston can observe the ball
        can_observe_prev = self._can_observe_ball(piston_index, prev_ball_x)
        can_observe_curr = self._can_observe_ball(piston_index, curr_ball_x)
        
        # If piston cannot observe ball in either state, return only time penalty
        if not can_observe_prev and not can_observe_curr:
            return self.time_penalty
        
        # If piston can observe ball, calculate ball-related reward
        if prev_ball_x > curr_ball_x:
            ball_reward = 0.5 * (prev_ball_x - curr_ball_x)  # Moving left (good)
        else:
            ball_reward = prev_ball_x - curr_ball_x  # Moving right (bad)
        
        return ball_reward + self.time_penalty

    def get_local_reward(self, prev_position, curr_position):
        """Calculate local reward based on ball movement (legacy method)."""
        if prev_position > curr_position:
            local_reward = 0.5 * (prev_position - curr_position)
        else:
            local_reward = prev_position - curr_position
        return local_reward

    def draw_background(self):
        """Draw the background."""
        outer_walls = pygame.Rect(0, 0, self.screen_width, self.screen_height)
        outer_wall_color = (58, 64, 65)
        pygame.draw.rect(self.screen, outer_wall_color, outer_walls)
        
        inner_walls = pygame.Rect(
            self.wall_width / 2,
            self.wall_width / 2,
            self.screen_width - self.wall_width,
            self.screen_height - self.wall_width,
        )
        inner_wall_color = (68, 76, 77)
        pygame.draw.rect(self.screen, inner_wall_color, inner_walls)
        self.draw_pistons()

    def draw_pistons(self, observable_pistons=None):
        """Draw all pistons. Observable pistons are red, others are blue."""
        piston_color = (65, 159, 221)  # blue
        observable_color = (220, 50, 50)  # red
        x_pos = self.wall_width
        if observable_pistons is None:
            observable_pistons = []
        for idx, piston in enumerate(self.pistonList):
            self.screen.blit(
                self.piston_body_sprite,
                (x_pos, self.screen_height - self.wall_width - self.piston_body_height),
            )
            height = (
                self.screen_height
                - self.wall_width
                - self.piston_body_height
                - (piston.position[1] + self.piston_radius)
                + (self.piston_body_height - 6)
            )
            body_rect = pygame.Rect(
                piston.position[0] + self.piston_radius + 1,
                piston.position[1] + self.piston_radius + 1,
                18,
                height,
            )
            color = observable_color if idx in observable_pistons else piston_color
            pygame.draw.rect(self.screen, color, body_rect)
            x_pos += self.piston_width

    def draw(self):
        """Draw the current state."""
        if self.ball is None:
            return
        if not self.valid_ball_position_rect.collidepoint(self.ball.position):
            self.draw_background()
        ball_x = int(self.ball.position[0])
        ball_y = int(self.ball.position[1])
        # Draw render area
        color = (255, 255, 255)
        pygame.draw.rect(self.screen, color, self.render_rect)
        # Draw ball
        color = (65, 159, 221)
        pygame.draw.circle(self.screen, color, (ball_x, ball_y), self.ball_radius)
        # Draw ball rotation indicator
        line_end_x = ball_x + (self.ball_radius - 1) * np.cos(self.ball.angle)
        line_end_y = ball_y + (self.ball_radius - 1) * np.sin(self.ball.angle)
        color = (58, 64, 65)
        pygame.draw.line(
            self.screen, color, (ball_x, ball_y), (line_end_x, line_end_y), 3
        )
        # Compute observable pistons
        observable_pistons = []
        if self.ball is not None:
            for i in range(self.n_pistons):
                if self._can_observe_ball(i, self.ball.position[0]):
                    observable_pistons.append(i)
        # Draw pistons
        for piston in self.pistonList:
            self.screen.blit(
                self.piston_sprite,
                (
                    piston.position[0] - self.piston_radius,
                    piston.position[1] - self.piston_radius,
                ),
            )
        self.draw_pistons(observable_pistons=observable_pistons)

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        if self.render_mode == "human" and not self.renderOn:
            self.enable_render()

        self.draw_background()
        self.draw()

        observation = np.array(pygame.surfarray.pixels3d(self.screen))
        if self.render_mode == "human":
            pygame.display.flip()
        return (
            np.transpose(observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )

    def enable_render(self):
        """Enable rendering mode."""
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.renderOn = True
        self.draw_background()
        self.draw()

    def close(self):
        """Close the environment."""
        if not self.closed:
            self.closed = True
            if self.renderOn:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
                self.renderOn = False
                pygame.event.pump()
                pygame.display.quit() 