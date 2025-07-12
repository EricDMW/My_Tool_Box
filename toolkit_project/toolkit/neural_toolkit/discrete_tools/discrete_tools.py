"""
Discrete Reinforcement Learning Tools

This module provides tools for discrete reinforcement learning including Q-tables, value tables,
policy tables, and related utilities for tabular methods.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from abc import ABC, abstractmethod
import random


class BaseDiscreteTable:
    """Base class for discrete tables (Q-table, Value table, etc.)"""
    
    def __init__(self, 
                 state_space_size: int,
                 action_space_size: Optional[int] = None,
                 initial_value: float = 0.0,
                 dtype: Any = np.float32):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.initial_value = initial_value
        self.dtype = dtype
        
        self._initialize_table()
    
    @abstractmethod
    def _initialize_table(self):
        """Initialize the table structure"""
        pass
    
    @abstractmethod
    def get_value(self, state: int, action: Optional[int] = None) -> float:
        """Get value for state (and action)"""
        pass
    
    @abstractmethod
    def set_value(self, state: int, value: float, action: Optional[int] = None):
        """Set value for state (and action)"""
        pass
    
    @abstractmethod
    def update_value(self, state: int, value: float, action: Optional[int] = None, 
                    learning_rate: float = 0.1):
        """Update value using learning rate"""
        pass


class QTable(BaseDiscreteTable):
    """Q-Table for discrete state-action spaces"""
    
    def _initialize_table(self):
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        
        self.table = np.full((self.state_space_size, self.action_space_size), 
                           self.initial_value, dtype=self.dtype)
    
    def get_value(self, state: int, action: int) -> float:
        """Get Q-value for state-action pair"""
        return float(self.table[state, action])
    
    def set_value(self, state: int, value: float, action: int):
        """Set Q-value for state-action pair"""
        self.table[state, action] = value
    
    def update_value(self, state: int, value: float, action: int, learning_rate: float = 0.1):
        """Update Q-value using learning rate"""
        current_value = self.table[state, action]
        self.table[state, action] = current_value + learning_rate * (value - current_value)
    
    def get_max_q_value(self, state: int) -> float:
        """Get maximum Q-value for a state"""
        return float(np.max(self.table[state]))
    
    def get_max_action(self, state: int) -> int:
        """Get action with maximum Q-value for a state"""
        return int(np.argmax(self.table[state]))
    
    def get_q_values(self, state: int) -> np.ndarray:
        """Get all Q-values for a state"""
        return self.table[state].copy()
    
    def get_policy(self, state: int, epsilon: float = 0.0) -> int:
        """Get action using epsilon-greedy policy"""
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        if random.random() < epsilon:
            return random.randint(0, self.action_space_size - 1)
        else:
            return self.get_max_action(state)
    
    def get_softmax_policy(self, state: int, temperature: float = 1.0) -> int:
        """Get action using softmax policy"""
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        q_values = self.get_q_values(state)
        exp_q = np.exp(q_values / temperature)
        probs = exp_q / np.sum(exp_q)
        return int(np.random.choice(self.action_space_size, p=probs))


class ValueTable(BaseDiscreteTable):
    """Value Table for discrete state spaces"""
    
    def _initialize_table(self):
        self.table = np.full(self.state_space_size, self.initial_value, dtype=self.dtype)
    
    def get_value(self, state: int, action: Optional[int] = None) -> float:
        """Get value for state"""
        return float(self.table[state])
    
    def set_value(self, state: int, value: float, action: Optional[int] = None):
        """Set value for state"""
        self.table[state] = value
    
    def update_value(self, state: int, value: float, action: Optional[int] = None, 
                    learning_rate: float = 0.1):
        """Update value using learning rate"""
        current_value = self.table[state]
        self.table[state] = current_value + learning_rate * (value - current_value)
    
    def get_values(self) -> np.ndarray:
        """Get all values"""
        return self.table.copy()


class PolicyTable(BaseDiscreteTable):
    """Policy Table for discrete state-action spaces"""
    
    def _initialize_table(self):
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for PolicyTable")
        
        # Initialize with uniform policy
        self.table = np.full((self.state_space_size, self.action_space_size), 
                           1.0 / self.action_space_size, dtype=self.dtype)
    
    def get_value(self, state: int, action: int) -> float:
        """Get policy probability for state-action pair"""
        return float(self.table[state, action])
    
    def set_value(self, state: int, value: float, action: int):
        """Set policy probability for state-action pair"""
        self.table[state, action] = value
        # Renormalize to ensure probabilities sum to 1
        self._normalize_state(state)
    
    def update_value(self, state: int, value: float, action: int, learning_rate: float = 0.1):
        """Update policy probability using learning rate"""
        current_value = self.table[state, action]
        self.table[state, action] = current_value + learning_rate * (value - current_value)
        # Renormalize to ensure probabilities sum to 1
        self._normalize_state(state)
    
    def _normalize_state(self, state: int):
        """Normalize probabilities for a state to sum to 1"""
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for PolicyTable")
        total = np.sum(self.table[state])
        if total > 0:
            self.table[state] /= total
        else:
            # If all probabilities are 0, set uniform distribution
            self.table[state] = 1.0 / self.action_space_size
    
    def get_policy(self, state: int) -> int:
        """Sample action from policy"""
        if self.action_space_size is None:
            raise ValueError("action_space_size must be specified for PolicyTable")
        probs = self.table[state]
        return int(np.random.choice(self.action_space_size, p=probs))
    
    def get_policy_probs(self, state: int) -> np.ndarray:
        """Get policy probabilities for a state"""
        return self.table[state].copy()
    
    def set_deterministic_policy(self, state: int, action: int):
        """Set deterministic policy for a state"""
        self.table[state] = 0.0
        self.table[state, action] = 1.0


class DiscreteTools:
    """Utility class for discrete reinforcement learning operations"""
    
    @staticmethod
    def create_q_table(state_space_size: int, action_space_size: int, 
                      initial_value: float = 0.0) -> QTable:
        """Create a Q-table"""
        return QTable(state_space_size, action_space_size, initial_value)
    
    @staticmethod
    def create_value_table(state_space_size: int, initial_value: float = 0.0) -> ValueTable:
        """Create a value table"""
        return ValueTable(state_space_size, initial_value=initial_value)
    
    @staticmethod
    def create_policy_table(state_space_size: int, action_space_size: int) -> PolicyTable:
        """Create a policy table"""
        return PolicyTable(state_space_size, action_space_size)
    
    @staticmethod
    def q_learning_update(q_table: QTable, state: int, action: int, reward: float, 
                         next_state: int, gamma: float = 0.99, alpha: float = 0.1):
        """Perform Q-learning update"""
        current_q = q_table.get_value(state, action)
        max_next_q = q_table.get_max_q_value(next_state)
        target_q = reward + gamma * max_next_q
        q_table.update_value(state, target_q, action, alpha)
    
    @staticmethod
    def sarsa_update(q_table: QTable, state: int, action: int, reward: float,
                    next_state: int, next_action: int, gamma: float = 0.99, alpha: float = 0.1):
        """Perform SARSA update"""
        current_q = q_table.get_value(state, action)
        next_q = q_table.get_value(next_state, next_action)
        target_q = reward + gamma * next_q
        q_table.update_value(state, target_q, action, alpha)
    
    @staticmethod
    def expected_sarsa_update(q_table: QTable, state: int, action: int, reward: float,
                            next_state: int, policy_table: PolicyTable, 
                            gamma: float = 0.99, alpha: float = 0.1):
        """Perform Expected SARSA update"""
        current_q = q_table.get_value(state, action)
        next_q_values = q_table.get_q_values(next_state)
        next_policy_probs = policy_table.get_policy_probs(next_state)
        expected_next_q = float(np.sum(next_q_values * next_policy_probs))
        target_q = reward + gamma * expected_next_q
        q_table.update_value(state, target_q, action, alpha)
    
    @staticmethod
    def value_iteration_update(value_table: ValueTable, state: int, 
                             q_table: QTable, gamma: float = 0.99):
        """Perform value iteration update"""
        q_values = q_table.get_q_values(state)
        max_q = np.max(q_values)
        value_table.set_value(state, max_q)
    
    @staticmethod
    def policy_iteration_update(policy_table: PolicyTable, state: int, q_table: QTable):
        """Perform policy iteration update"""
        q_values = q_table.get_q_values(state)
        best_action = int(np.argmax(q_values))
        policy_table.set_deterministic_policy(state, best_action)
    
    @staticmethod
    def epsilon_greedy_policy(q_table: QTable, state: int, epsilon: float) -> int:
        """Get action using epsilon-greedy policy"""
        return q_table.get_policy(state, epsilon)
    
    @staticmethod
    def softmax_policy(q_table: QTable, state: int, temperature: float) -> int:
        """Get action using softmax policy"""
        return q_table.get_softmax_policy(state, temperature)
    
    @staticmethod
    def boltzmann_policy(q_table: QTable, state: int, temperature: float) -> int:
        """Get action using Boltzmann policy (same as softmax)"""
        return q_table.get_softmax_policy(state, temperature)
    
    @staticmethod
    def ucb_policy(q_table: QTable, state: int, visit_counts: np.ndarray, 
                   exploration_constant: float = 1.0) -> int:
        """Get action using UCB (Upper Confidence Bound) policy"""
        if q_table.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        q_values = q_table.get_q_values(state)
        state_visits = visit_counts[state]
        
        if state_visits == 0:
            # If state has never been visited, choose random action
            return random.randint(0, q_table.action_space_size - 1)
        
        # Calculate UCB values
        ucb_values = q_values + exploration_constant * np.sqrt(np.log(state_visits) / 
                                                             (visit_counts[state, :] + 1e-8))
        return int(np.argmax(ucb_values))
    
    @staticmethod
    def thompson_sampling_policy(q_table: QTable, state: int, 
                               visit_counts: np.ndarray, 
                               prior_alpha: float = 1.0, 
                               prior_beta: float = 1.0) -> int:
        """Get action using Thompson sampling policy"""
        if q_table.action_space_size is None:
            raise ValueError("action_space_size must be specified for QTable")
        q_values = q_table.get_q_values(state)
        state_visits = visit_counts[state]
        
        # Sample from Beta distribution for each action
        sampled_values = []
        for action in range(q_table.action_space_size):
            visits = visit_counts[state, action]
            # Convert Q-value to probability (assuming Q-values are in reasonable range)
            q_prob = 1.0 / (1.0 + np.exp(-q_values[action]))  # Sigmoid transformation
            alpha = prior_alpha + visits * q_prob
            beta = prior_beta + visits * (1 - q_prob)
            sampled_value = np.random.beta(alpha, beta)
            sampled_values.append(sampled_value)
        
        return int(np.argmax(sampled_values))


class DiscreteEnvironment:
    """Base class for discrete environments"""
    
    def __init__(self, state_space_size: int, action_space_size: int):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.current_state = 0
    
    @abstractmethod
    def reset(self) -> int:
        """Reset environment and return initial state"""
        pass
    
    @abstractmethod
    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Take action and return (next_state, reward, done, info)"""
        pass
    
    def get_state(self) -> int:
        """Get current state"""
        return self.current_state


# Example usage and testing
if __name__ == "__main__":
    # Create tables
    state_size = 10
    action_size = 4
    
    q_table = DiscreteTools.create_q_table(state_size, action_size)
    value_table = DiscreteTools.create_value_table(state_size)
    policy_table = DiscreteTools.create_policy_table(state_size, action_size)
    
    # Test Q-learning update
    DiscreteTools.q_learning_update(q_table, state=0, action=1, reward=1.0, 
                                   next_state=2, gamma=0.9, alpha=0.1)
    
    # Test epsilon-greedy policy
    action = DiscreteTools.epsilon_greedy_policy(q_table, state=0, epsilon=0.1)
    
    # Test softmax policy
    action = DiscreteTools.softmax_policy(q_table, state=0, temperature=1.0)
    
    print("All discrete tools created and tested successfully!") 