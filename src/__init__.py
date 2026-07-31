"""
Deep Q-Learning Multi-Agent Reinforcement Learning Core Package Initializer.
Exposes algorithmic architectures, simulation wrappers, and allocation supervisors.
"""

from src.models import NoisyLinear, DuelingNoisyDQN
from src.buffer import SumTree, PrioritizedReplayBuffer
from src.environment import RealisticTradingEnv
from src.agent import ProductionDDQNAgent
from src.risk_manager import RiskManagementOverlay
from src.tracker import LiveAssetFeatureTracker
from src.allocator import MasterAllocationCritic

__all__ = [
    "NoisyLinear",
    "DuelingNoisyDQN",
    "SumTree",
    "PrioritizedReplayBuffer",
    "RealisticTradingEnv",
    "ProductionDDQNAgent",
    "RiskManagementOverlay",
    "LiveAssetFeatureTracker",
    "MasterAllocationCritic"
]
