import os
import torch

from config.settings import (
    CHECKPOINT_DIR,
    LATEST_CHECKPOINT_FILE,
    BEST_MODEL_FILE
)


class ModelCheckpointManager:
    """Manages saving and loading comprehensive training states for safety and deployment"""
    def __init__(self, directory=CHECKPOINT_DIR):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def save_checkpoint(self, agent, episode, reward, filename=LATEST_CHECKPOINT_FILE):
        """Saves weights, parameters, and optimizer states to disk"""
        filepath = os.path.join(self.directory, filename)
        
        checkpoint_dict = {
            'episode': episode,
            'policy_net_state': agent.policy_net.state_dict(),
            'target_net_state': agent.target_net.state_dict(),
            'optimizer_state': agent.optimizer.state_dict(),
            'buffer_entries': agent.memory.tree.n_entries,
            'metrics': {'last_reward': reward}
        }
        
        torch.save(checkpoint_dict, filepath)
        
        # Save a dedicated backup if it breaks a net worth record
        if filename == BEST_MODEL_FILE:
            print(f"[CHECKPOINT] Saved new optimal performance model parameter snapshot: {filepath}")

    def load_checkpoint(self, agent, filename=LATEST_CHECKPOINT_FILE):
        """
        Loads a saved checkpoint file.
        Returns the saved episode count to resume training seamlessly.
        """
        filepath = os.path.join(self.directory, filename)
        if not os.path.exists(filepath):
            print(f"[CHECKPOINT] No previous checkpoint found at {filepath}. Starting training from scratch.")
            return 0
            
        print(f"[CHECKPOINT] Loading weights from checkpoint: {filepath}...")
        checkpoint = torch.load(filepath, map_location=agent.device)
        
        # Restore state variables
        agent.policy_net.load_state_dict(checkpoint['policy_net_state'])
        agent.target_net.load_state_dict(checkpoint['target_net_state'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        print(f"[CHECKPOINT] Success! Resuming session from training Episode: {checkpoint['episode'] + 1}")
        return checkpoint['episode']
