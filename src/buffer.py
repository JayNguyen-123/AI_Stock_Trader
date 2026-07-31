import random
import numpy as np

from config.settings import (
    REPLAY_BUFFER_CAPACITY,
    PER_ALPHA,
    PER_BETA_START,
    PER_BETA_GROWTH
)


class SumTree:
    """
    A flat-array binary tree structure where each parent node 
    stores the mathematical sum of its children's priority scores.
    """
    def __init__(self, capacity=REPLAY_BUFFER_CAPACITY):
        self.capacity = capacity
        # A tree with N leaves has exactly 2N - 1 total structural nodes
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Recursively update parent node evaluations up to the root element"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """Updates a node's priority value and propagates the delta through the tree"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """Inserts a new experience entry at the current leaf position"""
        idx = self.write_pointer + self.capacity - 1
        self.data[self.write_pointer] = data
        self.update(idx, priority)
        
        # Circular array wrapping to maintain fixed buffer constraints
        self.write_pointer = (self.write_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def get_leaf(self, value):
        """Traverses down the binary tree to retrieve the leaf node matching the query value"""
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            
            # Terminal condition check: Leaf boundary reached
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx
        
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        """Returns the complete summation score sitting at the root of the tree"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """
    Experience memory storage that prioritizes transitions based on historical TD errors 
    and uses Importance Sampling (IS) weights to correct for sampling bias.
    """
    def __init__(self):
        self.tree = SumTree(REPLAY_BUFFER_CAPACITY)
        self.alpha = PER_ALPHA
        self.beta = PER_BETA_START
        self.beta_increment = PER_BETA_GROWTH
        self.epsilon = 1e-5  # Minimal non-zero fallback boundary

    def push(self, state, action, reward, next_state, done):
        """Saves a transition, initializing its priority to the highest current error value"""
        # Slice out the trailing edge leaf boundaries containing real weights
        leaf_priorities = self.tree.tree[-self.tree.capacity:]
        max_priority = np.max(leaf_priorities)
        
        if max_priority == 0.0:
            max_priority = 1.0  # Initial default fallback for an empty tree
            
        transition = (state, action, reward, next_state, done)
        self.tree.add(max_priority, transition)

    def sample(self, batch_size):
        """Samples a stratified batch of transitions based on priority values"""
        batch = []
        idxs = []
        priorities = []
        
        # Partition the total priority space into uniform segments
        segment = self.tree.total_priority / batch_size
        
        # Incrementally adjust the beta coefficient to smoothly eliminate sampling bias over time
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(batch_size):
            start = segment * i
            end = segment * (i + 1)
            query_val = random.uniform(start, end)
            
            idx, priority, data = self.tree.get_leaf(query_val)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # Calculate tracking probabilities: P(i) = p_i^alpha / sum(p_k^alpha)
        sampling_probabilities = np.array(priorities) / (self.tree.total_priority + 1e-9)
        
        # Calculate Importance Sampling weights: w_i = (N * P(i))^-beta
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= (is_weights.max() + 1e-9)  # Max normalization for variance stability

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states), 
            np.array(actions), 
            np.array(rewards, dtype=np.float32), 
            np.array(next_states), 
            np.array(dones, dtype=np.float32), 
            idxs, 
            np.array(is_weights, dtype=np.float32)
        )

    def update_priorities(self, idxs, errors):
        """Re-calibrates sample priorities inside the tree using absolute TD errors"""
        for idx, error in zip(idxs, errors):
            # Clip values to handle unexpected outliers
            clipped_error = np.minimum(np.abs(error) + self.epsilon, 1.0)
            priority = np.power(clipped_error, self.alpha)
            self.tree.update(idx, priority)
