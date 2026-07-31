import os
import ta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Package Layout Structural Imports
from config.settings import (
    STATE_DIMENSION,
    ACTION_DIMENSION,
    TOTAL_TRAINING_EPISODES,
    TARGET_NETWORK_UPDATE_FREQ,
    DATA_TRAIN_TEST_SPLIT,
    BEST_MODEL_FILE,
    LATEST_CHECKPOINT_FILE
)
from src.agent import ProductionDDQNAgent
from src.environment import RealisticTradingEnv
from utils.checkpoints import ModelCheckpointManager


def generate_synthetic_market_data(records=500):
    """Generates synthetic stock price history populated with core technical indicators"""
    print(f"[DATA Engine] Synthesizing {records} mock market records...")
    np.random.seed(42)
    
    # Create smooth, realistic sine-wave pricing oscillations mixed with white noise
    timeline = np.linspace(0, 20, records)
    price_trend = np.sin(timeline) * 10 + 100 + np.random.normal(0, 1.2, records)
    volume_trend = np.random.randint(1000, 7500, records)
    
    df = pd.DataFrame({'close': price_trend, 'volume': volume_trend})
    
    # Inject technical indicator column arrays
    df['rsi'] = ta.momentum.rsi(df['close'], window=14).fillna(50)
    df['macd'] = ta.trend.macd(df['close']).fillna(0)
    df['sma'] = ta.trend.sma_indicator(df['close'], window=15).fillna(100)
    
    return df


def main():
    # 1. Synthesize and split technical dataset
    raw_market_data = generate_synthetic_market_data(records=500)
    split_index = int(len(raw_market_data) * DATA_TRAIN_TEST_SPLIT)
    
    train_df = raw_market_data.iloc[:split_index].copy()
    print(f"[DATA Engine] Split assigned. Training set contains {len(train_df)} periods.")

    # 2. Instantiate system classes
    env = RealisticTradingEnv(train_df)
    agent = ProductionDDQNAgent(state_dim=STATE_DIMENSION, action_dim=ACTION_DIMENSION)
    checkpoint_mgr = ModelCheckpointManager()
    
    # Load past sessions if available
    start_episode = checkpoint_mgr.load_checkpoint(agent, LATEST_CHECKPOINT_FILE)

    # 3. Initialize Interactive Plotting Dashboard Infrastructure
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle("DDQN + PER + Dueling Noisy Trading Agent - Training Profile")
    
    episodes_history, rewards_history, loss_history = [], [], []
    
    line_worth, = ax1.plot([], [], 'g-', label="Episode Net Worth ($)")
    line_loss, = ax2.plot([], [], 'r-', label="Avg Operational Loss")
    
    ax1.legend(loc="upper left"); ax2.legend(loc="upper left")
    ax1.set_ylabel("Portfolio Value ($)"); ax2.set_ylabel("Loss Score")
    ax1.grid(True); ax2.grid(True)

    best_recorded_net_worth = 0.0

    # 4. Master Core Training Loop
    print("\n" + "="*50 + f"\n STARTING REINFORCEMENT LEARNING TRAINING ENGINE\n" + "="*50)
    
    for episode in range(start_episode, TOTAL_TRAINING_EPISODES):
        state, _ = env.reset()
        episode_losses = []
        
        while True:
            # Deterministic/Noisy evaluation action selection
            action = agent.select_action(state)
            
            # Step environment forward
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Save experiences to our prioritized binary SumTree buffer matrix
            agent.memory.push(state, action, reward, next_state, done)
            
            # Optimization gradient descent execution
            step_loss = agent.update_policy()
            if step_loss > 0:
                episode_losses.append(step_loss)
                
            state = next_state
            if done:
                break
                
        # Handle Target Network weight synchronization step
        if episode % TARGET_NETWORK_UPDATE_FREQ == 0:
            agent.synchronize_target_network()

        # Compile historical stats
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        episodes_history.append(episode + 1)
        rewards_history.append(env.net_worth)
        loss_history.append(avg_loss)

        # 5. Live Dashboard Canvas Update Pipeline
        line_worth.set_data(episodes_history, rewards_history)
        line_loss.set_data(episodes_history, loss_history)
        
        # Dynamically fit rendering limits based on moving history array sizes
        ax1.set_xlim(1, max(2, episode + 1))
        ax1.set_ylim(min(rewards_history) * 0.9, max(rewards_history) * 1.1)
        
        ax2.set_xlim(1, max(2, episode + 1))
        ax2.set_ylim(0, max(loss_history) * 1.2 if any(loss_history) else 1.0)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.01)

        # 6. Persistent File System Checkpointing Save Triggers
        checkpoint_mgr.save_checkpoint(agent, episode, env.net_worth, LATEST_CHECKPOINT_FILE)
        
        if env.net_worth > best_recorded_net_worth:
            best_recorded_net_worth = env.net_worth
            checkpoint_mgr.save_checkpoint(agent, episode, env.net_worth, BEST_MODEL_FILE)

        print(f"Ep: {episode+1}/{TOTAL_TRAINING_EPISODES} | "
              f"Final Portfolio Worth: ${env.net_worth:.2f} | "
              f"Avg Loss Score: {avg_loss:.4f}")

    print("\n" + "="*50 + "\n TRAINING COMPLETE. DISCONNECTING CANVASES.\n" + "="*50)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
