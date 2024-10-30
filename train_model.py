import os
from car_game_ai import CarGameAI
import pygame
import numpy as np
import matplotlib.pyplot as plt
from agent import Agent

# Constants
TOTAL_GAMETIME = 100000          # Total runtime for the game
N_EPISODES = 10000               # Total number of training episodes
REPLACE_TARGET = 10              # Frequency (in episodes) to update target network

# Game window dimensions
WIDTH, HEIGHT = 1000, 800
game = CarGameAI(WIDTH // 2, HEIGHT // 2 - 200)

# Game state and history variables
GameTime = 0
GameHistory = []

# Initialize the agent with parameters
agent = Agent(
    alpha=0.0005,                  # Learning rate
    gamma=0.99,                    # Discount factor for future rewards
    n_actions=4,                   # Number of actions the agent can take
    epsilon=1.00,                  # Initial exploration rate (start % random moves)
    epsilon_end=0.1,               # Minimum exploration rate (min % random moves)
    epsilon_dec=0.99995,           # Decay rate of epsilon
    replace_target=REPLACE_TARGET, # Frequency of replacing target network
    batch_size=256,                # Training batch size
    input_dims=10                  # Number of input dimensions (observations/data)
)

# Load or initialize model weights
if os.path.exists(agent.model_file):
    print("Loading model...")
    agent.load_model()
else:
    print("No model found, training from scratch.")
agent.update_network_parameters()

# Score tracking
ddqn_scores = []
eps_history = []

# Function to plot scores per episode
def plot_scores(scores):
    plt.close('all')
    plt.plot(scores)
    plt.title('Scores per Round')
    plt.xlabel('Round')
    plt.ylabel('Score')
    plt.show()

# Main training loop
def run():
    for e in range(N_EPISODES):
        # Reset environment and initial variables for each episode
        game.reset()
        game.checkpoint_passed = []
        game.score = 0
        game.rewards = 0

        counter = 0
        gtime = 0  # Local time counter per episode

        # Initial observation and reward
        observation_, reward, checkpoints, done = game.step([0, 0, 0, 0])
        observation = np.array(observation_)

        # Episode loop
        while not done:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # Choose action based on observations
            action = agent.choose_action(observation)
            observation_, reward, checkpoints, done = game.step(action)
            observation_ = np.array(observation_)

            # Train every 5 frames or when the car has crashed
            if gtime % 5 == 0 or done:
                agent.remember(observation, action, reward, observation_, int(done))
                agent.learn()

            # Update current observation
            observation = observation_

            # End episode if car is stuck (not gaining rewards)
            if reward <= 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            gtime += 1

        # Append episode results to history lists
        eps_history.append(agent.epsilon)
        ddqn_scores.append(game.rewards)

        # Plot scores at the end of each episode
        plot_scores(ddqn_scores)

        # Average score of last 100 games
        avg_score = np.mean(ddqn_scores[max(0, e - 100):(e + 1)])

        # Update target network at every few episodes
        if e % REPLACE_TARGET == 0 and e > REPLACE_TARGET:
            agent.update_network_parameters()

        # Save model every 10 episodes
        if e % 10 == 0 and e > 10:
            agent.save_model()
            print("Model saved.")

        # Print progress every episode
        print(
            f'Episode: {e}, Score: {game.rewards:.2f}, Average Score: {avg_score:.2f}, '
            f'Epsilon: {agent.epsilon}, Memory Size: {agent.memory.mem_cntr % agent.memory.mem_size}'
        )

# Run the main training loop
run()