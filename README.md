## Technologies/Packages Used
- **Double Deep Q-Learning (DDQN)** for reinforcement learning
- **Python**
- **Pygame**
- **PyTorch**
- **TensorFlow**

## Overview
This project implements a reinforcement learning-based car game using Double Deep Q-Learning (DDQN). 
The model learns to drive by optimizing its actions based on predicted rewards.

## Components

### Replay Buffer
The replay buffer class stores experiences of the model until it reaches a certain size before discarding old data. This helps the model learn from past experiences.

### Agent Class
- **Alpha**: The rate at which the model learns over time.
- **Gamma**: A factor to balance future and immediate rewards.
- **Epsilon**: The probability of choosing a random move (exploration).
- **Memory**: An instance of the replay buffer class.
- **Brain_eval and Brain_target**: Two neural networks used to evaluate the predicted Q-values.

### Brain Class
This class consists of the neural network structures that handle:
- **Inputs**: 
  - Velocity
  - Ray distances
  - Experience replay (state, action, reward, next state)

## Training
To train the model, run `train_model.py`. This script simulates generations of a driving model, training and updating the model based on the data received at every iteration.

## Struggles
- Not able to get the model to train well.
- The first turn might be too challenging for learning.
- The model occasionally goes backwards for some reason.
- Adjusted the weighting of the rewards and they still didn't work no matter what.
- Changed the possible moves to be simplified and it didn't help.
- Modified the sizes of the neural network.


## Sources
- https://github.com/CodeAndAction/DDQN-Car-Racing/blob/main/main.py
- https://www.youtube.com/watch?v=r428O_CMcpI
- https://www.youtube.com/watch?v=x83WmvbRa2I&ab_channel=CodeEmporium
- Claude
- ChatGPT