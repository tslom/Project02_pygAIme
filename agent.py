from keras.api.layers import Dense, Activation
from keras.api.models import Sequential, load_model
from keras.api.optimizers import Adam
import numpy as np
import tensorflow as tf

class ReplayBuffer:
    """
    ReplayBuffer class manages the memory buffer for storing and sampling past experiences.
    """

    def __init__(self, max_size, input_shape, n_actions, discrete=False):
        """
        Initialize the replay buffer.

        Parameters:
        - max_size (int): Maximum number of experiences to store.
        - input_shape (int): Shape of the input state.
        - n_actions (int): Number of possible actions.
        - discrete (bool): If True, actions are stored in discrete format (one-hot encoding).
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a single transition in memory.

        Parameters:
        - state: Current state.
        - action: Action taken in the current state.
        - reward: Reward received after taking the action.
        - state_: Next state after taking the action.
        - done: Whether the episode ended after this transition.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch of experiences from memory.

        Parameters:
        - batch_size (int): Number of samples to return.

        Returns:
        A tuple of (states, actions, rewards, next_states, terminal flags).
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]
        return states, actions, rewards, states_, terminal


class Agent:
    """
    Agent class for the DDQN agent, handling action selection, learning, and model updates.
    """

    def __init__(self, alpha, gamma, n_actions, epsilon, batch_size,
                 input_dims, epsilon_dec=0.999995, epsilon_end=0.25,
                 mem_size=25000, fname='ddqn_model.h5', replace_target=25):
        """
        Initialize the agent with parameters.

        Parameters:
        - alpha (float): Learning rate.
        - gamma (float): Discount factor balancing immediate and future rewards.
        - n_actions (int): Number of possible outputs (actions).
        - epsilon (float): Initial exploration rate (% of random outputs).
        - batch_size (int): Batch size for training.
        - input_dims (int): Input dimension of the state (how many inputs).
        - epsilon_dec (float): Decay rate for epsilon.
        - epsilon_end (float): Minimum epsilon.
        - mem_size (int): Size of the replay buffer.
        - fname (str): Filename for saving/loading the model.
        - replace_target (int): Steps to update target network.
        """
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.replace_target = replace_target
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.brain_eval = Brain(input_dims, n_actions, batch_size)
        self.brain_target = Brain(input_dims, n_actions, batch_size)

    def remember(self, state, action, reward, new_state, done):
        """
        Store a memory of a state transition.
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, state):
        """
        Choose an action based on epsilon.

        Parameters:
        - state: Current state.

        Returns:
        An array of 5 binary values representing actions.
        """
        state = np.array(state)[np.newaxis, :]
        if np.random.random() < self.epsilon:
            action = np.random.choice([0, 1], size=5)
        else:
            actions = self.brain_eval.predict(state)
            action = np.where(actions[0] > 0.7, 1, 0)
        return action

    def learn(self):
        """
        Train the agent using a batch of experiences from the replay buffer.
        """
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)
            q_next = self.brain_target.predict(new_state)
            q_eval = self.brain_eval.predict(new_state)
            q_pred = self.brain_eval.predict(state)
            max_actions = np.argmax(q_eval, axis=1)
            q_target = q_pred
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward + self.gamma * q_next[
                batch_index, max_actions.astype(int)] * done
            _ = self.brain_eval.train(state, q_target)
            self.epsilon = max(self.epsilon * self.epsilon_dec, self.epsilon_min)

    def update_network_parameters(self):
        """
        Copy weights from the evaluation network to the target network.
        """
        self.brain_target.copy_weights(self.brain_eval)

    def save_model(self):
        """
        Save the evaluation model to a file.
        """
        self.brain_eval.model.save(self.model_file)

    def load_model(self):
        """
        Load the evaluation model from a file and sync with the target network.
        """
        self.brain_eval.model = load_model(self.model_file)
        self.brain_target.model = load_model(self.model_file)
        self.brain_eval.model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0005))
        self.brain_target.model.compile(loss="mse", optimizer=Adam(learning_rate=0.0005))
        if self.epsilon == 0.0:
            self.update_network_parameters()


class Brain:
    """
    Brain class defines the neural network structure and training methods.
    """

    def __init__(self, NbrStates, NbrActions, batch_size=256):
        """
        Initialize the brain network with the specified parameters.

        Parameters:
        - NbrStates (int): Number of inputs.
        - NbrActions (int): Number of outputs.
        - batch_size (int): Batch size for training.
        """
        self.NbrStates = NbrStates
        self.NbrActions = NbrActions
        self.batch_size = batch_size
        self.model = self.createModel()

    def createModel(self):
        """
        Define and compile the neural network model.

        Returns:
        Compiled Keras model.
        """
        model = Sequential()
        model.add(Dense(128, activation=Activation('relu'), input_dim=self.NbrStates))
        model.add(Dense(self.NbrActions, activation=Activation('softmax')))
        model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.0005))
        return model

    def train(self, x, y, epoch=1, verbose=0):
        """
        Train the model on input-output pairs.

        Parameters:
        - x: Input features (state).
        - y: Target Q-values.
        - epoch (int): Number of epochs to train.
        - verbose (int): Verbosity level.
        """
        self.model.fit(x, y, batch_size=self.batch_size, verbose=verbose)

    def predict(self, s):
        """
        Predict Q-values for a batch of states.

        Parameters:
        - s: Input states.

        Returns:
        Predicted Q-values for each action.
        """
        return self.model.predict(s, verbose=0)

    def predictOne(self, s):
        """
        Predict Q-values for a single state.

        Parameters:
        - s: Input state.

        Returns:
        Predicted Q-values for each action as a flat array.
        """
        return self.model.predict(tf.reshape(s, [1, self.NbrStates])).flatten()

    def copy_weights(self, TrainNet):
        """
        Copy weights from another Brain instance.

        Parameters:
        - TrainNet: Another Brain instance to copy weights from.
        """
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())