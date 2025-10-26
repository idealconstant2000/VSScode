# %pip install gym
# !pip install --upgrade numpy==1.26.4
# !pip uninstall tensorflow -y 
# !pip install tensorflow==2.16.2 

import os 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys 
sys.setrecursionlimit(1500) 

import gymnasium as gym 
import numpy as np 

# Create the environment 
env = gym.make('CartPole-v1') 

# Set random seed for reproducibility 
np.random.seed(42) 
env.action_space.seed(42) 
env.observation_space.seed(42)

# Suppress warnings for a cleaner notebook or console experience
import warnings
warnings.filterwarnings('ignore')

# Override the default warning function
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Import necessary libraries for the Q-Learning model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input  # Import Input layer
from tensorflow.keras.optimizers import Adam
#import gym  # Ensure the environment library is available

# Define the model building function
def build_model(state_size, action_size): 
    model = Sequential() 
    model.add(Input(shape=(state_size,)))  # Use Input layer to specify the input shape 
    model.add(Dense(24, activation='relu')) 
    model.add(Dense(24, activation='relu')) 
    model.add(Dense(action_size, activation='linear')) 
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001)) 
    return model 

# Create the environment and set up the model
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0] 
action_size = env.action_space.n 
model = build_model(state_size, action_size)

import random
import numpy as np
from collections import deque
import tensorflow as tf

# Define epsilon and epsilon_decay
epsilon = 1.0  # Starting with a high exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.99  # Faster decay rate for epsilon after each episode

# Replay memory
memory = deque(maxlen=2000)

def remember(state, action, reward, next_state, done):
    """Store experience in memory."""
    memory.append((state, action, reward, next_state, done))

def replay(batch_size=64):  # Increased batch size
    """Train the model using a random sample of experiences from memory."""
    if len(memory) < batch_size:
        return  # Skip replay if there's not enough experience

    minibatch = random.sample(memory, batch_size)  # Sample a random batch from memory
    
    # Extract information for batch processing
    states = np.vstack([x[0] for x in minibatch])
    actions = np.array([x[1] for x in minibatch])
    rewards = np.array([x[2] for x in minibatch])
    next_states = np.vstack([x[3] for x in minibatch])
    dones = np.array([x[4] for x in minibatch])
    
    # Predict Q-values for the next states in batch
    q_next = model.predict(next_states)
    # Predict Q-values for the current states in batch
    q_target = model.predict(states)
    
    # Vectorized update of target values
    for i in range(batch_size):
        target = rewards[i]
        if not dones[i]:
            target += 0.95 * np.amax(q_next[i])  # Update Q value with the discounted future reward
        q_target[i][actions[i]] = target  # Update only the taken action's Q value
    
    # Train the model with the updated targets in batch
    model.fit(states, q_target, epochs=1, verbose=0)  # Train in batch mode

    # Reduce exploration rate (epsilon) after each training step
    global epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

def act(state):
    """Choose an action based on the current state and exploration rate."""
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)  # Explore: choose a random action
    act_values = model.predict(state)  # Exploit: predict action based on the state
    return np.argmax(act_values[0])  # Return the action with the highest Q-value

# Define the number of episodes you want to train the model for
episodes = 10  # You can set this to any number you prefer
train_frequency = 5  # Train the model every 5 steps

for e in range(episodes):
    state, _ = env.reset()  # Unpack the tuple returned by env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(200):  # Limit to 200 time steps per episode
        action = act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        
        if done:
            print(f"episode: {e+1}/{episodes}, score: {time}, e: {epsilon:.2}")
            break
        
        # Train the model every 'train_frequency' steps
        if time % train_frequency == 0:
            replay(batch_size=64)  # Call replay with larger batch size for efficiency

env.close()

for e in range(10):  

    state, _ = env.reset()  # Unpack the state from the tuple 
    state = np.reshape(state, [1, state_size])  # Reshape the state correctly 
    for time in range(500):  
        env.render()  
        action = np.argmax(model.predict(state)[0])  
        next_state, reward, terminated, truncated, _ = env.step(action)  # Unpack the five return values 
        done = terminated or truncated  # Check if the episode is done 
        next_state = np.reshape(next_state, [1, state_size])  
        state = next_state  
        if done:  
            print(f"episode: {e+1}/10, score: {time}")  
            break  

env.close() 

#--------------

# Install gym if necessary
# !pip install gym

# import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from collections import deque
import random

# Initialize the environment
env = gym.make('CartPole-v1')

# Global settings
episodes = 10  # Number of episodes
batch_size = 32  # Size of the mini-batch for training
memory = deque(maxlen=2000)  # Memory buffer to store experiences

# Define state size and action size based on the environment
state_size = env.observation_space.shape[0]  # State space size from the environment
action_size = env.action_space.n  # Number of possible actions from the environment

# Define the model
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Input(shape=(state_size,)))  # Explicit Input layer
    model.add(Dense(32, activation='relu'))  # Smaller hidden layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

# Re-initialize the model with the new architecture
model = build_model(state_size, action_size)

# Placeholder for your action function (e.g., epsilon-greedy)
def act(state):
    return env.action_space.sample()  # For now, a random action is taken

# Function to remember experiences in memory
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

# Optimized function to replay experiences from memory and train the model
def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    states = np.vstack([sample[0] for sample in minibatch])
    next_states = np.vstack([sample[3] for sample in minibatch])
    targets = model.predict(states)
    target_next = model.predict(next_states)
    
    for i, (state, action, reward, next_state, done) in enumerate(minibatch):
        target = reward if done else reward + 0.95 * np.amax(target_next[i])
        targets[i][action] = target
        
    model.fit(states, targets, epochs=1, verbose=0)

# Train the model with the modified architecture
for e in range(episodes):
    state, _ = env.reset()  # Unpack the state from the tuple
    state = np.reshape(state, [1, state_size])
    for time in range(200):  # Reduced number of steps per episode
        action = act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            print(f"episode: {e+1}/{episodes}, score: {time}")
            break
        
        if len(memory) > batch_size and time % 10 == 0:  # Train every 10 steps
            replay(batch_size)  # Pass the batch size to replay()

env.close()