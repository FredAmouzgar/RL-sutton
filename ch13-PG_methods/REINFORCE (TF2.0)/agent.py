import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp
import numpy as np
from collections import deque

class Agent:
    def __init__(self, state_size, action_size, env, env_name, gamma=0.99, alpha=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.env = env
        self.env_name = env_name
        self.gamma = gamma
        self.alpha = alpha

        self.policy = keras.Sequential([keras.layers.Dense(100, activation="relu", input_shape=(self.state_size,)),
                                       keras.layers.Dense(50, activation="relu"),
                                       keras.layers.Dense(action_size, activation=keras.activations.sigmoid)])

        self.optim = keras.optimizers.Adam(learning_rate=0.001)
        self.scores = []

    def new_episode(self):
        pass  # DO WE NEED THIS?
        #  self.I = 1

    def act(self, state):
        # TODO: Returns action from policy
        state = self.process_state(state)
        logits = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=logits)
        action = action_probs.sample()

        return action.numpy()[0], action_probs

    def generate_episode(self, max_step=300):
        rewards = []
        log_probs = []
        state = self.env.reset()
        done = False
        step = 0

        while not done and step <= max_step:
            step += 1
            action, action_probs = self.act(state)
            state_, reward, done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(action_probs.log_prob(action).numpy()[0])
            state = state_
        self.scores.append(sum(rewards))
        return rewards, log_probs

    def learn(self, rewards, log_probs):
        discounts = [self.gamma ** i for i in range(len(rewards))]
        all_G = np.array(discounts) * np.array(rewards)
        for i in range(len(all_G)):
            G = all_G[i:].sum()
            with tf.GradientTape() as tape:
                loss = -1 * self.alpha * (self.gamma ** i) * tf.convert_to_tensor(G)
                gradients = tape.gradient(loss, self.policy.trainable_variables)
            self.optim.apply_gradients(zip(gradients, self.policy.trainable_variables))


    def process_state(self, state):
        """"
        Description: Adding an extra dimension to prepare it for the NN. Then, we convert it to Tensor.
        """
        return tf.convert_to_tensor(state.reshape(1, self.state_size), dtype=tf.float32)
    
    def save(self):
        self.actor.save(f"actor_{self.env_name}.h5")
        self.critic.save(f"critic_{self.env_name}.h5")
        
    def load(self):
        self.actor = tf.keras.models.load_model(f"actor_{self.env_name}.h5")
        self.critic = tf.keras.models.load_model(f"critic_{self.env_name}.h5")