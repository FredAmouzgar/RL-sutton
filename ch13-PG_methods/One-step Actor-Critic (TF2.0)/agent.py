import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

class Agent:
    def __init__(self, state_size, action_size, env_name, gamma=0.99, alpha_w=0.15, alpha_theta=0.1):
        self.I = 1
        self.state_size = state_size
        self.action_size = action_size
        self.env_name = env_name
        self.gamma = gamma
        self.alpha_w = alpha_w
        self.alpha_theta = alpha_theta
        self.critic = keras.Sequential([keras.layers.Dense(50, activation="relu", input_shape=(self.state_size,)),
                                        keras.layers.Dense(1)])
        #self.critic.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha_w))

        self.actor = keras.Sequential([keras.layers.Dense(50, activation="relu", input_shape=(self.state_size,)),
                                       keras.layers.Dense(action_size, activation=keras.activations.sigmoid)])

        self.optim1 = keras.optimizers.Adam(learning_rate=0.001)
        self.optim2 = keras.optimizers.Adam(learning_rate=0.001)
        #self.actor.compile(optimizer=keras.optimizers.Adam(learning_rate=self.alpha_theta))

    def new_episode(self):
        self.I = 1

    def act(self, state):
        # TODO: Returns action from policy
        state = self.process_state(state)
        logits = self.actor(state)
        action_probs = tfp.distributions.Categorical(probs=logits)
        action = action_probs.sample()

        return action.numpy()[0]
        #return tf.random.categorical(logits, 1).numpy()[0,0]

    def learn(self, state, action, state_, reward, done):
        ## TD
        state = self.process_state(state)
        state_ = self.process_state(state_)

        target = reward + self.gamma * self.critic(state_) * (1 - done)
        td = target - self.critic(state)

        with tf.GradientTape() as tape:
            logits = self.actor(state)
            action_probs = tfp.distributions.Categorical(probs=logits)
            log_prob = action_probs.log_prob(action)
            actor_loss = -1 * self.alpha_theta * self.I * td * tf.squeeze(log_prob)
        gradient_actor = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.optim1.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))
        # self.actor.optimizer.apply_gradients(zip(gradient_actor, self.actor.trainable_variables))
        #self.actor.set_weights(self.actor.get_weights() + gradient_actor)
        self.I *= self.gamma

        with tf.GradientTape() as tape:
            critic_loss = -1 * self.alpha_w * td * tf.squeeze(self.critic(state))
        gradient_critic = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optim2.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))

        # self.critic.optimizer.apply_gradients(zip(gradient_critic, self.critic.trainable_variables))
        #self.critic.set_weights(self.critic.get_weights() + gradient_critic)

    def process_state(self, state):
        return tf.convert_to_tensor(state.reshape(1, self.state_size), dtype=tf.float32)
    
    def save(self):
        self.actor.save(f"actor_{self.env_name}.h5")
        self.critic.save(f"critic_{self.env_name}.h5")
        
    def load(self):
        self.actor = tf.keras.models.load_model(f"actor_{self.env_name}.h5")
        self.critic = tf.keras.models.load_model(f"critic_{self.env_name}.h5")