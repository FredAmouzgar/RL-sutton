import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class Bandit:
    
    def __init__(self, mu = 1, k = 10):
        """
        By default creates an array of size k and mean of 1 which represents the q*(a).
        """
        self.mu = mu
        self.k = k
        self.Qstars = np.ones(k) * mu
        #print(self.Qstars)
        
    def __step(self):
        for i in range(0, len(self.Qstars)):
            self.Qstars[i] += np.random.normal(0, 0.01)
        #print(self.Qstars)
            
    def pull_a_bandit(self, lever):
        if lever < len(self.Qstars) and lever >=0:
            reward = np.random.normal(self.Qstars[lever], 1)
            self.__step()
            return reward
        else:
            raise Exception(f"Number {lever} out of range")
            
    def reset(self):
        self.Qstars = np.ones(self.k) * self.mu
            
    def print_bandit_content(self):
        print(self.Qstars)

class Bandit_Agent:
    
    def __init__(self, bandit, initial_val=5):
        self.action_values = np.zeros(10) # Q(a)
        self.counts = np.zeros(10) # N(a)
        self.iteration_number = 0
        self.total_reward = 0
        self.bandit = bandit
        self.initial_val = initial_val # = 5 as used by Sutton in section 2.6 (P.34)
        
    def greedy_update(self):
        self.iteration_number += 1
        action_index = np.argmax(self.action_values) # Choose the best action
        reward = self.bandit.pull_a_bandit(action_index) # Get the reward from the bandit
        self.total_reward += reward # Add the reward to the accumulated reward
        self.counts[action_index] += 1 # Increment the number of that action
        self.action_values[action_index] = self.action_values[action_index] + (reward - self.action_values[action_index])/self.counts[action_index]
        return self.total_reward/self.iteration_number
    
    def optimistic_initial_value_greedy_update(self, alpha=0.1):
        if self.iteration_number == 0:   # Checks for the first iteration and the first optimistic initializations 
            self.action_values = np.ones(10) * self.initial_val
        self.iteration_number += 1
        action_index = np.argmax(self.action_values) # Choose the best action
        reward = self.bandit.pull_a_bandit(action_index) # Get the reward from the bandit
        self.total_reward += reward # Add the reward to the accumulated reward
        self.counts[action_index] += 1 # Increment the number of that action
        self.action_values[action_index] = self.action_values[action_index] + alpha * (reward - self.action_values[action_index])/self.counts[action_index]
        return self.total_reward/self.iteration_number
    
    def ucb_update(self, alpha=0.1,c=2):
        """
        Upper-Confidence-Bound Action Selection
        """
        self.iteration_number += 1 # t
        action_index = np.argmax(self.action_values + c * np.sqrt(np.log(self.iteration_number) / self.counts)) # Choose the best action
        reward = self.bandit.pull_a_bandit(action_index) # Get the reward from the bandit
        self.total_reward += reward # Add the reward to the accumulated reward
        self.counts[action_index] += 1 # Increment the number of that action
        self.action_values[action_index] = self.action_values[action_index] + alpha * (reward - self.action_values[action_index])
        return self.total_reward/self.iteration_number
    
    def epsilon_greedy_update(self, epsilon=0.01):
        self.iteration_number += 1
        if np.random.rand() > epsilon:
            action_index = np.argmax(self.action_values) # Choose the best action
        else:
            action_index = np.random.randint(0,10) # Choose a random action
        reward = self.bandit.pull_a_bandit(action_index) # Get the reward from the bandit
        self.total_reward += reward # Add the reward to the accumulated reward
        self.counts[action_index] += 1 # Increment the number of that action
        self.action_values[action_index] = self.action_values[action_index] + (reward - self.action_values[action_index])/self.counts[action_index]
        return self.total_reward/self.iteration_number
    
    def epsilon_greedy_constant_update(self, epsilon=0.1, alpha=0.1):
        self.iteration_number += 1
        if np.random.rand() > epsilon:
            action_index = np.argmax(self.action_values) # Choose the best action
        else:
            action_index = np.random.randint(0,10) # Choose a random action
        reward = self.bandit.pull_a_bandit(action_index) # Get the reward from the bandit
        self.total_reward += reward # Add the reward to the accumulated reward
        self.counts[action_index] += 1 # Increment the number of that action
        self.action_values[action_index] = self.action_values[action_index] + alpha * (reward - self.action_values[action_index])
        return self.total_reward/self.iteration_number
        
    def obj_print(self):
        print(f"{self.action_values},avg_reward={self.total_reward/self.iteration_number},best_action={self.best_action()}")
        
    def best_action(self):
        return np.max(self.action_values)
    
if __name__ == "__main__":
    agent_constant_update_bandit = Bandit() 
    #agent_constant_update = Bandit_Agent(bandit = agent_constant_update_bandit) # Moved them before the second for loop, to be honest don't know how it worked before!

    RUNS = 20
    EPOCHS = 2000 #= 200000
    epochs_for_average_last_rewards = int(EPOCHS / 2) #= 100000
    alphas = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1])
    epsilons = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1])
    average_rewards = np.zeros((RUNS, epsilons.shape[0], alphas.shape[0]))

    stime = time.time()
    for alpha in range(len(alphas)):
        print(f"The alphas[{alpha}] = {alphas[alpha]} began.")
        for epsilon in range(len(epsilons)):
            print(f"\tThe epsilons[{epsilon}] = {epsilons[epsilon]} began.")
            for run in range(0, RUNS):
                agent_constant_update_bandit.reset() # Resetting the bandit to its initial values
                agent_constant_update = Bandit_Agent(bandit = agent_constant_update_bandit) # Creating a new agent for this epsilon
                #print(f"Doing Run #{run} epsilons[{epsilon}] = {epsilons[epsilon]} ...")
                for e in range(0,EPOCHS):
                    #epsilon_greedy_constant_rewards[i] = agent_constant_update.epsilon_greedy_constant_update(epsilon=e, alpha=0.1) # Reading e from the outer for loop
                    current_reward = agent_constant_update.epsilon_greedy_constant_update(epsilon=epsilons[epsilon], alpha=alpha) # epsilons[i] sets the epsilon
                    if e >= epochs_for_average_last_rewards:
                        average_rewards[run, epsilon, alpha] += current_reward
                average_rewards[run, epsilon, alpha] /= epochs_for_average_last_rewards
            #print(f"The epsilons[{epsilon}] = {epsilons[epsilon]} finished.")
    etime = time.time()
    print("It took {} seconds to run this cell".format(etime - stime))

    # Processing collected rewards for chart

    average_rewards_mean = np.zeros((len(epsilons), len(alphas)))
    average_rewards_std = np.zeros((len(epsilons), len(alphas)))
    for alpha in range(len(alphas)):
        for epsilon in range(len(epsilons)):
            average_rewards_mean[epsilon, alpha] = average_rewards[0:RUNS, epsilon, alpha].mean()
            average_rewards_std[epsilon, alpha] = average_rewards[0:RUNS, epsilon, alpha].std()
    # Charting

    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = plt.axes(projection='3d')

    # Make data.
    X = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]) # epsilon
    Y = np.array([1/256, 1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1]) # alpha
    X_r, Y_r = np.meshgrid(X,Y)
    Z = average_rewards_mean
    surf = ax.plot_surface(X_r, Y_r, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(0, 1.5)
    ax.set_xlabel("epsilons")
    ax.set_ylabel("alphas")
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
