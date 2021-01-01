import numpy as np
import matplotlib.pyplot as plt


class RandomWalk:
    def __init__(self, non_terminal_states=5, init_state=3):
        self.world = np.arange(non_terminal_states + 2)
        ##
        self.init_state = init_state
        self.state = self.init_state
        self.actions = {'left': -1, 'right': 1}
        self.actions_n = len(self.actions)
        self.goal = self.world[-1]
        self.normal_reward = 0
        self.goal_reward = 1

    def reset(self):
        self.state = self.init_state

    def random_step(self):
        action = np.random.choice(['left', 'right'])
        return self.step(action)

    def step(self, action):
        """Take a step in the environment based on the current state, and the given action

        Parameter: action = 'left' / 'right'
        Returns: A tuple consists of (s',r,done)
        """
        new_state = self.state + self.actions[action]
        self.state = new_state
        reward = self.normal_reward if new_state != self.goal else self.goal_reward
        if new_state == self.goal or new_state == self.world[0]:
            done = True
        else:
            done = False
        return new_state, reward, done

    def render(self, mode="g"):
        world_for_repr = np.zeros(np.stack([self.world, self.world]).shape)
        world_for_repr[:, 0] = 3
        world_for_repr[:, -1] = 3
        world_for_repr[:, self.state] = 6

        if mode == "g":
            plt.imshow(world_for_repr)
            msg = "State: " + str(self.state)
            plt.title(msg)
            plt.show()
        elif mode == "t":
            print(world_for_repr)

    @property
    def observation_space(self):
        n = len(self.world)
        return {"n": len(self.world), "Terminal states": 2, "Non-terminal states": len(self.world) - 2}

    @property
    def action_space(self):
        return list(self.actions.keys())