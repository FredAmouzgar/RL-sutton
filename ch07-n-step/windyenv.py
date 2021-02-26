import numpy as np
import matplotlib.pyplot as plt

class WindyGridWorld:
    def __init__(self, rows=7, cols=10, init_state=30, goal = 37):
        self.rows = 7
        self.cols = 10
        self.world = np.arange(0,70).reshape((self.rows, self.cols))
        # Wind
        self.wind = np.zeros((self.rows, self.cols))
        self.wind[:,[3,4,5,8]]=1
        self.wind[:,[6,7]] = 2
        ##
        self.init_state = init_state
        self.state = self.init_state
        self.actions = {0:'right', 1:'left', 2:'up', 3:'down'}
        self.actions_n = len(self.actions)
        self.states_n = rows * cols
        self.goal = goal
    
    def get_state(self,x,y):
        return x*10+y
    
    def coord(self, s):
        return (int(s//10), int(s%10))
    
    def _move(self,s,a):
        X, Y = self.coord(s)
        if a == 0:   # right
            dx, dy = 0,1
        elif a == 1: # left
            dx, dy = 0,-1
        elif a == 2: # up
            dx, dy = -1, 0
        elif a == 3: # down
            dx, dy = 1, 0
        wind = self.wind[X,Y]
        dx -= wind
        X += dx
        Y += dy
        if X < 0:
            X=0
        elif X > 6:
            X=6
        if Y<0:
            Y=0
        elif Y>9:
            Y=9
        return self.get_state(X,Y)
    
    def reset(self):
        self.state = self.init_state
    
    def step(self, action):
        """Take a step in the environment based on the current state, and the given action
        
        Parameter: action
        Returns: A tuple consists of (s',r,done)
        """
        new_state = self._move(self.state, action)
        self.state = new_state
        reward = -1 if new_state != self.goal else 0
        done = False if new_state != self.goal else True
        return new_state,reward,done
    
    def render(self, mode="g"):
        world_for_repr = np.copy(self.wind)
        X,Y = self.coord(self.state)
        world_for_repr[X,Y] = 4
        world_for_repr[3,7] = 3
        
        if mode=="g":
            plt.imshow(world_for_repr)
            if X == 3 and Y == 7:
                msg = "State: ({},{}) - Won!".format(X,Y)
            else:
                msg = "State: ({},{})".format(X,Y)
            plt.title(msg)
            plt.show()
        elif mode == "t":
            print(world_for_repr)