class MyEnv():
    def __init__(self, size=10, reward_type='sparse'):
        self.size = size
        self.reward_type = reward_type
        # distance threshold
        self.thr = 0.5 
    
    def reset(self):
        # range (-size~size)
        self.goal = self.size * (2*np.random.random(2)-1) 
        self.state = self.size * (2*np.random.random(2)-1)
        return np.copy(self.state/self.size), np.copy(self.goal/self.size)
    
    def reward_func(self, state, goal):
        good_done = np.linalg.norm(state-goal) <= self.thr
        bad_done = np.max(np.abs(state)) > self.size
        if self.reward_type == 'sparse':
            reward = 0 if good_done else -1
        else:
            reward = 10 if good_done else -10 if bad_done else -np.linalg.norm(state-goal)/200
        return good_done or bad_done, reward
    
    def step(self, action, scale=4):
        self.state += action/scale
        done, reward = self.reward_func(self.state, self.goal)
        return np.copy(self.state/self.size), reward, done
    
    def render(self):
        print("\rstate :", np.array_str(self.state), "goal :", np.array_str(self.goal),end=' '*10)