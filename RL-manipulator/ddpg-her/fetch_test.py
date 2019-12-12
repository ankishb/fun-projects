
import gym

env = gym.make("FetchReach-v1")

print("action_space: ", env.action_space)
print("observation_space: ", env.observation_space)

for i in range(100):
	print("**"*50)
	env.reset()
	print("**"*50)
	for j in range(1000):
		env.render()
		env.step(env.action_space.sample())