import gymnasium as gym
from gym_env import AGIBoxEnv
from evol import AGIAgent, Phenotype, Population  # Make sure Population is in evol.py
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define env factory for Population
def make_env():
    return AGIBoxEnv()


# Step 2: Set up evolution
pop = Population(
    size=20,
    input_size=10,     # 3x3 view + 1 energy
    hidden_size=16,
    output_size=6,
    env_fn=make_env
)

generations = 10
for gen in range(generations):
    pop.evolve()
    best_agent, score = pop.best_agent()
    print(f"Generation {gen+1}: Best Score = {score}")


plt.plot(pop.history["best"], label="Best Score")
plt.plot(pop.history["avg"], label="Average Score")
plt.plot(pop.history["worst"], label="Worst Score")
plt.xlabel("Generation")
plt.ylabel("Fitness Score")
plt.title("Evolution Progress")
plt.legend()
plt.grid(True)
plt.savefig("evolution_progress.png")

# Step 3 (optional): Run best agent visually after evolution
# print("\nRunning best evolved agent:")
# env = make_env()
# obs, _ = env.reset()
# done = False
# total_reward = 0
# while not done:
    # action = best_agent.act(obs)
    # obs, reward, done, _, _ = env.step(action)
    # total_reward += reward
    # env.render()

# print(f"Total Reward (Evolved): {total_reward}")
