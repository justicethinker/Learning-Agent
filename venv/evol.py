import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt



class Genome:
    def __init__(self, input_size, hidden_size, output_size):
        # Genotype: a flat list of weights and biases for a single-layer neural net
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Total number of weights in a simple feedforward NN
        self.weights_input_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.bias_hidden = np.random.uniform(-1, 1, (hidden_size,))
        self.weights_hidden_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (output_size,))

    def mutate(self, mutation_rate=0.1, mutation_strength=0.5):
        def mutate_array(arr):
            mask = np.random.rand(*arr.shape) < mutation_rate
            noise = np.random.normal(0, mutation_strength, arr.shape)
            arr += mask * noise

        mutate_array(self.weights_input_hidden)
        mutate_array(self.bias_hidden)
        mutate_array(self.weights_hidden_output)
        mutate_array(self.bias_output)

    def crossover(parent1, parent2):

        child = Genome(parent1.input_size, parent1.hidden_size, parent1.output_size)

        def crossover_array(a, b):

            mask = np.random.rand(*a.shape) < 0.5
            return np.where(mask, a, b)

            child.weights_input_hidden = crossover_array(parent1.weights_input_hidden, parent2.weights_input_hidden)
            child.bias_hidden = crossover_array(parent1.bias_hidden, parent2.bias_hidden)
            child.weights_hidden_output = crossover_array(parent1.weights_hidden_output, parent2.weights_hidden_output)
            child.bias_output = crossover_array(parent1.bias_output, parent2.bias_output)

            return child
    

class Phenotype:
    def __init__(self, genome: Genome):
        self.w1 = genome.weights_input_hidden
        self.b1 = genome.bias_hidden
        self.w2 = genome.weights_hidden_output
        self.b2 = genome.bias_output

    def forward(self, obs):
        # obs is a dict from the environment
        local_view = obs["local_view"].flatten() / 3.0  # Normalize
        energy = obs["energy"][0] / 100.0
        input_vector = np.concatenate([local_view, [energy]])

        # Simple feedforward NN
        h = np.tanh(np.dot(self.w1, input_vector) + self.b1)
        out = np.dot(self.w2, h) + self.b2

        action = np.argmax(out)
    
        print(f"Input: {input_vector[:5]}... | Output: {out} | Action: {action}")
    
        # Return action
        return action


class Population:
    def __init__(self, size, input_size, hidden_size, output_size, env_fn):
        self.size = size
        self.env_fn = env_fn
        self.population = [AGIAgent(input_size, hidden_size, output_size) for _ in range(size)]
        self.history = {
            "best": [],
            "avg": [],
            "worst": []
        }


    def evaluate_agent(self, agent):
        env = self.env_fn()
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        energy_collected = 0
        danger_eaten = 0
        visited = set()
    
        while not done:
            action = agent.act(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1

            x, y = env.agent_pos
            visited.add((x, y))

            if reward >= 9:  # indicates energy collected
                energy_collected += 1
            if reward <= -9:  # likely danger consumed
                danger_eaten += 1

        # Final fitness computation
        fitness = (
            steps +                                 # longevity
            (energy_collected * 10) +               # energy collected
            (len(visited)) +                        # exploration
            (0.1 * env.energy) -                    # leftover energy efficiency
            (5 * danger_eaten)                      # penalty for danger
        )
        return fitness

    
    def evolve(self, elite_fraction=0.2, mutation_rate=0.1):
        # Step 1: Evaluate fitness of current population
        fitness_scores = []
        for agent in self.population:
            env = self.env_fn()
            score = self.evaluate_agent(agent)
            agent.fitness = score
            fitness_scores.append(score)

        self.history["best"].append(np.max(fitness_scores))
        self.history["avg"].append(np.mean(fitness_scores))
        self.history["worst"].append(np.min(fitness_scores))

         # Select elites
        elite_count = max(1, int(self.size * elite_fraction))
        sorted_agents = sorted(self.population, key=lambda a: a.fitness, reverse=True)
        elites = sorted_agents[:elite_count]
       
        # Step 4: Generate new population through mutation of elites
        new_population = []
        while len(new_population) < self.size:
            parent = np.random.choice(elites)
            child_genome = deepcopy(parent.genome)
            child_genome.mutate(mutation_rate)
            child = AGIAgent(
                input_size=parent.genome.input_size,
                hidden_size=parent.genome.hidden_size,
                output_size=parent.genome.output_size
            )
            child.genome = child_genome
            child.controller = Phenotype(child_genome)
            new_population.append(child)

        # Step 5: Replace old population
        self.population = new_population


    def best_agent(self):
        fitness_scores = [(agent, self.evaluate_agent(agent)) for agent in self.population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        return fitness_scores[0]

       
        

class AGIAgent:
    def __init__(self, input_size=10, hidden_size=16, output_size=6):
        self.genome = Genome(input_size, hidden_size, output_size)
        self.controller = Phenotype(self.genome)

    def act(self, observation):
        return self.controller.forward(observation)
