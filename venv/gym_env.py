import gymnasium as gym
from gymnasium import spaces
import numpy as np

class AGIBoxEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=10, render_mode=None):
        self.grid_size = grid_size
        self.observation_space = spaces.Dict({
            "local_view": spaces.Box(low=0, high=3, shape=(3, 3), dtype=np.uint8),
            "energy": spaces.Box(low=0, high=100, shape=(1,), dtype=np.uint8)
        })
        self.action_space = spaces.Discrete(6)  # up, down, left, right, rest, consume

        self.render_mode = render_mode
        self.max_energy = 100

        self.energy_stages = [60, 30, 15, 7]
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        self.agent_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        self.energy = 50
        self.steps = 0
        self.total_steps = 0
        self.stage_index = 0
        self.energy_deadline = self.energy_stages[self.stage_index]
        self.steps_since_last_energy = 0

        # Spawn initial energy and danger
        for _ in range(self.grid_size // 2):
            x, y = np.random.randint(0, self.grid_size, size=2)
            self.grid[x, y] = np.random.choice([1, 2], p=[0.8, 0.2])  # 1 = energy, 2 = danger

        return self._get_obs(), {}

    def _get_obs(self):
        x, y = self.agent_pos
        view = self.grid[max(0, x-1):x+2, max(0, y-1):y+2]
        padded_view = np.zeros((3, 3), dtype=np.uint8)
        padded_view[:view.shape[0], :view.shape[1]] = view
        return {
            "local_view": padded_view,
            "energy": np.array([self.energy], dtype=np.uint8)
        }

    def step(self, action):
        self.total_steps += 1
        self.steps_since_last_energy += 1
        reward = -1  # default metabolic penalty
        done = False

        x, y = self.agent_pos.copy()

        if action == 0 and x > 0:           # up
            self.agent_pos[0] -= 1
        elif action == 1 and x < self.grid_size - 1:  # down
            self.agent_pos[0] += 1
        elif action == 2 and y > 0:         # left
            self.agent_pos[1] -= 1
        elif action == 3 and y < self.grid_size - 1:  # right
            self.agent_pos[1] += 1
        elif action == 4:                   # rest
            pass
        elif action == 5:                   # consume
            x, y = self.agent_pos
            if self.grid[x, y] == 1:
                reward += 10
                self.energy = min(self.max_energy, self.energy + 20)
                self.grid[x, y] = 0
                self.steps_since_last_energy = 0

                # Shrink deadline window
                if self.stage_index + 1 < len(self.energy_stages):
                    self.stage_index += 1
                self.energy_deadline = self.energy_stages[self.stage_index]

                # Respawn new energy
                self._spawn_energy()

            elif self.grid[x, y] == 2:
                reward -= 10  # danger zone penalty

        # Apply energy cost
        self.energy -= 1

        # Death by starvation
        if self.steps_since_last_energy >= self.energy_deadline:
            done = True
            reward -= 20  # penalty for missing deadline

        # Death by energy drain or too many steps
        if self.energy <= 0 or self.total_steps >= 500:
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {}

    def _spawn_energy(self):
        while True:
            x = np.random.randint(0, self.grid_size)
            y = np.random.randint(0, self.grid_size)
            if (x, y != self.agent_pos.tolist()) and self.grid[x, y] == 0:
                self.grid[x, y] = 1
                break

    def render(self):
        if self.render_mode == "human":
            grid_copy = self.grid.copy().astype(str)
            x, y = self.agent_pos
            grid_copy[x, y] = "A"
            print("\n".join(" ".join(row) for row in grid_copy))
            print(f"Energy: {self.energy} | Steps: {self.total_steps} | Deadline: {self.energy_deadline - self.steps_since_last_energy}")
