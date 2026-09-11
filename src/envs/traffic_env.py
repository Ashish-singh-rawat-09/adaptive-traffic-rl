import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci


class SumoTrafficEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, net_file, route_file, use_gui=False, sim_max_time=1000):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui
        self.sim_max_time = sim_max_time

        self.sumo_binary = "sumo-gui" if self.use_gui else "sumo"

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(8,), dtype=np.float32
        )

        self.tls_id = None
        self.incoming_lanes = []
        self.current_phase = 0
        self.yellow_time = 3
        self.green_step = 10
        self.state_len = 8

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        try:
            traci.close()
        except Exception:
            pass

        sumo_cmd = [
            self.sumo_binary,
            "-n", self.net_file,
            "-r", self.route_file,
            "--no-step-log", "true",
            "--waiting-time-memory", "1000",
            "--time-to-teleport", "-1"
        ]

        traci.start(sumo_cmd)

        tls_list = traci.trafficlight.getIDList()
        if not tls_list:
            raise RuntimeError("No traffic light found in the network!")
        self.tls_id = tls_list[0]

        all_lanes = traci.trafficlight.getControlledLanes(self.tls_id)
        self.incoming_lanes = list(dict.fromkeys(all_lanes))

        # Dynamic light state length check
        current_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        self.state_len = len(current_state)

        self.current_phase = 0
        self._apply_phase(self.current_phase)

        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        if action != self.current_phase:
            # Yellow transition
            half = self.state_len // 2
            if self.current_phase == 0:
                yellow_state = "y" * half + "r" * (self.state_len - half)
            else:
                yellow_state = "r" * half + "y" * (self.state_len - half)
            
            traci.trafficlight.setRedYellowGreenState(self.tls_id, yellow_state)
            for _ in range(self.yellow_time):
                traci.simulationStep()

            self.current_phase = action
            self._apply_phase(self.current_phase)

        for _ in range(self.green_step):
            traci.simulationStep()

        obs = self._get_observation()
        reward = self._compute_reward()

        current_sim_time = traci.simulation.getTime()
        terminated = current_sim_time >= self.sim_max_time
        truncated = False
        info = {"simulation_time": current_sim_time}

        if terminated:
            try:
                traci.close()
            except Exception:
                pass

        return obs, reward, terminated, truncated, info

    def _apply_phase(self, phase_index):
        half = self.state_len // 2
        if phase_index == 0:
            phase_state = "G" * half + "r" * (self.state_len - half)
        else:
            phase_state = "r" * half + "G" * (self.state_len - half)
        traci.trafficlight.setRedYellowGreenState(self.tls_id, phase_state)

    def _get_observation(self):
        queues = []
        for lane in self.incoming_lanes[:8]:
            queues.append(float(traci.lane.getLastStepHaltingNumber(lane)))

        while len(queues) < 8:
            queues.append(0.0)

        return np.array(queues[:8], dtype=np.float32)

    def _compute_reward(self):
        total_waiting_time = 0.0
        total_queue = 0.0
        for lane in self.incoming_lanes:
            total_waiting_time += traci.lane.getWaitingTime(lane)
            total_queue += traci.lane.getLastStepHaltingNumber(lane)

        return float(-(0.5 * total_queue + 0.1 * total_waiting_time))

    def close(self):
        try:
            traci.close()
        except Exception:
            pass