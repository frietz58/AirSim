import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv


class MyAirSimDroneEnv(gym.Env):
    def __init__(self, ip_address, step_length):
        super().__init__()
        self.step_length = step_length

        self.state = {
            "position": np.zeros(3),
            "velocity": np.zeros(3),
            "target_position": np.zeros(3),
            "water_collision": 0,
            "floor_collision": 0,
        }

        self.drone = airsim.MultirotorClient(ip=ip_address)

        self.observation_space = spaces.Box(
            low=np.array([
                -6,  # x pos
                -6,  # y pos
                -2,  # y pos
                -1,  # x vel
                -1,  # y vel
                -1,  # z vel
                -6,  # x target pos
                -6,  # y target pos
                -2,  # y target pos
                0,  # water collision
                0  # floor collision
            ]),
            high=np.array([6, 6, -6, 1, 1, 1, 6, 6, -6, 1, 1]),
            dtype=np.float32)

        self.action_space = spaces.Discrete(7)
        # self._setup_flight()

    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.takeoffAsync().join()
        # self.drone.moveToPositionAsync(0, 0, 5, 10).join()
        # self.drone.moveByVelocityAsync(1, -0.67, -0.8, 5).join()

    def _get_obs(self):
        self.drone_state = self.drone.getMultirotorState()

        self.state["position"] = np.array([
            self.drone_state.kinematics_estimated.position.x_val,
            self.drone_state.kinematics_estimated.position.y_val,
            self.drone_state.kinematics_estimated.position.z_val
        ])
        # self.state["position"] = self.drone_state.kinematics_estimated.position

        self.state["velocity"] = np.array([
            self.drone_state.kinematics_estimated.linear_velocity.x_val,
            self.drone_state.kinematics_estimated.linear_velocity.y_val,
            self.drone_state.kinematics_estimated.linear_velocity.z_val
        ])
        # self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity

        self.state["target_position"] = np.array([
            self.drone.simGetObjectPose("SimpleTarget").position.x_val,
            self.drone.simGetObjectPose("SimpleTarget").position.y_val,
            self.drone.simGetObjectPose("SimpleTarget").position.z_val
        ])
        # self.state["target_postion"] = self.drone.simGetObjectPose("SimpleTarget").position

        # this does not appeart to work with my water elements, TODO check whether this works while drone is in air
        # collision = self.drone.simGetCollisionInfo().has_collided

        # check distances manually
        water_dist = 1000
        for i in range(5):
            water_position = self.drone.simGetObjectPose("Water{}".format(i)).position
            water_position = np.array([
                water_position.x_val,
                water_position.y_val,
                water_position.z_val,
            ])
            water_dist = min(water_dist, np.linalg.norm(self.state["position"] - water_position))
        water_collision = 1 if water_dist < 1 else 0
        self.state["water_collision"] = water_collision

        floor_position_z = self.drone.simGetObjectPose("Floor").position.z_val
        floor_dist = np.linalg.norm(self.state["position"][2] - floor_position_z)
        floor_collision = 1 if floor_dist < 0.5 else 0
        self.state["floor_collision"] = floor_collision

        return np.concatenate([
            self.state["position"],
            self.state["velocity"],
            self.state["target_position"],
            [self.state["water_collision"]],
            [self.state["floor_collision"]]])

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_vel.x_val + quad_offset[0],
            quad_vel.y_val + quad_offset[1],
            quad_vel.z_val + quad_offset[2],
            5,
        ).join()
        # this join makes it so that we have to wait for the movement to terminate before continuing

    def _compute_reward(self):
        done = 0

        # define distance close enough to target
        distance_thresh = 1

        # magnitude of vector between position and goal
        dist = np.linalg.norm(self.state["position"] - self.state["target_position"])

        if self.state["water_collision"]:
            reward = -100
            done = 1
        elif self.state["floor_collision"]:
            reward = -50
            done = 1
        else:
            if dist < distance_thresh:
                # close enough to goal
                reward = 100
                done = 1
            else:
                reward = -dist

        if max(self.state["position"]) > 6 or min(self.state["position"]) < -6:
            # drones flies out of range in either x, y or z dir
            done = 1

        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        # reset drone for next episode
        self._setup_flight()

        # set new random goal pos
        # new_target_position = self.drone.simGetObjectPose("SimpleTarget")
        # new_target_position.x_val = random.randint(-6, 6)
        # new_target_position.y_val = random.randint(-6, 6)
        # new_target_position.z_val = random.randint(3, 4)

        # there is a bug here, z value appears to be inverted?
        # new_target_position.z_val = new_target_position.z_val * -1

        # target_pose.position = new_target_position
        # client.simSetObjectPose("SimpleTarget", target_pose)

        return self._get_obs()

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        else:
            quad_offset = (0, 0, 0)

        return quad_offset
