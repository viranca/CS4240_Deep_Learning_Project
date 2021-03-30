from item import Item
from robot import Robot
from utils import *
import numpy as np
import copy
import random
import gym
from gym import spaces
import networkx as nx
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv

class Warehouse(gym.Env):
    """
    warehouse environment
    """
    metadata = {'render.modes': ['console']}
  # Define constants for clearer code
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(self, seed, parameters):
        super(Warehouse, self).__init__() 
        self.n_columns = 7
        self.n_rows = 7
        self.n_robots_row = 1
        self.n_robots_column = 1
        self.distance_between_shelves = 6
        self.robot_domain_size = [7, 7]
        self.prob_item_appears = 0.05
        # The learning robot
        self.learning_robot_id = 0
        self.max_episode_length = 100
        self.render_bool = True
        self.render_delay = 0.
        self.obs_type = 'vector'
        self.items = []
        self.img = None
        # self.reset()
        self.max_waiting_time = 8
        self.total_steps = 0
        self.parameters = parameters
        self.reset()
        self.seed(seed)
        n_actions = 4
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=np.array([0., 0.]), high=np.array([7.0, 7.0]))
    ############################## Override ###############################

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self._add_items()
        obs = self._get_observation()
        if self.parameters['num_frames'] > 1:
            self.prev_obs = np.zeros(self.parameters['obs_size']-len(obs))
            obs = np.append(obs, self.prev_obs)
            self.prev_obs = np.copy(obs)
        self.episode_length = 0
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        self._robots_act([action])
        self._increase_item_waiting_time()
        reward = self._compute_reward(self.robots[self.learning_robot_id])
        self._remove_items()
        self._add_items()
        obs = self._get_observation()
        if self.parameters['num_frames'] > 1:
            obs = np.append(obs, self.prev_obs[:-len(obs)])
            self.prev_obs = np.copy(obs)
        # Check whether learning robot is done
        # done = self.robots[self.learning_robot_id].done
        self.total_steps += 1
        self.episode_length += 1
        done = (self.max_episode_length <= self.episode_length)
        if self.render_bool:
            self.render()
        return obs, reward, done, []


    def render(self, mode='console', delay=0.0):
        """
        Renders the environment
        """
        if mode != 'console':
            raise NotImplementedError()
        bitmap = self._get_state()
        position = self.robots[self.learning_robot_id].get_position
        bitmap[position[0], position[1], 1] += 1
        im = bitmap[:, :, 0] - 2*bitmap[:, :, 1]
        print(im)

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    ######################### Private Functions ###########################

    def _place_robots(self):
        """
        Sets robots initial position at the begining of every episode
        """
        self.robots = []
        domain_rows = np.arange(0, self.n_rows, self.robot_domain_size[0]-1)
        domain_columns = np.arange(0, self.n_columns, self.robot_domain_size[1]-1)
        for i in range(self.n_robots_row):
            for j in range(self.n_robots_column):
                robot_domain = [domain_rows[i], domain_columns[j],
                                domain_rows[i+1], domain_columns[j+1]]
                robot_position = [robot_domain[0] + self.robot_domain_size[0]//2,
                                  robot_domain[1] + self.robot_domain_size[1]//2]
                self.robots.append(Robot(self.robot_id, robot_position,
                                                  robot_domain))
                self.robot_id += 1

    def _add_items(self):
        """
        Add new items to the designated locations in the environment which
        need to be collected by the robots
        """
        item_columns = np.arange(0, self.n_columns)
        item_rows = np.arange(0, self.n_rows, self.distance_between_shelves)
        item_locs = None
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if np.random.uniform() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1
        item_rows = np.arange(0, self.n_rows)
        item_columns = np.arange(0, self.n_columns, self.distance_between_shelves)
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if np.random.uniform() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1


    def _get_state(self):
        """
        Generates a 3D bitmap: First layer shows the location of every item.
        Second layer shows the location of the robots.
        """
        state_bitmap = np.zeros([self.n_rows, self.n_columns, 2], dtype=np.int)
        for item in self.items:
            item_pos = item.get_position
            state_bitmap[item_pos[0], item_pos[1], 0] = 1 #item.get_waiting_time
        for robot in self.robots:
            robot_pos = robot.get_position
            state_bitmap[robot_pos[0], robot_pos[1], 1] = 1
        return state_bitmap

    def _get_observation(self):
        """
        Generates the individual observation for every robot given the current
        state and the robot's designated domain.
        """
        state = self._get_state()
        observation = self.robots[self.learning_robot_id].observe(state, self.obs_type)
        return observation

    def _robots_act(self, actions):
        """
        All robots take an action in the environment.
        """
        for action,robot in zip(actions, self.robots):
            robot.act(action)

    def _compute_reward(self, robot):
        """
        Computes reward for the learning robot.
        """
        reward = 0
        robot_pos = robot.get_position
        robot_domain = robot.get_domain
        for item in self.items:
            item_pos = item.get_position
            if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                reward += 1
                robot.items_collected += 1
        return reward


    def _remove_items(self):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in self.items:
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)
                elif item.get_waiting_time >= self.max_waiting_time:
                    self.items.remove(item)

    def _increase_item_waiting_time(self):
        """
        Increases items waiting time
        """
        for item in self.items:
            item.increase_waiting_time()
