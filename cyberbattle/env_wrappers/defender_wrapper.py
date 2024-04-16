import time
import copy
import logging
import networkx
from networkx import convert_matrix
from typing import NamedTuple, Optional, Tuple, List, Dict, TypeVar, TypedDict, cast

import numpy
import gym
from gym import spaces
from gym.utils import seeding

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from cyberbattle._env.shared_cyberbattle_env import EnvironmentBounds, AttackerGoal, DefenderGoal, DefenderConstraint
from cyberbattle._env.defender import DefenderAgent, ScanAndReimageCompromisedMachines
from cyberbattle.simulation.model import PortName, PrivilegeLevel
from cyberbattle.simulation import commandcontrol, model, actions
from cyberbattle._env.discriminatedunion import DiscriminatedUnion
from cyberbattle.agents.baseline import agent_wrapper as w


import numpy as np
import random
from gym import spaces

from gym import spaces
import numpy as np


class DefenderWrapper:
    def __init__(self, environment, name):
        self.env = environment
        self.network = environment.network
        self.defender_actions = DefenderAgentActions(environment)
        self.id_mapping = self.create_id_mapping()  # Mapping of numerical IDs to string IDs
        self.action_space = self.define_action_space()
        self.observation_space = self.define_observation_space()
        self.last_attacker_reward = 0
        self.name = name

    def create_id_mapping(self):
        # creates a mapping from numerical IDs to string IDs
        mapping = {}
        for i, node in enumerate(self.env.network.nodes):
            mapping[i] = node
        return mapping

    def render(self):
        # implements rendering logic here
        print("Rendering the current state of the environment")

    def close(self):
        pass

    '''
    Defender's Action Space
    '''

    def define_action_space(self):
        # define the action space for the defender
        return spaces.Dict({
            'action_type': spaces.Discrete(5),  # 5 different actions
            'node_id': spaces.Discrete(len(self.id_mapping)),  # Based on the number of nodes
            'parameter': spaces.Discrete(10)  # Example parameter space
        })

    '''
    Observation Space for the Defender
    '''

    def define_observation_space(self):
        # define the observation space for the defender
        return spaces.Dict({
            'infected_nodes': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'firewall_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'service_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32)
        })

    def reimage_node(self, node_id, environment):
        # mark the node for re-imaging and make it unavailable until re-imaging completes
        self.node_reimaging_progress[node_id] = self.REIMAGING_DURATION
        node_info = environment.get_node(node_id)

    def get_node(self, node_id):
        # implement this method to access node information from the environment
        return self.env.get_node(node_id)

    def validate_action(self, action):
        # validate the action taken by the defender
        action_type, numerical_node_id, parameter = action['action_type'], action['node_id'], action['parameter']
        node_id = self.id_mapping.get(numerical_node_id)
        if node_id is None:
            return False  # Invalid node ID

        node_info = self.env.network.nodes[node_id]['data']
        if action_type == 0 and not node_info.reimagable:
            return False  # Node is not re-imageable

        return True  # Action is valid

    def step(self, action):
        reward = 0
        done = False
        info = {}

        if self.validate_action(action):
            action_type, numerical_node_id = action['action_type'], action['node_id']
            node_id = self.id_mapping[numerical_node_id]
            parameter = action['parameter']
            action_result = self.perform_action(action_type, node_id, parameter)
            if action_result is None:
                print(f"Error: Action result is None for action {action_type} on node {node_id}")
                observation = self.get_observation()
                observation['action_mask'] = self.compute_defender_action_mask()
                return observation, -10, done, {'error': 'Invalid action result'}
            info['action_result'] = action_result

            reward = self.calculate_reward(action_result)

        else:
            reward = -10
            info['action_result'] = {'status': 'invalid', 'reason': 'Invalid action parameters'}

        observation = self.get_observation()
        observation['action_mask'] = self.compute_defender_action_mask()
        return observation, reward, done, info

    def perform_action(self, action_type, node_id, parameter):
        if action_type == 0:  # Reimage Node
            return self.defender_actions.reimage_node(node_id, self.env)
        elif action_type == 1:  # Block Traffic
            return self.defender_actions.block_traffic(node_id, parameter, incoming=True)
        elif action_type == 2:  # Allow Traffic
            return self.defender_actions.allow_traffic(node_id, parameter, incoming=True)
        elif action_type == 3:  # Stop Service
            return self.defender_actions.stop_service(node_id, parameter)
        elif action_type == 4:  # Start Service
            return self.defender_actions.start_service(node_id, parameter)
        else:
            return {'status': 'error', 'reason': 'Unknown action type'}

    def calculate_reward(self, action_result):
        if action_result['status'] == 'success':
            if action_result.get('type') == 'reimage':
                return 50  # Higher reward for re-imaging a compromised node
            else:
                return 25  # Standard reward for other successful actions
        elif action_result['status'] == 'failed':
            return -15  # Penalty for failed actions
        elif action_result['status'] == 'unnecessary':
            return -5  # Smaller penalty for unnecessary actions
        return 0  # Default reward for other cases

    def get_observation(self):
        # Generate the defender's observation
        observation = {
            'infected_nodes': np.random.randint(0, 2, size=len(self.id_mapping)),
            'firewall_status': np.random.randint(0, 2, size=len(self.id_mapping)),
            'service_status': np.random.randint(0, 2, size=len(self.id_mapping))
        }
        return observation

    def update_last_attacker_reward(self, reward):
        # Update the last reward obtained by the attacker
        self.last_attacker_reward = reward

    def default_or_preventive_action(self):
        random_node_id = random.choice(list(self.id_mapping.keys()))
        return self.create_action('Check Firewall', random_node_id)

    def reset(self):
        initial_observation = self.env.reset()
        return initial_observation

    def choose_action(self, state):
        """
        Choose an action based on the current state of the environment.
        """
        print("Choosing action based on the current state")

        # Detect infected nodes
        infected_nodes = self.find_infected_nodes()
        print(f"Infected nodes: {infected_nodes}")

        # If there are infected nodes, reimage the first one
        if infected_nodes:
            #node_id_to_reimage = infected_nodes[0]
            node_id_to_reimage = random.choice(infected_nodes)
            print(f"Reimaging node: {node_id_to_reimage}")
            return self.create_action('Reimage Node', node_id_to_reimage)

        # List of preventive actions
        preventive_actions = ['Check Firewall', 'Block Traffic', 'Allow Traffic', 'Stop Service', 'Start Service']
        print("No infected nodes found, selecting a preventive action")

        # Randomly select a preventive action
        selected_preventive_action = random.choice(preventive_actions)
        print(f"Selected preventive action: {selected_preventive_action}")

        # Randomly select a node for the preventive action
        node_id_for_preventive_action = self.select_node_for_preventive_action()
        print(f"Node selected for preventive action: {node_id_for_preventive_action}")

        return self.create_action(selected_preventive_action, node_id_for_preventive_action)

    def find_infected_nodes(self):
        """
        Find and return a list of numerical IDs of infected (or compromised) nodes.
        """
        infected_nodes = []
        for numerical_id, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            if node_info.agent_installed or self.has_vulnerabilities(node_id):
                infected_nodes.append(numerical_id)
        return infected_nodes

    def has_vulnerabilities(self, node_id):
        """
        Check if a node has vulnerabilities.
        """
        node_info = self.env.network.nodes[node_id]['data']
        return bool(node_info.vulnerabilities)

    def select_node_for_preventive_action(self):
        """
        Select a node for a preventive action
        """
        return random.choice(list(self.id_mapping.keys()))

    def create_action(self, action_type, node_id):
        """
        Create an action dictionary based on the action type and node ID.
        """
        action_type_mapping = {
            'Reimage Node': 0,
            'Block Traffic': 1,
            'Allow Traffic': 2,
            'Stop Service': 3,
            'Start Service': 4
        }
        if action_type in action_type_mapping:
            action_number = action_type_mapping[action_type]
        else:
            action_number = random.choice(list(action_type_mapping.values()))
        return {'action_type': action_number, 'node_id': node_id, 'parameter': 0}

    def compute_defender_action_mask(self):
        # initialise a blank action mask
        action_mask = {
            'Reimage Node': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Block Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Allow Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Stop Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Start Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32)
        }

        # update the action mask based on the current state of the environment
        for numerical_id, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']

            # if node is infected or has vulnerabilities, defender can reimage node
            if node_info.agent_installed or self.has_vulnerabilities(node_id):
                action_mask['Reimage Node'][numerical_id] = 1

            # defender can block traffic and allow traffic on any node
            action_mask['Block Traffic'][numerical_id] = 1
            action_mask['Allow Traffic'][numerical_id] = 1

            # if a service is running on the node, defender can stop service
            if any(service.running for service in node_info.services):
                action_mask['Stop Service'][numerical_id] = 1

            # if a service is stopped on the node, defender can start service
            if any(not service.running for service in node_info.services):
                action_mask['Start Service'][numerical_id] = 1

        return action_mask

    def defender_goal_reached(self):
        for _, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            if node_info.agent_installed:  # Assuming agent_installed indicates attacker's presence
                return False
        return True

    def sample_valid_action(self):
        """
        Randomly selects a valid action for the defender.
        """
        # randomly select an action type
        action_type = random.randint(0, 4)  # Assuming 5 different actions (0 to 4)

        # randomly select a node ID
        node_id = random.choice(list(self.id_mapping.keys()))

        # create a random parameter if needed
        parameter = random.randint(0, 9)  # Assuming parameter space is 0 to 9

        # construct the action
        action = {
            'action_type': action_type,
            'node_id': node_id,
            'parameter': parameter
        }

        # ensure the action is valid, otherwise re-sample
        while not self.validate_action(action):
            action['action_type'] = random.randint(0, 4)
            action['node_id'] = random.choice(list(self.id_mapping.keys()))
            action['parameter'] = random.randint(0, 9)

        return action
