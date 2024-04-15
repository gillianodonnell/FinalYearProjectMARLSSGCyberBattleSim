import time
import copy
import logging
import networkx
from networkx import convert_matrix
from typing import NamedTuple, Optional, Tuple, List, Dict, TypeVar, TypedDict, cast
import cyberbattle.agents.baseline.agent_tabularqlearning as a
import cyberbattle.agents.baseline.learner as learner
from cyberbattle.agents.baseline.agent_wrapper import Verbosity
import numpy
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import logging
import progressbar
from typing import Optional, List
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from cyberbattle.agents.baseline.agent_wrapper import ActionTrackingStateAugmentation, StateAugmentation
from cyberbattle.agents.baseline.plotting import PlotTraining, plot_averaged_cummulative_rewards
import progressbar
import math
from cyberbattle.agents.baseline import learner
import sys
from cyberbattle.simulation.model import FirewallRule
from cyberbattle._env.shared_cyberbattle_env import EnvironmentBounds, AttackerGoal, DefenderGoal, DefenderConstraint
from cyberbattle._env.defender import DefenderAgent,ScanAndReimageCompromisedMachines
from cyberbattle.simulation.model import PortName, PrivilegeLevel
from cyberbattle.simulation import commandcontrol, model, actions
from cyberbattle._env.discriminatedunion import DiscriminatedUnion
from cyberbattle.agents.baseline import agent_wrapper as w
import cyberbattle.agents.baseline.learner as learner
from cyberbattle.agents.baseline.learner import Learner
import cyberbattle.agents.baseline.agent_tabularqlearning as a
import cyberbattle.agents.baseline.agent_ddql as ddqla
import cyberbattle.agents.baseline.agent_dql as dqla
from typing import Dict
import datetime
from cyberbattle.simulation import model
from cyberbattle.simulation.model import PrivilegeLevel, MachineStatus
from cyberbattle.simulation.model import Identifiers, NodeID, NodeInfo, Environment
from cyberbattle._env.cyberbattle_chain import CyberBattleChain
import numpy as np
from cyberbattle._env.cyberbattle_env import Action
import numpy as np
from typing import Tuple, Optional
import logging
import abc
from cyberbattle._env import cyberbattle_env
import numpy as np
import random
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import logging
import progressbar
from typing import Optional, List
from collections import defaultdict
#new
import numpy as np
from typing import Tuple

Breakdown = TypedDict('Breakdown', {
    'local': int,
    'remote': int,
    'connect': int
})

Outcomes = TypedDict('Outcomes', {
    'reward': Breakdown,
    'noreward': Breakdown
})

Stats = TypedDict('Stats', {
    'exploit': Outcomes,
    'explore': Outcomes,
    'exploit_deflected_to_explore': int
})

TrainedLearner = TypedDict('TrainedLearner', {
    'all_episodes_rewards': List[List[float]],
    'all_episodes_availability': List[List[float]],
    'learner': Learner,
    'trained_on': str,
    'title': str
})

class DefenderWrapper:
    def __init__(self, environment, name):
        self.env = environment
        self.network = environment.network
        self.defender_actions = DefenderAgentActions(environment)
        self.id_mapping = self.create_id_mapping()
        self.action_space = self.define_action_space()
        #self.observation_space = self.define_observation_space()
        self.last_attacker_reward = 0
        self.name = name
        self.performance_metrics = {
            'actions_taken': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'unnecessary':0
        }

    def create_id_mapping(self):
        mapping = {}
        for i, node in enumerate(self.env.network.nodes):
            mapping[i] = node
        return mapping

    def define_action_space(self):
        return spaces.Dict({
            'action_type': spaces.Discrete(5),
            'node_id': spaces.Discrete(len(self.id_mapping)),
            'parameter': spaces.Discrete(10)
        })

    def define_observation_space(self):
        return spaces.Dict({
            'infected_nodes': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'firewall_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'service_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32)
        })

    def validate_action(self, action):
        action_type, numerical_node_id, _ = action.values()
        print(action_type,numerical_node_id,_)
        node_id = self.id_mapping.get(numerical_node_id)
        action1 = self.create_action(action_type, node_id)
        #print('new action1',action1)
        action_type1, numerical_node_id, _ = action.values()
        #print('action type1',action1['action_type'])
        #print('node id',node_id)
        #print('numerical_node_id',numerical_node_id)
        if node_id is None or action1['action_type'] not in [0, 1, 2, 3, 4]:
            #print(node_id)
            #print(action_type)
            return False
        return True

    def step(self, action):
        print(f"[DEBUG] Received action: {action}")

        if not self.validate_action(action):
            #print("[WARNING] Invalid action")
            return self.construct_step_response(-10, {'error': 'Invalid action'}, False)

        action_type, numerical_node_id, parameter = action.values()
        node_id = self.id_mapping[numerical_node_id]
        #print(f"[DEBUG] Processed Node ID: {node_id} (Numerical ID: {numerical_node_id})")
        action1 = self.create_action(action_type, node_id)
        action_result = self.perform_action(action_type, node_id, parameter)

        if action_result is None:
            #print("[ERROR] Action result is None")
            return self.construct_step_response(-10, {'error': 'Action result is None'}, False)

        reward = self.calculate_reward(action_result)
        done = self.defender_goal_reached()
        self.update_performance_metrics(action_result)

        response = self.construct_step_response(reward, action_result, done)
        #print(f"[DEBUG] Step response: {response}")
        return response

    def perform_action(self, action_type, node_id, parameter):
        # Get the current state of the node
        node_info = self.env.get_node(node_id)

        if action_type == 0:  # Reimage Node
            return self.defender_actions.reimage_node(node_id, self.env)

        elif action_type == 1:  # Block Traffic
            return self.defender_actions.block_traffic(node_id, parameter, incoming=True)

        elif action_type == 2:  # Allow Traffic
            return self.defender_actions.allow_traffic(node_id, parameter, incoming=True)

        elif action_type == 3:  # Stop Service
            if any(service.name == parameter and service.running for service in node_info.services):
                return self.defender_actions.stop_service(node_id, parameter)
            else:
                return {'status': 'unnecessary', 'reason': 'Service not running or not found', 'action': 'stop_service'}

        elif action_type == 4:  # Start Service
            if any(service.name == parameter and not service.running for service in node_info.services):
                return self.defender_actions.start_service(node_id, parameter)
            else:
                return {'status': 'unnecessary', 'reason': 'Service already running or not found', 'action': 'start_service'}

        else:
            return {'status': 'error', 'reason': 'Unknown action type', 'action': 'unknown'}


    def calculate_reward(self, action_result):
      if action_result['status'] == 'success':
          if action_result['action'] == 'reimage_node':
              return 5000  #higher reward for re-imaging a compromised node
          else:
              return 4000  #standard reward for other successful actions
      elif action_result['status'] == 'failed':
          #print('failed')
          #print(action_result)
          if action_result['action'] == 'reimage_node':
              #print('reimaged')
              return -4000
          else:
              #print('not reimaged')
              return -5000  #penalty for failed actions
      elif action_result['status'] == 'unnecessary':
          return -1000  #smaller penalty for unnecessary actions
      return 0

    def get_observation(self):
      #print('environment nodes',self.env.network.nodes)
      infected_nodes = [int(self.env.is_node_infected(node)) for node in self.env.network.nodes]
      #print('infected_nodes',infected_nodes)
      firewall_status = [int(self.env.get_firewall_status(node)) for node in self.env.network.nodes]
      service_status = [int(self.env.get_service_status(node)) for node in self.env.network.nodes]
      #print('firewall_status',firewall_status)
      #print('service_status',service_status)
      return {
          'infected_nodes': np.array(infected_nodes, dtype=np.int32),
          'firewall_status': np.array(firewall_status, dtype=np.int32),
          'service_status': np.array(service_status, dtype=np.int32)
      }


    def update_performance_metrics(self, action_result):
        self.performance_metrics['actions_taken'] += 1
        if action_result['status'] == 'success':
            self.performance_metrics['successful_actions'] += 1
        elif action_result['status'] == 'failed':
            self.performance_metrics['failed_actions'] += 1
        else:
            self.performance_metrics['unnecessary'] += 1


    def construct_step_response(self, reward, info, done):
        observation = self.get_observation()
        observation['action_mask'] = self.compute_defender_action_mask()
        info['performance_metrics'] = self.performance_metrics
        network_availability = self.env.network_availability
        info['network_availability'] = network_availability

        return observation, reward, done, info


    def compute_defender_action_mask(self):
        #initialise a blank action mask
        action_mask = {
            'Reimage Node': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Block Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Allow Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Stop Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Start Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32)
        }

        #update the action mask based on the current state of the environment
        for numerical_id, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            #print(f"Node ID: {node_id}, Info: {node_info}")
            #if node is infected or has vulnerabilities, defender can reimage node
            if node_info.agent_installed or self.has_vulnerabilities(node_id):
                action_mask['Reimage Node'][numerical_id] = 1
                #print(f"Node {node_id} can be reimaged (Infected or Vulnerable)")
            #defender can block traffic and allow traffic on any node
            action_mask['Block Traffic'][numerical_id] = 1
            action_mask['Allow Traffic'][numerical_id] = 1
            #print(f"Node {node_id} can have traffic blocked/allowed")
            #if a service is running on the node, defender can stop service
            if any(service.running for service in node_info.services):
                action_mask['Stop Service'][numerical_id] = 1
                #print(f"Node {node_id} has running services that can be stopped")
            #if a service is stopped on the node, defender can start service
            if any(not service.running for service in node_info.services):
                action_mask['Start Service'][numerical_id] = 1
                #print(f"Node {node_id} has stopped services that can be started")
        return action_mask

    def update_last_attacker_reward(self, reward):
        self.last_attacker_reward = reward

    def default_or_preventive_action(self):
        random_node_id = random.choice(list(self.id_mapping.keys()))
        return self.create_action('Check Firewall', random_node_id)

    def defender_goal_reached(self):
        for _, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            if node_info.agent_installed:
                return False
        return True

    def reset(self):
        initial_observation = self.env.reset()
        return initial_observation

    def reimage_node(self, node_id, environment):
        #mark the node for re-imaging
        self.node_reimaging_progress[node_id] = self.REIMAGING_DURATION
        return self.defender_actions.reimage_node(node_id, environment)


    def get_node(self, node_id):
        #implement this method to access node information from the environment
        return self.env.get_node(node_id)

    def choose_action(self, state):
        """
        Choose an action based on the current state of the environment.
        """
        print("Choosing action based on the current state")
        #detect infected nodes
        infected_nodes = self.find_infected_nodes()
        #print('infected nodes',infected_nodes)
        #print(f"Infected nodes: {infected_nodes}")
        '''
        #If there are infected nodes, reimage the first one
        if infected_nodes:
            #node_id_to_reimage = infected_nodes[0]
            node_id_to_reimage = random.choice(infected_nodes)
            #print(f"Reimaging node: {node_id_to_reimage}")
            return self.create_action('Reimage Node', node_id_to_reimage)
        '''
        #List of preventive actions
        preventive_actions = ['Reimage Node','Check Firewall', 'Block Traffic', 'Allow Traffic', 'Stop Service', 'Start Service']
        #print("No infected nodes found, selecting a preventive action")
        #Randomly select a preventive action
        selected_preventive_action = random.choice(preventive_actions)
        #print(f"Selected preventive action: {selected_preventive_action}")
        #Randomly select a node for the preventive action
        node_id_for_preventive_action = self.select_node_for_preventive_action()
        #print(f"Node selected for preventive action: {node_id_for_preventive_action}")
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

    def sample_valid_action(self):
        """
        Randomly selects a valid action for the defender.
        """
        action_type = random.randint(0, 4)
        node_id = random.choice(list(self.id_mapping.keys()))
        parameter = random.randint(0, 9)
        action = {
            'action_type': action_type,
            'node_id': node_id,
            'parameter': parameter
        }
        while not self.validate_action(action):
            action['action_type'] = random.randint(0, 4)
            action['node_id'] = random.choice(list(self.id_mapping.keys()))
            action['parameter'] = random.randint(0, 9)

        return action

    def render(self):
        #implements rendering logic here
        print("Rendering the current state of the environment")

    def close(self):
        pass

class DefenderAgentActions:
    """Actions reserved for defender agents"""

    #number of steps it takes to completely reimage a node
    REIMAGING_DURATION = 15

    def __init__(self, environment: CyberBattleChain):
        #map nodes being reimaged to the remaining number of steps to completion
        self.node_reimaging_progress: Dict[model.NodeID, int] = dict()
        #last calculated availability of the network
        self.__network_availability: float = 1.0
        self._environment = environment

    @property
    def network_availability(self):
        return self.__network_availability

    def print_initial_node_states(self):
        print("Initial node states:")
        for node_id in self._environment.network.nodes:
            node_data = self._environment.get_node(node_id)
            print(f"Node {node_id}: {node_data}")

    def reimage_node(self, node_id: model.NodeID, environment: model.Environment):
        """Re-image a computer node"""
        node_info = environment.get_node(node_id)
        action_result = {}

        #check if the agent is installed - Precondition for reimaging
        if not node_info.agent_installed:
            return {"action": "reimage_node", "status": "unnecessary", "node_id": node_id, "reason": "Reimaging not required - no agent installed"}
        if node_info.status != MachineStatus.Imaging:
            node_info.agent_installed = False
            node_info.privilege_level = PrivilegeLevel.NoAccess
            node_info.status = MachineStatus.Imaging
            node_info.last_reimaging = datetime.datetime.now()
            action_result = {"action": "reimage_node", "status": "success", "node_id": node_id}
        else:
            action_result = {"action": "reimage_node", "status": "unnecessary", "node_id": node_id}

        return action_result


    def on_attacker_step_taken(self):
        """Function to be called each time a step is taken in the simulation"""
        for node_id in list(self.node_reimaging_progress.keys()):
            remaining_steps = self.node_reimaging_progress[node_id]
            if remaining_steps > 0:
                self.node_reimaging_progress[node_id] -= 1
            else:
                print(f"Machine re-imaging completed: {node_id}")
                node_data = self._environment.get_node(node_id)
                node_data.status = MachineStatus.Running
                self.node_reimaging_progress.pop(node_id)
        #calculate the network availability metric based on machines and services that are running
        total_node_weights = 0
        network_node_availability = 0
        for node_id, node_info in self._environment._cyberbattle_env.__environment.nodes():
            total_service_weights = 0
            running_service_weights = 0
            if isinstance(node_info, ChainNodeInfo):
                for service in node_info.services:
                    total_service_weights += service.sla_weight
                    running_service_weights += service.sla_weight * int(service.running)

                if node_info.status == MachineStatus.Running:
                    adjusted_node_availability = (1 + running_service_weights) / (
                        1 + total_service_weights
                    )
                else:
                    adjusted_node_availability = 0.0

                total_node_weights += node_info.sla_weight
                network_node_availability += (
                    adjusted_node_availability * node_info.sla_weight
                )

        self.__network_availability = network_node_availability / total_node_weights
        assert self.__network_availability <= 1.0 and self.__network_availability >= 0.0

    def override_firewall_rule(
            self,
            node_id: model.NodeID,
            port_name: model.PortName,
            incoming: bool,
            permission: model.RulePermission,
        ):
            node_data = self._environment.get_node(node_id)

            def add_or_patch_rule(rules) -> List[FirewallRule]:
                new_rules = []
                has_matching_rule = False
                for r in rules:
                    if r.port == port_name:
                        has_matching_rule = True
                        new_rules.append(FirewallRule(r.port, permission))
                    else:
                        new_rules.append(r)

                if not has_matching_rule:
                    new_rules.append(model.FirewallRule(port_name, permission))
                return new_rules

            if incoming:
                node_data.firewall.incoming = add_or_patch_rule(node_data.firewall.incoming)
            else:
                node_data.firewall.outgoing = add_or_patch_rule(node_data.firewall.outgoing)

    #blocks network traffic on specific port of node, if node is running, the traffic on the node can be blocked
    #failure is is machine not running
    def block_traffic(self, node_id: model.NodeID, port_name: model.PortName, incoming: bool):
        node_data = self._environment.get_node(node_id)
        if node_data.status == MachineStatus.Running:
            self.override_firewall_rule(node_id, port_name, incoming, model.RulePermission.BLOCK)
            action_result = {"action": "block_traffic", "status": "success", "node_id": node_id, "port_name": port_name}
        else:
            action_result = {"action": "block_traffic", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Machine not running"}
        return action_result

    #allows network traffic on specific port of node, if node is running, the traffic on the node is allowed
    #failure is is machine not running
    def allow_traffic(self, node_id: model.NodeID, port_name: model.PortName, incoming: bool):
        node_data = self._environment.get_node(node_id)
        if node_data.status == MachineStatus.Running:
            self.override_firewall_rule(node_id, port_name, incoming, model.RulePermission.ALLOW)
            action_result = {"action": "allow_traffic", "status": "success", "node_id": node_id, "port_name": port_name}
        else:
            action_result = {"action": "allow_traffic", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Machine not running"}
        return action_result

    #stops service on specific node, if node is running and the specified service is running, it can be stopped successfully
    #failure is is if machine is not running and service is not found
    def stop_service(self, node_id: model.NodeID, port_name: model.PortName):
      """ Stop a service on a given node """
      node_data = self._environment.get_node(node_id)
      action_result = {}
      if node_data.status == MachineStatus.Running:
          service_found = False
          for service in node_data.services:
              if service.name == port_name:
                  if service.running:
                      service.running = False
                      service_found = True
                      action_result = {"action": "stop_service", "status": "success", "node_id": node_id, "port_name": port_name}
                      break
                  else:
                      action_result = {"action": "stop_service", "status": "unnecessary", "node_id": node_id, "port_name": port_name, "reason": "Service already stopped"}
                      break
          if not service_found:
              action_result = {"action": "stop_service", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Service not found"}
      else:
          action_result = {"action": "stop_service", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Machine not running"}
      return action_result

    #starts service on specific node
    #success if specified service is not running and can be started
    #unneccesary = service already running
    #failure = machine not running and srvice not running
    def start_service(self, node_id: model.NodeID, port_name: model.PortName):
        """ Start a service on a given node """
        node_data = self._environment.get_node(node_id)
        action_result = {}
        if node_data.status == MachineStatus.Running:
            service_found = False
            for service in node_data.services:
                if service.name == port_name:
                    if not service.running:
                        service.running = True
                        service_found = True
                        action_result = {"action": "start_service", "status": "success", "node_id": node_id, "port_name": port_name}
                        break
                    else:
                        action_result = {"action": "start_service", "status": "unnecessary", "node_id": node_id, "port_name": port_name, "reason": "Service already running"}
                        break
            if not service_found:
                action_result = {"action": "start_service", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Service not found"}
        else:
            action_result = {"action": "start_service", "status": "failed", "node_id": node_id, "port_name": port_name, "reason": "Machine not running"}
        return action_result

class EpsilonGreedyLearner:
    def __init__(self, env, defender, epsilon=0.1):
        self.env = env
        self.defender = defender
        self.epsilon = epsilon

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return self.explore()
        else:
            return self.exploit(state)

    def explore(self):
        #randomly select an action from the defender's valid actions
        return self.defender.sample_valid_action()

    def exploit(self, state):
        return self.defender.sample_valid_action()

    def update_state(self, state, reward):
        pass

    def end_of_episode(self, episode_number):
        pass

class TrainedDefender:
    pass

class DefenderEpsilonGreedyLearner(EpsilonGreedyLearner):
    def __init__(self, env, defender, ep, gamma, learning_rate, epsilon, state_space_size, action_space_size):
        super().__init__(env, defender, epsilon)
        self.defender = defender
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.q_matrix = np.zeros((self.state_space_size, self.action_space_size))
        self.gamma = gamma


    def decide_action(self, wrapped_env, observation) -> Tuple[str, cyberbattle_env.Action, object]:
        if np.random.rand() < self.epsilon:
            return self.explore(wrapped_env)
        else:
            return self.exploit(wrapped_env, observation)

    def explore(self, wrapped_env) -> Tuple[str, cyberbattle_env.Action, object]:
        action = wrapped_env.sample_valid_action()
        action_metadata = None
        action_style = 'explore'
        return action_style, action, action_metadata

    def exploit(self, wrapped_env, observation) -> Tuple[str, Optional[cyberbattle_env.Action], object]:
        action = wrapped_env.sample_valid_action()
        action_metadata = None
        action_style = 'exploit'
        return action_style, action, action_metadata

    def on_step(self, wrapped_env, observation, reward, done, info, action_metadata):
        current_state = self.encode_state(observation)
        action = self.encode_action(action_metadata)
        next_state = self.encode_state(observation) if not done else None
        self.update_q_matrix(current_state, action, reward, next_state)

    def update_q_matrix(self, state, action, reward, next_state):
        max_future_q = 0 if next_state is None else np.max(self.q_matrix[next_state])
        current_q = self.q_matrix[state, action]
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.gamma * max_future_q)
        self.q_matrix[state, action] = new_q

    def parameters_as_string(self):
        return "Defender learner parameters: "

    def encode_state(self, observation):
        state_index = 0
        return state_index

    def encode_action(self, action_metadata):
        action_index = 0
        return action_index

    def new_episode(self):
        pass

    def end_of_episode(self, i_episode, t):
        pass

    def end_of_iteration(self, t, done):
        pass

    def all_parameters_as_string(self):
        return ''

    def loss_as_string(self):
        return ''

    def stateaction_as_string(self, action_metadata):
        return ''


def plot_exploration_exploitation(exploration_count, exploitation_count, episode_count):
    episodes = np.arange(1, episode_count + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, exploration_count, label='Exploration Actions', marker='o')
    plt.plot(episodes, exploitation_count, label='Exploitation Actions', marker='x')

    plt.title('Exploration vs Exploitation Actions Over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Action Count')
    plt.legend()
    plt.grid(True)
    plt.show()

def epsilon_greedy_defender_training(
    cyberbattle_gym_env: cyberbattle_env.CyberBattleEnv,
    environment_properties: w.EnvironmentBounds,
    learner: DefenderEpsilonGreedyLearner,
    title: str,
    episode_count: int,
    iteration_count: int,
    epsilon: float,
    epsilon_minimum=0.0,
    epsilon_multdecay: Optional[float] = None,
    epsilon_exponential_decay: Optional[int] = None,
    render=True,
    render_last_episode_rewards_to: Optional[str] = None,
    verbosity: Verbosity = Verbosity.Normal,
    plot_episodes_length=True
) -> TrainedLearner:
    """Epsilon greedy search for CyberBattle gym environments

    Parameters
    ==========

    - cyberbattle_gym_env -- the CyberBattle environment to train on

    - learner --- the policy learner/exploiter

    - episode_count -- Number of training episodes

    - iteration_count -- Maximum number of iterations in each episode

    - epsilon -- explore vs exploit
        - 0.0 to exploit the learnt policy only without exploration
        - 1.0 to explore purely randomly

    - epsilon_minimum -- epsilon decay clipped at this value.
    Setting this value too close to 0 may leed the search to get stuck.

    - epsilon_decay -- epsilon gets multiplied by this value after each episode

    - epsilon_exponential_decay - if set use exponential decay. The bigger the value
    is, the slower it takes to get from the initial `epsilon` to `epsilon_minimum`.

    - verbosity -- verbosity of the `print` logging

    - render -- render the environment interactively after each episode

    - render_last_episode_rewards_to -- render the environment to the specified file path
    with an index appended to it each time there is a positive reward
    for the last episode only

    - plot_episodes_length -- Plot the graph showing total number of steps by episode
    at th end of the search.

    Note on convergence
    ===================

    Setting 'minimum_espilon' to 0 with an exponential decay <1
    makes the learning converge quickly (loss function getting to 0),
    but that's just a forced convergence, however, since when
    epsilon approaches 0, only the q-values that were explored so
    far get updated and so only that subset of cells from
    the Q-matrix converges.

    """
    exploration_count = [] #new
    exploitation_count = [] #new
    #epsilon = epsilon
    epsilon_values = []  # To track epsilon values over episodes
    successful_actions_per_episode = [] #track successful actions
    failed_actions_per_episode = []
    unnecessary_actions_per_episode = []
    epsilon_values = []

    #action counters
    successful_reimaging_count = 0
    starting_services_count = 0
    allowing_traffic_count = 0
    blocking_traffic_count = 0
    stopping_services_count = 0
    total_actions_count = 0

    all_episodes_rewards = []
    all_episodes_availability = []


    print(f"###### {title}\n"
          f"Learning with: episode_count={episode_count},"
          f"iteration_count={iteration_count},"
          f"ϵ={epsilon},"
          f'ϵ_min={epsilon_minimum}, '
          + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '')
          + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') +
          f"{learner.parameters_as_string()}")

    initial_epsilon = epsilon
    successful_reimaging_count = 0
    starting_services_count = 0
    allowing_traffic_count = 0
    blocking_traffic_count = 0
    stopping_services_count = 0
    all_episodes_rewards = []
    all_episodes_availability = []
    defender_name = 'MyDefenderEnv'
    wrapped_env = DefenderWrapper(cyberbattle_gym_env,defender_name)
    steps_done = 0
    plot_title = f"{title} (epochs={episode_count}, ϵ={initial_epsilon}, ϵ_min={epsilon_minimum}," \
        + (f"ϵ_multdecay={epsilon_multdecay}," if epsilon_multdecay else '') \
        + (f"ϵ_expdecay={epsilon_exponential_decay}," if epsilon_exponential_decay else '') \
        + learner.parameters_as_string()
    plottraining = PlotTraining(title=plot_title, render_each_episode=render)

    render_file_index = 1

    for i_episode in range(1, episode_count + 1):
        exploration_actions = 0 #new
        exploitation_actions = 0 #new
        epsilon_values.append(epsilon)
        print(f"  ## Episode: {i_episode}/{episode_count} '{title}' "
              f"ϵ={epsilon:.4f}, "
              f"{learner.parameters_as_string()}")

        observation = wrapped_env.reset()
        total_reward = 0.0
        all_rewards = []
        all_availability = []
        successful_actions = 0
        failed_actions = 0
        unnecessary_actions = 0
        learner.new_episode()

        stats = Stats(exploit=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      explore=Outcomes(reward=Breakdown(local=0, remote=0, connect=0),
                                       noreward=Breakdown(local=0, remote=0, connect=0)),
                      exploit_deflected_to_explore=0
                      )

        episode_ended_at = None
        sys.stdout.flush()

        bar = progressbar.ProgressBar(
            widgets=[
                'Episode ',
                f'{i_episode}',
                '|Iteration ',
                progressbar.Counter(),
                '|',
                progressbar.Variable(name='reward', width=6, precision=10),
                '|',
                progressbar.Variable(name='last_reward_at', width=4),
                '|',
                progressbar.Timer(),
                progressbar.Bar()
            ],
            redirect_stdout=False)


        for t in bar(range(1, 1 + iteration_count)):

            if epsilon_exponential_decay:
                epsilon = epsilon_minimum + math.exp(-1. * steps_done /
                                                     epsilon_exponential_decay) * (initial_epsilon - epsilon_minimum)

            steps_done += 1

            x = np.random.rand()
            if x <= epsilon:
                action_style, gym_action, action_metadata = learner.explore(wrapped_env)
                exploration_actions += 1 #new
            else:
                action_style, gym_action, action_metadata = learner.exploit(wrapped_env, observation)
                exploitation_actions += 1 #new
                if not gym_action:
                    stats['exploit_deflected_to_explore'] += 1
                    _, gym_action, action_metadata = learner.explore(wrapped_env)

            logging.debug(f"gym_action={gym_action}, action_metadata={action_metadata}")
            observation, reward, done, info = wrapped_env.step(gym_action)

            # Print the simplified action taken
            action_type = gym_action.get('action_type', 'Unknown')
            if action_type == 0:  #Reimage Node
                action_type = "Reimage Node"
            elif action_type == 1:  #Block Traffic
                action_type = "Block Traffic"
            elif action_type == 2:  #Allow Traffic
                action_type = "Reimage Node"
            elif action_type == 3:  #Stop Service
                action_type = "Allow Traffic"
            elif action_type == 4:  #Start Service
                action_type = "Start Service"

            print('action type: ',action_type)
            node_id = gym_action.get('node_id', 'Unknown')
            parameter = gym_action.get('parameter', 'Unknown')
            print(f"Iteration {t}: Action Taken: Type={action_type}, NodeID={node_id}, Parameter={parameter}")

            action_status = info.get('status', '')

            if action_status == 'success':
                action_name = info.get('action', '')
                if action_name == 'reimage_node':
                    successful_reimaging_count += 1
                elif action_name == 'start_service':
                    starting_services_count += 1
                elif action_name == 'allow_traffic':
                    allowing_traffic_count += 1
                elif action_name == 'block_traffic':
                    blocking_traffic_count += 1
                elif action_name == 'stop_service':
                    stopping_services_count += 1

            if action_status == 'success':
                successful_actions += 1
            elif action_status == 'fail':
                failed_actions += 1
            else:
                unnecessary_actions += 1


            print('info',info)

            learner.on_step(wrapped_env, observation, reward, done, info, action_metadata)
            assert np.shape(reward) == ()

            all_rewards.append(reward)
            if 'network_availability' in info:
                all_availability.append(info['network_availability'])
            else:
                all_availability.append(0)
            total_reward += reward
            bar.update(t, reward=total_reward)
            if reward > 0:
                bar.update(t, last_reward_at=t)

            if verbosity == Verbosity.Verbose or (verbosity == Verbosity.Normal and reward > 0):
                sign = ['-', '+'][reward > 0]

                print(f"    {sign} t={t} {action_style} r={reward} cum_reward:{total_reward} "
                      f"a={action_metadata}-{gym_action} "
                      f"creds={len(observation['credential_cache_matrix'])} "
                      f" {learner.stateaction_as_string(action_metadata)}")

            if i_episode == episode_count \
                    and render_last_episode_rewards_to is not None \
                    and reward > 0:
                fig = cyberbattle_gym_env.render_as_fig()
                fig.write_image(f"{render_last_episode_rewards_to}-e{i_episode}-{render_file_index}.png")
                render_file_index += 1

            learner.end_of_iteration(t, done)
            '''
            if done:
                episode_ended_at = t
                bar.finish(dirty=True)
                break
            '''
        sys.stdout.flush()

        loss_string = learner.loss_as_string()
        if loss_string:
            loss_string = "loss={loss_string}"
        exploration_count.append(exploration_actions) #new
        exploitation_count.append(exploitation_actions) #new
        successful_actions_per_episode.append(successful_actions)
        failed_actions_per_episode.append(failed_actions)
        unnecessary_actions_per_episode.append(unnecessary_actions)
        if episode_ended_at:
            print(f"  Episode {i_episode} ended at t={episode_ended_at} {loss_string}")
        else:
            print(f"  Episode {i_episode} stopped at t={iteration_count} {loss_string}")

        print_stats(stats)
        print(f"  Episode {i_episode} breakdown:")
        print(f"    Successful Reimaging Count: {successful_reimaging_count}")
        print(f"    Starting Services Count: {starting_services_count}")
        print(f"    Allowing Traffic Count: {allowing_traffic_count}")
        print(f"    Blocking Traffic Count: {blocking_traffic_count}")
        print(f"    Stopping Services Count: {stopping_services_count}")

        all_episodes_rewards.append(all_rewards)
        all_episodes_availability.append(all_availability)

        length = episode_ended_at if episode_ended_at else iteration_count
        learner.end_of_episode(i_episode=i_episode, t=length)
        if plot_episodes_length:
            plottraining.episode_done(length)
        if render:
            wrapped_env.render()

        if epsilon_multdecay:
            epsilon = max(epsilon_minimum, epsilon * epsilon_multdecay)
        successful_actions_per_episode.append(0)

    wrapped_env.close()
    print("simulation ended")
    #1. Plotting Average Success Rate Over Time
    success_rate_per_episode = [
    successful / (successful + failed + unnecessary) if (successful + failed + unnecessary) > 0 else 0
    for successful, failed, unnecessary in zip(successful_actions_per_episode, failed_actions_per_episode, unnecessary_actions_per_episode)
]
         #Calculate the moving average for smoothing
    window_size = 3
    moving_avg = np.convolve(success_rate_per_episode, np.ones(window_size)/window_size, mode='valid')
    mean_success_rate = np.mean(success_rate_per_episode)
    plt.figure(figsize=(12, 6))
    #plt.style.use('seaborn-darkgrid')  # Use a visually appealing style
    plt.plot(success_rate_per_episode, marker='o', linestyle='-', color='skyblue', label='Success Rate')
    plt.plot(np.arange(window_size - 1, len(success_rate_per_episode)), moving_avg, color='darkblue', linestyle='-', linewidth=2, label='Moving Average')
    plt.axhline(y=mean_success_rate, color='red', linestyle='--', label='Mean Success Rate')
    plt.title('Average Success Rate Over Time', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.xticks(range(len(success_rate_per_episode)), fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    plt.show()

    # 2. Bar Chart of Defender Actions Overview
    total_actions = sum(successful_actions_per_episode) + sum(failed_actions_per_episode) + sum(unnecessary_actions_per_episode)
    action_counts = [total_actions, sum(successful_actions_per_episode), sum(failed_actions_per_episode), sum(unnecessary_actions_per_episode)]
    action_types = ['Total Actions', 'Successful Actions', 'Failed Actions', 'Unnecessary Actions']

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(action_types))
    plt.bar(y_pos, action_counts, align='center', alpha=0.7)
    plt.xticks(y_pos, action_types)
    plt.ylabel('Count')
    plt.title('Defender Actions Overview')
    if plot_episodes_length:
        plottraining.plot_end()
    plot_exploration_exploitation(exploration_count, exploitation_count, episode_count)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, episode_count + 1), epsilon_values, label='Epsilon value')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon value')
    plt.title('Epsilon Decay Over Episodes')
    plt.legend()
    plt.grid(True)
    plt.show()
    return TrainedLearner(
        all_episodes_rewards=all_episodes_rewards,
        all_episodes_availability=all_episodes_availability,
        learner=learner,
        trained_on=cyberbattle_gym_env.name,
        title=plot_title
    )


def random_argtop_percentile(array: np.ndarray, percentile: float):
    """Just like `argmax` but if there are multiple elements with the max
    return a random index to break ties instead of returning the first one."""
    top_percentile = np.percentile(array, percentile)
    indices = np.where(array >= top_percentile)[0]
    if len(indices) == 0:
        return random_argmax(array)
    elif indices.shape[0] > 1:
        max_index = int(np.random.choice(indices))
    else:
        max_index = int(indices)

    return top_percentile, max_index

#from standard qlearning algorithm
class QMatrix:
    """Q-Learning matrix for a given state and action space
        state_space  - Features defining the state space
        action_space - Features defining the action space
        qm           - Optional: initialization values for the Q matrix
    """
    #quality matrix
    qm: np.ndarray

    def __init__(self, name,
                 statespace: w.Feature,
                 actionspace: w.Feature,
                 qm: Optional[np.ndarray] = None):
        """Initialise the Q-matrix"""

        self.name = name
        self.statespace = statespace
        self.actionspace = actionspace
        self.statedim = statespace.flat_size()
        self.actiondim = actionspace.flat_size()
        self.qm = self.clear() if qm is None else qm

        #error calculated for the last update to the Q-matrix
        self.last_error = 0

    def shape(self):
        return (self.statedim, self.actiondim)

    def clear(self):
        """Re-initialise the Q-matrix to 0"""
        self.qm = np.zeros(shape=self.shape())
        #self.qm = np.random.rand(*self.shape()) / 100
        return self.qm

    def print(self):
        print(f"[{self.name}]\n"
              f"state: {self.statespace}\n"
              f"action: {self.actionspace}\n"
              f"shape = {self.shape()}")

    def update(self, current_state: int, action: int, next_state: int, reward, gamma, learning_rate):
        """Update the Q matrix after taking `action` in state 'current_State'
        and obtaining reward=R[current_state, action]"""

        maxq_atnext, max_index = random_argmax(self.qm[next_state, ])

        #bellman equation for Q-learning
        temporal_difference = reward + gamma * maxq_atnext - self.qm[current_state, action]
        self.qm[current_state, action] += learning_rate * temporal_difference

        #the loss is calculated using the squared difference between
        #target Q-Value and predicted Q-Value
        square_error = temporal_difference * temporal_difference
        self.last_error = square_error

        return self.qm[current_state, action]

    def exploit(self, features, percentile) -> Tuple[int, float]:
        """exploit: leverage the Q-matrix.
        Returns the expected Q value and the chosen action."""
        expected_q, action = random_argtop_percentile(self.qm[features, :], percentile)
        return int(action), float(expected_q)

class DefenderQMatrix(QMatrix):
    def __init__(self, statespace, actionspace, defender):
        super().__init__('Defender Q-Matrix', statespace, actionspace)
        self.defender = defender

    '''
    update the Q-matrix based on the temporal difference error. It calculates the temporal difference, updates the Q-value, and returns the updated Q-value.
    '''
    def update(self, current_state: int, action: int, next_state: int, reward, gamma, learning_rate):
        """Update the Q matrix for the defender agent"""
        maxq_atnext, max_index = random_argmax(self.qm[next_state, ])
        #temporal difference calculation
        temporal_difference = reward + gamma * maxq_atnext - self.qm[current_state, action]
        self.qm[current_state, action] += learning_rate * temporal_difference
        #error calculation as needed
        square_error = temporal_difference * temporal_difference
        self.last_error = square_error

        return self.qm[current_state, action]


    def exploit(self, state_index):
        #encode the current observation into a state index
        #state_index = self.encode_observation_to_state(observation)
        # Check if the state index is valid
        if 0 <= state_index < self.qm.shape[0]:
            # Find the action with the maximum Q-value for the current state
            action_index, q_value = max(enumerate(self.qm[state_index, :]), key=lambda x: x[1])
            return action_index, q_value
        else:
            # Handle invalid state index, perhaps by choosing a random action or default action
            print(f"State index {state_index} out of bounds. Choosing default action.")
            return 1, 1  # Example default action

    '''
    returns the maximum Q-value for a given state, handling terminal states by returning a default value.
    '''
    def get_max_q_value(self, state):
        """Return the maximum Q-value for a given state."""
        if state == -1:
            return 0

        if 0 <= state < self.qm.shape[0]:
            return np.max(self.qm[state, :])
        else:
            raise IndexError(f"State index out of bounds: {state}")

    '''
    returns the Q-value for a specific state-action pair, checking if the indices are within bounds.
    '''
    def get_q_value(self, state, action):
        """Return the Q-value for a specific state-action pair."""
        print(f"Accessing Q-matrix with state={state}, action={action}")
        if state >= 0 and state < self.qm.shape[0] and action >= 0 and action < self.qm.shape[1]:
            return self.qm[state, action]
        else:
            raise IndexError("State or action index out of bounds.")

    def encode_observation_to_state(self, observation): #weighting features in state_space
        if 'firewall_status' not in observation:
            print("firewall_status is missing from observation")
        if 'infected_nodes' not in observation:
            print("infected_nodes is missing from observation")
        if 'service_status' not in observation:
            print("service_status is missing from observation")
        #define weights
        weight_firewall = 0.5
        weight_infected = 2
        weight_service = 1
        #calculate weighted counts
        firewall_active = sum(observation['firewall_status']) * weight_firewall
        infected_nodes = sum(observation['infected_nodes']) * weight_infected
        services_active = sum(observation['service_status']) * weight_service
        #combine into a single state index with adjusted weights
        state = int(firewall_active * 10000 + infected_nodes * 100 + services_active)
        return state

    '''
    allows for manually updating a specific state-action pair's Q-value.
    '''
    def update_q_value(self, state, action, new_q_value):
        """Update the Q-value for a specific state-action pair."""
        if state >= 0 and state < self.qm.shape[0] and action >= 0 and action < self.qm.shape[1]:
            self.qm[state, action] = new_q_value
        else:
            raise IndexError("State or action index out of bounds.")


class DefenderQLearner(EpsilonGreedyLearner):
    def __init__(self, env, defender, ep, q_matrix: DefenderQMatrix, epsilon, gamma, learning_rate):
        super().__init__(env, defender, epsilon)
        self.defender = defender
        self.q_matrix = q_matrix
        self.epsilon = epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate
        #self.defender_actions = defender_actions
        self.SOME_THRESHOLD = 3

    def explore(self, wrapped_env) -> Tuple[str, Action, Optional[object]]:
        action = wrapped_env.sample_valid_action()
        print(action)
        action_style = "explore"
        action_metadata = None
        return action_style, action, action_metadata

    def convert_action_index_to_action(self, action_index, node_id, parameter):
        """
        Convert an action index to an action representation.

        Args:
            action_index (int): The index of the action to be converted.
            node_id (int): The ID of the node associated with the action.
            parameter (int): The parameter value for the action.

        Returns:
            dict: An action representation as a dictionary.
        """
        #define a mapping from action indices to action types
        action_mapping = {
            0: 'Reimage Node',
            1: 'Block Traffic',
            2: 'Allow Traffic',
            3: 'Stop Service',
            4: 'Start Service'
        }

        #if the action index is valid
        if action_index not in action_mapping:
            raise ValueError("Invalid action index")

        #action dictionary
        action_type = action_mapping[action_index]
        action = {
            'action_type': action_type,
            'node_id': node_id,
            'parameter': parameter
        }

        return action
    def encode_observation_to_state(self, observation):
        #define weights
        weight_firewall = 0.1  #adjusted weight
        weight_infected = 0.2  #adjusted weight
        weight_service = 0.1  #adjusted weight
        #normalise counts to range [0, 1] based on expected max values
        max_firewall = max(observation['firewall_status'])
        max_infected = len(observation['infected_nodes'])
        max_service = len(observation['service_status'])
        #calculate normalised and weighted counts
        firewall_active = (sum(observation['firewall_status']) / max_firewall) * weight_firewall
        infected_nodes = (sum(observation['infected_nodes']) / max_infected) * weight_infected
        services_active = (sum(observation['service_status']) / max_service) * weight_service
        #simplified state calculation
        state = int((firewall_active + infected_nodes + services_active) * 100) #scaling factor adjusted
        print(f"firewall_active: {firewall_active}, infected_nodes: {infected_nodes}, services_active: {services_active}, calculated state: {state}")
        #cap state at max_state_index
        max_state_index = 4999  # Example maximum state index
        state = min(state, max_state_index)
        return state

    def validate_action(self, action):
        if 'action_type' not in action or 'node_id' not in action or 'parameter' not in action:
            raise ValueError("Action format is incorrect")

    def exploit(self, state,observation) -> Tuple[str, Action, Optional[object]]:
        observation = self.defender.get_observation()
        state_index = self.encode_observation_to_state(observation)
        #if no infected nodes are detected, use choose_action to select a preventive action
        infected_nodes = observation.get('infected_nodes', [])
        if sum(infected_nodes) == 0:
            print("No infected nodes detected. Choosing a preventive action.")
            preventive_action = self.defender.choose_action(observation)
            action_style = "exploit (preventive)"
            return action_style, preventive_action, None
        else:
            #normal exploitation based on Q-matrix
            action_index, _ = self.q_matrix.exploit(state_index)
            node_id = self.extract_node_id_from_observation(observation)
            parameter = self.determine_parameter_based_on_state(observation)
            action = self.convert_action_index_to_action(action_index, node_id, parameter)
            print(action)
            action_style = "exploit"
            return action_style, action, None


    def extract_node_id_from_observation(self, observation):
        infected_nodes = observation.get('infected_nodes', [])
        print(infected_nodes)
        for index, is_infected in enumerate(infected_nodes):
            if is_infected == 1:
                return index
        return None

    def determine_parameter_based_on_state(self, observation):
        infected_nodes = observation.get('infected_nodes', [])
        number_of_infected_nodes = sum(infected_nodes)
        parameter = 1 if number_of_infected_nodes > self.SOME_THRESHOLD else 0
        return parameter




    def update_q_matrix(self, state, action, reward, next_state):
        print(f"Updating Q-matrix for state={state}, action={action}, next_state={next_state}")
        max_future_q = self.q_matrix.get_max_q_value(next_state)
        current_q = self.q_matrix.get_q_value(state, action)
        #calculate the new Q-value
        new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.gamma * max_future_q)
        #update the Q-matrix with the new Q-value
        self.q_matrix.update_q_value(state, action, new_q)

    def on_step(self, wrapped_env, action, reward, next_observation, done, info):
        current_observation = self.defender.get_observation()
        current_state = self.encode_observation_to_state(current_observation)
        next_state = self.encode_observation_to_state(next_observation) if not done else -1
        action_index = self.encode_action(action)
        # Update Q-matrix using encoded states
        self.update_q_matrix(current_state, action_index, reward, next_state)
    '''
    def on_step(self, wrapped_env, action, reward, next_observation, done, info):
        observation = self.defender.get_observation()
        current_state = self.q_matrix.statespace.encode(observation)
        next_state = self.q_matrix.statespace.encode(observation) if not done else -1
        action_index = self.encode_action(action)
        self.update_q_matrix(current_state, action_index, reward, next_state)
    '''

    def end_of_iteration(self, current_iteration: int, total_iterations: int):
        pass

    def end_of_episode(self, i_episode=None, t=None):
        """
        Modified end_of_episode method with additional parameters.
        """
        pass


    def parameters_as_string(self):
        return "Defender learner parameters: []"

    def encode_state(self, observation):
        state_index = 0
        return state_index

    def encode_action(self, action_metadata):
        action_index = 0
        return action_index

    def new_episode(self):
        pass

    def all_parameters_as_string(self):
        return ''

    def loss_as_string(self):
        return ''

    def stateaction_as_string(self, action_metadata):
        return ''

def print_stats(stats):
    """Print learning statistics"""
    def print_breakdown(stats, actiontype: str):
        def ratio(kind: str) -> str:
            x, y = stats[actiontype]['reward'][kind], stats[actiontype]['noreward'][kind]
            sum = x + y
            if sum == 0:
                return 'NaN'
            else:
                return f"{(x / sum):.2f}"

        def print_kind(kind: str):
            print(
                f"    {actiontype}-{kind}: {stats[actiontype]['reward'][kind]}/{stats[actiontype]['noreward'][kind]} "
                f"({ratio(kind)})")
        print_kind('local')
        print_kind('remote')
        print_kind('connect')

    print("  Breakdown [Reward/NoReward (Success rate)]")
    print_breakdown(stats, 'explore')
    print_breakdown(stats, 'exploit')
    print(f"  exploit deflected to exploration: {stats['exploit_deflected_to_explore']}")



class UntrainedDefenderWrapper:
    def __init__(self, environment, name):
        self.env = environment
        self.network = environment.network
        self.defender_actions = DefenderAgentActions(environment)
        self.id_mapping = self.create_id_mapping()
        self.action_space = self.define_action_space()
        #self.observation_space = self.define_observation_space()
        self.last_attacker_reward = 0
        self.name = name
        self.performance_metrics = {
            'actions_taken': 0,
            'successful_actions': 0,
            'failed_actions': 0,
            'unnecessary':0
        }

    def create_id_mapping(self):
        mapping = {}
        for i, node in enumerate(self.env.network.nodes):
            mapping[i] = node
        return mapping

    def define_action_space(self):
        return spaces.Dict({
            'action_type': spaces.Discrete(5),
            'node_id': spaces.Discrete(len(self.id_mapping)),
            'parameter': spaces.Discrete(10)
        })

    def define_observation_space(self):
        return spaces.Dict({
            'infected_nodes': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'firewall_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32),
            'service_status': spaces.Box(low=0, high=1, shape=(len(self.id_mapping),), dtype=np.int32)
        })

    def validate_action(self, action):
        action_type, numerical_node_id, _ = action.values()
        print(action_type,numerical_node_id,_)
        node_id = self.id_mapping.get(numerical_node_id)
        action1 = self.create_action(action_type, node_id)
        #print('new action1',action1)
        action_type1, numerical_node_id, _ = action.values()
        #print('action type1',action1['action_type'])
        #print('node id',node_id)
        #print('numerical_node_id',numerical_node_id)
        if node_id is None or action1['action_type'] not in [0, 1, 2, 3, 4]:
            #print(node_id)
            #print(action_type)
            return False
        return True

    def step(self, action=None):
        """Perform a step in the environment using a randomly chosen action."""
        print(f"[DEBUG] Received external action: {action} (this action will be ignored)")
        #generate a random action internally
        random_action = self.sample_random_action()
        action_type, numerical_node_id, parameter = random_action.values()
        node_id = self.id_mapping[numerical_node_id]
        print(f"[DEBUG] Processed Node ID: {node_id} (Numerical ID: {numerical_node_id}) with internally generated action")
        action_result = self.perform_action(action_type, node_id, parameter)
        if action_result is None:
            print("[ERROR] Action result is None")
            return self.construct_step_response(-10, {'error': 'Action result is None'}, False)
        reward = self.calculate_reward(action_result)
        done = self.defender_goal_reached()
        self.update_performance_metrics(action_result)
        response = self.construct_step_response(reward, action_result, done)
        print(f"[DEBUG] Step response: {response}")
        return response


    def perform_action(self, action_type, node_id, parameter):
        #get the current state of the node
        node_info = self.env.get_node(node_id)
        if action_type == 0:  #reimage Node
            return self.defender_actions.reimage_node(node_id, self.env)
        elif action_type == 1:  #block traffic
            return self.defender_actions.block_traffic(node_id, parameter, incoming=True)

        elif action_type == 2:  #allow traffic
            return self.defender_actions.allow_traffic(node_id, parameter, incoming=True)

        elif action_type == 3:  #stop service
            if any(service.name == parameter and service.running for service in node_info.services):
                return self.defender_actions.stop_service(node_id, parameter)
            else:
                return {'status': 'unnecessary', 'reason': 'Service not running or not found', 'action': 'stop_service'}

        elif action_type == 4:  #start service
            if any(service.name == parameter and not service.running for service in node_info.services):
                return self.defender_actions.start_service(node_id, parameter)
            else:
                return {'status': 'unnecessary', 'reason': 'Service already running or not found', 'action': 'start_service'}

        else:
            return {'status': 'error', 'reason': 'Unknown action type', 'action': 'unknown'}


    def calculate_reward(self, action_result):
      if action_result['status'] == 'success':
          if action_result['action'] == 'reimage_node':
              return 5000  #higher reward for re-imaging a compromised node
          else:
              return 4000  #standard reward for other successful actions
      elif action_result['status'] == 'failed':
          if action_result['action'] == 'reimage_node':
              #print('reimaged')
              return -4000
          else:
              #print('not reimaged')
              return -5000  #penalty for failed actions
      elif action_result['status'] == 'unnecessary':
          return -1000  #smaller penalty for unnecessary actions
      return 0

    def get_observation(self):
      #print('environment nodes',self.env.network.nodes)
      infected_nodes = [int(self.env.is_node_infected(node)) for node in self.env.network.nodes]
      #print('infected_nodes',infected_nodes)
      firewall_status = [int(self.env.get_firewall_status(node)) for node in self.env.network.nodes]
      service_status = [int(self.env.get_service_status(node)) for node in self.env.network.nodes]
      #print('firewall_status',firewall_status)
      #print('service_status',service_status)
      return {
          'infected_nodes': np.array(infected_nodes, dtype=np.int32),
          'firewall_status': np.array(firewall_status, dtype=np.int32),
          'service_status': np.array(service_status, dtype=np.int32)
      }

    def update_performance_metrics(self, action_result):
        self.performance_metrics['actions_taken'] += 1
        if action_result['status'] == 'success':
            self.performance_metrics['successful_actions'] += 1
        elif action_result['status'] == 'failed':
            self.performance_metrics['failed_actions'] += 1
        else:
            self.performance_metrics['unnecessary'] += 1

    def construct_step_response(self, reward, info, done):
        observation = self.get_observation()
        observation['action_mask'] = self.compute_defender_action_mask()
        info['performance_metrics'] = self.performance_metrics
        network_availability = self.env.network_availability
        info['network_availability'] = network_availability

        return observation, reward, done, info


    def compute_defender_action_mask(self):
        #initialise a blank action mask
        action_mask = {
            'Reimage Node': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Block Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Allow Traffic': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Stop Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32),
            'Start Service': numpy.zeros(len(self.id_mapping), dtype=numpy.int32)
        }

        #update the action mask based on the current state of the environment
        for numerical_id, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            #print(f"Node ID: {node_id}, Info: {node_info}")
            #if node is infected or has vulnerabilities, defender can reimage node
            if node_info.agent_installed or self.has_vulnerabilities(node_id):
                action_mask['Reimage Node'][numerical_id] = 1
                #print(f"Node {node_id} can be reimaged (Infected or Vulnerable)")
            #defender can block traffic and allow traffic on any node
            action_mask['Block Traffic'][numerical_id] = 1
            action_mask['Allow Traffic'][numerical_id] = 1
            #print(f"Node {node_id} can have traffic blocked/allowed"
            #if a service is running on the node, defender can stop service
            if any(service.running for service in node_info.services):
                action_mask['Stop Service'][numerical_id] = 1
                #print(f"Node {node_id} has running services that can be stopped")
            #if a service is stopped on the node, defender can start service
            if any(not service.running for service in node_info.services):
                action_mask['Start Service'][numerical_id] = 1
                #print(f"Node {node_id} has stopped services that can be started")

        return action_mask

    def update_last_attacker_reward(self, reward):
        self.last_attacker_reward = reward


    def default_or_preventive_action(self):
        random_node_id = random.choice(list(self.id_mapping.keys()))
        return self.create_action('Check Firewall', random_node_id)

    def defender_goal_reached(self):
        for _, node_id in self.id_mapping.items():
            node_info = self.env.network.nodes[node_id]['data']
            if node_info.agent_installed:
                return False
        return True

    def reset(self):
        initial_observation = self.env.reset()
        return initial_observation

    def reimage_node(self, node_id, environment):
        #mark the node for re-imaging
        self.node_reimaging_progress[node_id] = self.REIMAGING_DURATION
        return self.defender_actions.reimage_node(node_id, environment)


    def get_node(self, node_id):
        #access node information from the environment
        return self.env.get_node(node_id)

    def choose_action(self, state):
        """
        Choose an action based on the current state of the environment.
        """
        print("Choosing action based on the current state")
        #detect infected nodes
        infected_nodes = self.find_infected_nodes()
        #print('infected nodes',infected_nodes)
        #print(f"Infected nodes: {infected_nodes}")
        #List of preventive actions
        preventive_actions = ['Reimage Node','Block Traffic', 'Allow Traffic', 'Stop Service', 'Start Service']
        #print("No infected nodes found, selecting a preventive action")
        #Randomly select a preventive action
        selected_preventive_action = random.choice(preventive_actions)
        #print(f"Selected preventive action: {selected_preventive_action}")
        #Randomly select a node for the preventive action
        node_id_for_preventive_action = self.select_node_for_preventive_action()
        #print(f"Node selected for preventive action: {node_id_for_preventive_action}")
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

    def sample_random_action(self):
        """Override to sample only 'Reimage Node' actions."""
        action_type = 0  # 'Reimage Node'
        node_id = random.choice(list(self.id_mapping.keys()))
        parameter = 0

        return {
            'action_type': action_type,
            'node_id': node_id,
            'parameter': parameter
        }

    def render(self):
        print("Rendering the current state of the environment")

    def close(self):
        pass
