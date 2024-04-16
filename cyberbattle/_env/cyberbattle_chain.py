# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CyberBattle environment based on a simple chain network structure"""

from ..samples.chainpattern import chainpattern
from . import shared_cyberbattle_env
#from . import cyberbattle_env
import networkx as nx
from cyberbattle.simulation.model import PortName, PrivilegeLevel, NodeInfo, NodeID, Environment
from ..simulation import commandcontrol, model, actions


class CyberBattleChain(shared_cyberbattle_env.CyberBattleEnvTest):
    """CyberBattle environment based on a simple chain network structure"""

    def __init__(self, size, **kwargs):
        self.size = size
        initial_environment = chainpattern.new_environment(size)
        print("Generated nodes:")
        print(str(initial_environment.network.nodes(data=True)))
        super().__init__(
            initial_environment=initial_environment,
            **kwargs)
        self.network = initial_environment.network.copy()
        print("Network initialized with the following nodes:")
        for node_id in self.network.nodes:
            print(node_id)

    def get_node(self, node_id: NodeID) -> NodeInfo:
        """Retrieve info for the node with the specified ID, ensuring correct type."""
        node_id_str = str(node_id) if not isinstance(node_id, str) else node_id

        if node_id_str not in self.network.nodes:
            raise ValueError(f"Node ID {node_id_str} does not exist in the network.")

        node_info: NodeInfo = self.network.nodes[node_id_str]['data']
        return node_info

    def print_all_node_ids(self):
        """Print all node IDs in the network."""
        print("All node IDs in the network:")
        for node_id in self.network.nodes:
            print(node_id)

    def add_node_to_network(self, node_id, node_data):
        """Add a new node to the network."""
        if node_id in self.network.nodes:
            raise ValueError(f"Node ID {node_id} already exists in the network.")

        self.network.add_node(node_id, data=node_data)

    @ property
    def name(self) -> str:
        return f"CyberBattleChain-{self.size}"
