from typing import List, Tuple
import logging
from .partitioning_strategy import PartitioningStrategy, Partition
from .topology import Topology
from exo.topology.device_capabilities import DeviceCapabilities

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AdvancedPartitioningStrategy(PartitioningStrategy):
    def __init__(self, latency_weight: float = 0.4, memory_weight: float = 0.3, flops_weight: float = 0.3):
        self.latency_weight = latency_weight
        self.memory_weight = memory_weight
        self.flops_weight = flops_weight
        logger.info(f"Initializing AdvancedPartitioningStrategy with latency_weight={latency_weight}, "
                    f"memory_weight={memory_weight}, flops_weight={flops_weight}")

    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        n = len(nodes)
        
        node_scores = [
            (node_id, self._calculate_node_score(capabilities, topology))
            for node_id, capabilities in nodes
        ]
        
        logger.debug(f"Node scores: {node_scores}")
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        total_score = sum(score for _, score in node_scores)
        
        partitions = []
        start = 0
        
        for node_id, score in node_scores:
            end = round(start + (score / total_score), 5)
            partitions.append(Partition(node_id, start, end))
            start = end
        
        logger.debug(f"Initial partitions: {partitions}")
        
        optimized_partitions = self._optimize_latency(partitions, topology)
        
        logger.debug(f"Final partitions: {optimized_partitions}")
        
        return optimized_partitions

    def _calculate_node_score(self, capabilities: DeviceCapabilities, topology: Topology) -> float:
        max_memory = 1024 * 1024 * 1024 * 1024  # 1 TB
        max_flops = 1e15  # 1 PetaFLOPS
        normalized_memory = capabilities.memory / max_memory
        
        # Calculate total FLOPS
        total_flops = capabilities.flops.fp32 + capabilities.flops.fp16 + capabilities.flops.int8
        normalized_flops = total_flops / (3 * max_flops)  # Divide by 3 since we're summing three types of FLOPS
        
        # Calculate average latency to other nodes
        node_id = next(node for node, caps in topology.all_nodes() if caps == capabilities)
        other_nodes = [other_node for other_node, _ in topology.all_nodes() if other_node != node_id]
        
        if other_nodes:
            avg_latency = sum(topology.get_latency(node_id, other_node) for other_node in other_nodes) / len(other_nodes)
            normalized_latency = 1 - (avg_latency / 1.0)  # Assuming max latency is 1 second
        else:
            # If there are no other nodes, set normalized_latency to 1 (best case)
            normalized_latency = 1.0
        
        score = (self.memory_weight * normalized_memory + 
                 self.flops_weight * normalized_flops + 
                 self.latency_weight * normalized_latency)
        
        logger.debug(f"Node score: memory={capabilities.memory}, total_flops={total_flops}, "
                     f"avg_latency={avg_latency if 'avg_latency' in locals() else 'N/A'}, score={score}")
        
        return score

    def _optimize_latency(self, partitions: List[Partition], topology: Topology) -> List[Partition]:
        n = len(partitions)
        optimized = partitions.copy()
        
        # Calculate total latency for the current order
        current_latency = sum(topology.get_latency(optimized[i].node_id, optimized[(i+1)%n].node_id) for i in range(n))
        
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    new_order = optimized.copy()
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_latency = sum(topology.get_latency(new_order[k].node_id, new_order[(k+1)%n].node_id) for k in range(n))
                    
                    if new_latency < current_latency:
                        optimized = new_order
                        current_latency = new_latency
                        improved = True
                        logger.debug(f"Swapped partitions {i} and {j}, new latency: {current_latency}")
        
        # Adjust partition sizes based on the new order
        start = 0
        for partition in optimized:
            end = round(start + (partition.end - partition.start), 5)
            partition.start = start
            partition.end = end
            start = end
        
        return optimized