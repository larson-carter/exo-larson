from typing import List
from .partitioning_strategy import PartitioningStrategy, Partition
from .topology import Topology
import numpy as np
from exo.topology.device_capabilities import DeviceCapabilities

class AdvancedPartitioningStrategy(PartitioningStrategy):
    def __init__(self, latency_weight: float = 0.3, throughput_weight: float = 0.7):
        self.latency_weight = latency_weight
        self.throughput_weight = throughput_weight

    def partition(self, topology: Topology) -> List[Partition]:
        nodes = list(topology.all_nodes())
        n = len(nodes)
        
        node_scores = [
            (node_id, self._calculate_node_score(capabilities))
            for node_id, capabilities in nodes
        ]
        
        node_scores.sort(key=lambda x: x[1], reverse=True)
        
        total_score = sum(score for _, score in node_scores)
        
        partitions = []
        start = 0
        
        for node_id, score in node_scores:
            end = round(start + (score / total_score), 5)
            partitions.append(Partition(node_id, start, end))
            start = end
        
        self._optimize_latency(partitions, topology)
        
        return partitions

    def _calculate_node_score(self, capabilities: DeviceCapabilities) -> float:
        max_memory = 1024 * 1024 * 1024 * 1024
        max_flops = 1e15
        normalized_memory = capabilities.memory / max_memory
        normalized_flops = capabilities.flops.fp32 / max_flops 
        
        return (self.throughput_weight * (0.5 * normalized_memory + 0.5 * normalized_flops) + self.latency_weight * normalized_flops)

    def _optimize_latency(self, partitions: List[Partition], topology: Topology):
        n = len(partitions)
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_swap(partitions[i], partitions[j], topology):
                    partitions[i], partitions[j] = partitions[j], partitions[i]

    def _should_swap(self, p1: Partition, p2: Partition, topology: Topology) -> bool:
        latency_before = topology.get_latency(p1.node_id, p2.node_id)
        latency_after = topology.get_latency(p2.node_id, p1.node_id)
        return latency_after < latency_before