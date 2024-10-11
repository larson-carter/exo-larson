from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition
from itertools import permutations

class AdvancedStrategy(PartitioningStrategy):
    def __init__(self, mode='balanced'):
        self.mode = mode  # 'latency', 'throughput', or 'balanced'

    def partition(self, topology: Topology, model_memory_requirement: float = None) -> List[Partition]:
        nodes = list(topology.all_nodes())

        # Collect device capabilities
        device_info = {}
        for node_id, device_cap in nodes:
            device_info[node_id] = {
                'flops': device_cap.flops.fp32,
                'memory': device_cap.memory,
                'max_fraction': 1.0,
            }

        if model_memory_requirement:
            total_model_memory = model_memory_requirement
            for node_id in device_info:
                device_memory = device_info[node_id]['memory']
                device_info[node_id]['max_fraction'] = min(1.0, device_memory / total_model_memory)

        partitions = {node_id: 0.0 for node_id in device_info}
        total_layers = 1.0

        # User selection control flow
        if self.mode == 'throughput':
            self._optimize_throughput(device_info, partitions, total_layers)
        elif self.mode == 'latency':
            self._optimize_latency(device_info, partitions, total_layers)
        else:
            self._optimize_balanced(device_info, partitions, total_layers, topology)

        sorted_nodes = sorted(partitions.keys())
        start = 0.0
        final_partitions = []
        for node_id in sorted_nodes:
            assigned_fraction = partitions[node_id]
            end = start + assigned_fraction
            final_partitions.append(Partition(node_id, round(start, 5), round(end, 5)))
            start = end

        return final_partitions

    def _optimize_throughput(self, device_info, partitions, total_layers):
        # Assign layers proportional to FLOPS, respecting memory constraints
        total_flops = sum(info['flops'] for info in device_info.values())
        unassigned_layers = total_layers
        for node_id, info in device_info.items():
            initial_fraction = info['flops'] / total_flops * total_layers
            assigned_fraction = min(initial_fraction, info['max_fraction'])
            partitions[node_id] = assigned_fraction
            unassigned_layers -= assigned_fraction

        while unassigned_layers > 1e-6:
            available_devices = {node_id: info for node_id, info in device_info.items() if partitions[node_id] < info['max_fraction'] - 1e-6}

            # Test case for no more available devices
            if not available_devices:
                break

            total_available_flops = sum(info['flops'] for info in available_devices.values())
            for node_id, info in available_devices.items():
                available_fraction = info['max_fraction'] - partitions[node_id]
                fraction_share = (info['flops'] / total_available_flops) * unassigned_layers
                assignable_fraction = min(fraction_share, available_fraction)
                partitions[node_id] += assignable_fraction
                unassigned_layers -= assignable_fraction

        for node_id, fraction in partitions.items():
            partitions[node_id] = min(fraction, device_info[node_id]['max_fraction'])

    def _optimize_latency(self, device_info, partitions, total_layers):
        # Assign as many initial layers as possible to the fastest device
        fastest_node = max(device_info.items(), key=lambda x: x[1]['flops'])[0]
        assigned_fraction = min(total_layers, device_info[fastest_node]['max_fraction'])
        partitions[fastest_node] = assigned_fraction
        unassigned_layers = total_layers - assigned_fraction

        if unassigned_layers > 1e-6:
            # Order remaining devices by FLOPS
            remaining_devices = sorted(((node_id, info) for node_id, info in device_info.items() if node_id != fastest_node), key=lambda x: x[1]['flops'], reverse=True)
            for node_id, info in remaining_devices:
                available_fraction = info['max_fraction']
                assignable_fraction = min(available_fraction, unassigned_layers)
                partitions[node_id] = assignable_fraction
                unassigned_layers -= assignable_fraction
                if unassigned_layers <= 1e-6:
                    break

    def _optimize_balanced(self, device_info, partitions, total_layers, topology):
        # Step 1: Assign layers proportional to FLOPS, respecting memory constraints
        total_flops = sum(info['flops'] for info in device_info.values())
        for node_id, info in device_info.items():
            initial_fraction = info['flops'] / total_flops * total_layers
            assigned_fraction = min(initial_fraction, info['max_fraction'])
            partitions[node_id] = assigned_fraction

        device_ids = list(device_info.keys())
        min_latency_order = self._find_min_latency_order(device_ids, topology)

        partitions_ordered = {}
        for node_id in min_latency_order:
            partitions_ordered[node_id] = partitions[node_id]

        partitions.clear()
        partitions.update(partitions_ordered)
        total_assigned_fraction = sum(partitions.values())

        scaling_factors = []
        for node_id, fraction in partitions.items():
            max_scaling = device_info[node_id]['max_fraction'] / fraction if fraction > 0 else float('inf')
            scaling_factors.append(max_scaling)
        scaling_factor = min(total_layers / total_assigned_fraction, min(scaling_factors))

        for node_id in partitions:
            partitions[node_id] *= scaling_factor
            partitions[node_id] = min(partitions[node_id], device_info[node_id]['max_fraction'])

        total_assigned_fraction = sum(partitions.values())

        unassigned_layers = total_layers - total_assigned_fraction
        if unassigned_layers > 1e-6:
            while unassigned_layers > 1e-6:
                available_devices = {
                    node_id: info for node_id, info in device_info.items()
                    if partitions[node_id] < info['max_fraction'] - 1e-6
                }
                if not available_devices:
                    break

                total_available_flops = sum(info['flops'] for node_id, info in available_devices.items())
                for node_id, info in available_devices.items():
                    available_fraction = info['max_fraction'] - partitions[node_id]
                    fraction_share = (info['flops'] / total_available_flops) * unassigned_layers
                    assignable_fraction = min(fraction_share, available_fraction)
                    partitions[node_id] += assignable_fraction
                    unassigned_layers -= assignable_fraction

        for node_id, fraction in partitions.items():
            partitions[node_id] = min(fraction, device_info[node_id]['max_fraction'])

    def _find_min_latency_order(self, device_ids, topology):
        # Generate all permutations and select the one that avoids high-latency links
        best_order = None
        min_total_latency = float('inf')

        for perm in permutations(device_ids):
            has_high_latency = False
            total_latency = 0.0
            for i in range(len(perm) - 1):
                latency = topology.get_latency(perm[i], perm[i + 1])
                if latency > 50:  # Threshold for high latency
                    has_high_latency = True
                    break
                total_latency += latency
            if has_high_latency:
                continue
            if total_latency < min_total_latency:
                min_total_latency = total_latency
                best_order = perm

        if best_order:
            return list(best_order)
        else:
            return device_ids
