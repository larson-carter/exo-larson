from typing import List
from .partitioning_strategy import PartitioningStrategy
from .topology import Topology
from .partitioning_strategy import Partition

class AdvancedStrategy(PartitioningStrategy):
    def partition(self, topology: Topology, model_memory_requirement: float = None) -> List[Partition]:
        nodes = list(topology.all_nodes())

        # Collect total FLOPS across all nodes
        total_flops = sum(node[1].flops.fp32 for node in nodes)
        total_layers = 1.0  # Represented as the fraction from 0 to 1

        # Step 1: Compute initial layer fractions proportional to FLOPS
        initial_assignments = []
        for node_id, device_cap in nodes:
            flops = device_cap.flops.fp32
            initial_fraction = flops / total_flops
            initial_assignments.append((node_id, device_cap, initial_fraction))

        # Step 2: Compute maximum fraction per node based on memory constraints
        if model_memory_requirement:
            max_memory_fraction_per_node = []
            for node_id, device_cap in nodes:
                max_fraction = min(1.0, device_cap.memory / model_memory_requirement)
                max_memory_fraction_per_node.append((node_id, device_cap, max_fraction))

            # Adjust initial assignments to not exceed memory-based maximums
            assigned_fractions = []
            for (node_id, device_cap, initial_fraction), (_, _, max_fraction) in zip(initial_assignments, max_memory_fraction_per_node):
                assigned_fraction = min(initial_fraction, max_fraction)
                assigned_fractions.append([node_id, device_cap, assigned_fraction, max_fraction])

            total_assigned_fraction = sum(assigned_fraction for (_, _, assigned_fraction, _) in assigned_fractions)

            # Redistribute remaining layers
            while total_assigned_fraction < total_layers:
                remaining_fraction = total_layers - total_assigned_fraction

                # Find devices with available capacity
                available_devices = []
                for idx, (node_id, device_cap, assigned_fraction, max_fraction) in enumerate(assigned_fractions):
                    available_fraction = max_fraction - assigned_fraction
                    if available_fraction > 1e-6:
                        available_devices.append((idx, node_id, device_cap, assigned_fraction, available_fraction, device_cap.flops.fp32))

                if not available_devices:
                    # No devices can take more layers
                    break

                total_available_flops = sum(flops for (_, _, _, _, _, flops) in available_devices)

                # Distribute remaining_fraction to available devices proportionally to their FLOPS
                for idx, node_id, device_cap, assigned_fraction, available_fraction, flops in available_devices:
                    fraction_share = (flops / total_available_flops) * remaining_fraction
                    fraction_share = min(fraction_share, available_fraction)
                    assigned_fractions[idx][2] += fraction_share  # Update assigned_fraction

                total_assigned_fraction = sum(assigned_fraction for (_, _, assigned_fraction, _) in assigned_fractions)

        else:
            # No memory constraints; use initial assignments
            assigned_fractions = [[node_id, device_cap, assigned_fraction, 1.0] for (node_id, device_cap, assigned_fraction) in initial_assignments]

        # Final assigned fractions without normalization to avoid exceeding max fractions
        total_assigned_fraction = sum(assigned_fraction for (_, _, assigned_fraction, _) in assigned_fractions)

        # Normalize only if total is less than 1.0 due to rounding errors
        if abs(total_assigned_fraction - total_layers) > 1e-6:
            assigned_fractions = [
                (node_id, device_cap, assigned_fraction / total_assigned_fraction)
                for (node_id, device_cap, assigned_fraction, _) in assigned_fractions
            ]
        else:
            assigned_fractions = [
                (node_id, device_cap, assigned_fraction)
                for (node_id, device_cap, assigned_fraction, _) in assigned_fractions
            ]

        # Step 4: Create partitions
        partitions = []
        start = 0.0
        for node_id, device_cap, assigned_fraction in assigned_fractions:
            end = start + assigned_fraction
            partitions.append(Partition(node_id, round(start, 5), round(end, 5)))
            start = end

        return partitions
