import unittest
from .topology import Topology
from .device_capabilities import DeviceCapabilities, DeviceFlops
from .advanced_strategy import AdvancedStrategy
from .ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy


class TestStrategyComparison(unittest.TestCase):

    def setUp(self):

        self.topology = Topology()
        devices = [
            ("node_1", "Device_A", "Chip_A", 8000, DeviceFlops(fp32=10, fp16=20, int8=40)),
            ("node_2", "Device_B", "Chip_B", 16000, DeviceFlops(fp32=20, fp16=40, int8=80)),
            ("node_3", "Device_C", "Chip_C", 12000, DeviceFlops(fp32=15, fp16=30, int8=60)),
        ]
        for node_id, model, chip, memory, flops in devices:
            device_cap = DeviceCapabilities(model=model, chip=chip, memory=memory, flops=flops)
            self.topology.update_node(node_id, device_cap)

        # Establish connections (e.g., a triangle loop)
        self.topology.add_edge("node_1", "node_2")
        self.topology.add_edge("node_2", "node_3")
        self.topology.add_edge("node_3", "node_1")

    def test_compare_strategies(self):

        advanced_strategy = AdvancedStrategy()
        ring_strategy = RingMemoryWeightedPartitioningStrategy()

        advanced_partitions = advanced_strategy.partition(self.topology)
        ring_partitions = ring_strategy.partition(self.topology)

        advanced_total_fraction = sum(p.end - p.start for p in advanced_partitions)
        ring_total_fraction = sum(p.end - p.start for p in ring_partitions)

        self.assertAlmostEqual(advanced_total_fraction, 1.0, places=4)
        self.assertAlmostEqual(ring_total_fraction, 1.0, places=4)

        self.assertEqual(len(advanced_partitions), len(ring_partitions))

        for adv_partition, ring_partition in zip(advanced_partitions, ring_partitions):
            adv_size = adv_partition.end - adv_partition.start
            ring_size = ring_partition.end - ring_partition.start
            self.assertAlmostEqual(adv_size, ring_size, delta=0.25)

        print("\nAdvancedStrategy Partitions:")
        for partition in advanced_partitions:
            print(f"Node {partition.node_id}: start = {partition.start}, end = {partition.end}, size = {partition.end - partition.start:.6f}")

        print("\nRingMemoryWeightedPartitioningStrategy Partitions:")
        for partition in ring_partitions:
            print(f"Node {partition.node_id}: start = {partition.start}, end = {partition.end}, size = {partition.end - partition.start:.6f}")

if __name__ == "__main__":
    unittest.main()
