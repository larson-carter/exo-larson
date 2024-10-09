import unittest
import logging
from exo.topology.topology import Topology
from exo.topology.advanced_partitioning_strategy import AdvancedPartitioningStrategy
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestPartitioningStrategies(unittest.TestCase):
    def setUp(self):
        self.topology = Topology()
        self.topology.update_node(
            "node1",
            DeviceCapabilities(model="High Memory, Low FLOPS", chip="test1", memory=256*1024*1024*1024, flops=DeviceFlops(fp32=1e12, fp16=2e12, int8=4e12)),
        )
        self.topology.update_node(
            "node2",
            DeviceCapabilities(model="Low Memory, High FLOPS", chip="test2", memory=32*1024*1024*1024, flops=DeviceFlops(fp32=5e12, fp16=1e13, int8=2e13)),
        )
        self.topology.update_node(
            "node3",
            DeviceCapabilities(model="Balanced", chip="test3", memory=128*1024*1024*1024, flops=DeviceFlops(fp32=3e12, fp16=6e12, int8=1.2e13)),
        )
        self.topology.add_latency("node1", "node2", 0.2)
        self.topology.add_latency("node2", "node1", 0.25)
        self.topology.add_latency("node2", "node3", 0.05)
        self.topology.add_latency("node3", "node2", 0.08)
        self.topology.add_latency("node3", "node1", 0.1)
        self.topology.add_latency("node1", "node3", 0.15)

    def test_ring_memory_weighted_strategy(self):
        logger.info("Testing RingMemoryWeightedPartitioningStrategy")
        strategy = RingMemoryWeightedPartitioningStrategy()
        partitions = strategy.partition(self.topology)
        logger.info(f"RingMemoryWeightedPartitioningStrategy partitions: {partitions}")
        self.assertEqual(len(partitions), 3)
        self.assertAlmostEqual(sum(p.end - p.start for p in partitions), 1.0)

    def test_advanced_strategy(self):
        logger.info("Testing AdvancedPartitioningStrategy")
        strategy = AdvancedPartitioningStrategy(latency_weight=0.4, memory_weight=0.3, flops_weight=0.3)
        partitions = strategy.partition(self.topology)
        logger.info(f"AdvancedPartitioningStrategy partitions: {partitions}")
        self.assertEqual(len(partitions), 3)
        self.assertAlmostEqual(sum(p.end - p.start for p in partitions), 1.0)

    def test_strategy_comparison(self):
        logger.info("Comparing RingMemoryWeightedPartitioningStrategy and AdvancedPartitioningStrategy")
        ring_strategy = RingMemoryWeightedPartitioningStrategy()
        advanced_strategy = AdvancedPartitioningStrategy(latency_weight=0.4, memory_weight=0.3, flops_weight=0.3)

        ring_partitions = ring_strategy.partition(self.topology)
        advanced_partitions = advanced_strategy.partition(self.topology)

        logger.info(f"RingMemoryWeightedPartitioningStrategy partitions: {ring_partitions}")
        logger.info(f"AdvancedPartitioningStrategy partitions: {advanced_partitions}")

        self.assertNotEqual(ring_partitions, advanced_partitions, "Partitions should be different for the two strategies")

        # Compare partition sizes
        ring_sizes = [p.end - p.start for p in ring_partitions]
        advanced_sizes = [p.end - p.start for p in advanced_partitions]
        logger.info(f"RingMemoryWeightedPartitioningStrategy partition sizes: {ring_sizes}")
        logger.info(f"AdvancedPartitioningStrategy partition sizes: {advanced_sizes}")

        # Check if the order of nodes is different
        ring_order = [p.node_id for p in ring_partitions]
        advanced_order = [p.node_id for p in advanced_partitions]
        logger.info(f"RingMemoryWeightedPartitioningStrategy node order: {ring_order}")
        logger.info(f"AdvancedPartitioningStrategy node order: {advanced_order}")

        self.assertNotEqual(ring_order, advanced_order, "Node order should be different for the two strategies")

if __name__ == '__main__':
    unittest.main()