import unittest
from exo.topology.topology import Topology
from exo.topology.advanced_partitioning_strategy import AdvancedPartitioningStrategy
from exo.topology.device_capabilities import DeviceCapabilities, DeviceFlops
from exo.topology.partitioning_strategy import Partition

class TestAdvancedPartitioningStrategy(unittest.TestCase):
    def test_partition_triangle(self):
        # triangle
        # node1 -> node2 -> node3 -> node1
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(model="test1", chip="test1", memory=3000, flops=DeviceFlops(fp32=1e12, fp16=2e12, int8=4e12)),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(model="test2", chip="test2", memory=1000, flops=DeviceFlops(fp32=5e11, fp16=1e12, int8=2e12)),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(model="test3", chip="test3", memory=6000, flops=DeviceFlops(fp32=2e11, fp16=4e11, int8=8e11)),
        )
        topology.add_edge("node1", "node2")
        topology.add_edge("node2", "node3")
        topology.add_edge("node3", "node1")
        topology.add_latency("node1", "node2", 0.1)
        topology.add_latency("node2", "node3", 0.2)
        topology.add_latency("node3", "node1", 0.15)

        strategy = AdvancedPartitioningStrategy(latency_weight=0.4, memory_weight=0.3, flops_weight=0.3)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 3)
        
        # Check that all nodes are represented
        node_ids = set(p.node_id for p in partitions)
        self.assertEqual(node_ids, {"node1", "node2", "node3"})
        
        # Check that partitions cover the entire range from 0 to 1
        self.assertAlmostEqual(partitions[0].start, 0.0)
        self.assertAlmostEqual(partitions[-1].end, 1.0)
        
        # Check that partitions are contiguous
        for i in range(1, len(partitions)):
            self.assertAlmostEqual(partitions[i-1].end, partitions[i].start)

    def test_partition_rounding(self):
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test1",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=1e12, fp16=2e12, int8=4e12),
            ),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(
                model="Mac Studio",
                chip="test2",
                memory=192*1024*1024*1024,
                flops=DeviceFlops(fp32=2e12, fp16=4e12, int8=8e12),
            ),
        )
        topology.update_node(
            "node3",
            DeviceCapabilities(
                model="MacBook Pro",
                chip="test3",
                memory=128*1024*1024*1024,
                flops=DeviceFlops(fp32=1e12, fp16=2e12, int8=4e12),
            ),
        )
        topology.add_latency("node1", "node2", 0.05)
        topology.add_latency("node2", "node3", 0.05)
        topology.add_latency("node3", "node1", 0.05)

        strategy = AdvancedPartitioningStrategy(latency_weight=0.4, memory_weight=0.3, flops_weight=0.3)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 3)
        # Check that the partitions sum to 1 and are rounded to 5 decimal places
        self.assertAlmostEqual(sum(p.end - p.start for p in partitions), 1.0, places=5)
        for p in partitions:
            self.assertEqual(round(p.start, 5), p.start)
            self.assertEqual(round(p.end, 5), p.end)

    def test_extreme_cases(self):
        topology = Topology()
        topology.update_node(
            "node1",
            DeviceCapabilities(model="Supercomputer", chip="test1", memory=1024*1024*1024*1024, flops=DeviceFlops(fp32=1e15, fp16=2e15, int8=4e15)),
        )
        topology.update_node(
            "node2",
            DeviceCapabilities(model="Weak node", chip="test2", memory=1*1024*1024*1024, flops=DeviceFlops(fp32=1e9, fp16=2e9, int8=4e9)),
        )
        topology.add_latency("node1", "node2", 0.1)

        strategy = AdvancedPartitioningStrategy(latency_weight=0.4, memory_weight=0.3, flops_weight=0.3)
        partitions = strategy.partition(topology)

        self.assertEqual(len(partitions), 2)
        self.assertGreater(partitions[0].end - partitions[0].start, 0.9)
        self.assertLess(partitions[1].end - partitions[1].start, 0.1)
        self.assertEqual(partitions[0].node_id, "node1")
        self.assertEqual(partitions[1].node_id, "node2")

if __name__ == '__main__':
    unittest.main()