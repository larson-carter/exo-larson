import unittest
from .topology import Topology
from .device_capabilities import DeviceCapabilities, DeviceFlops
from .partitioning_strategy import Partition
from .advanced_strategy import AdvancedStrategy

class TestAdvancedStrategy(unittest.TestCase):

    # 4 Identical Devices
    def test_uniform_devices(self):
        topology = Topology()
        for i in range(4):
            device_cap = DeviceCapabilities(model=f"Device_{i}", chip="TestChip", memory=8000, flops=DeviceFlops(fp32=10, fp16=20, int8=40))
            topology.update_node(f"node_{i}", device_cap)

        strategy = AdvancedStrategy()
        partitions = strategy.partition(topology)

        for partition in partitions:
            self.assertAlmostEqual(partition.end - partition.start, 0.25, places=4)

    # Memory Constraints - Not really needed
    def test_memory_constraints(self):
        topology = Topology()

        # Device A: High FLOPS but low memory
        device_a = DeviceCapabilities(model="Device_A", chip="FastChip",memory=4000, flops=DeviceFlops(fp32=20, fp16=40, int8=80))

        # Device B: Low FLOPS but high memory
        device_b = DeviceCapabilities(model="Device_B", chip="SlowChip", memory=16000, flops=DeviceFlops(fp32=5, fp16=10, int8=20))

        topology.update_node("node_a", device_a)
        topology.update_node("node_b", device_b)

        strategy = AdvancedStrategy()
        model_memory_requirement = 10000
        partitions = strategy.partition(topology, model_memory_requirement=model_memory_requirement)

        partition_a = next(p for p in partitions if p.node_id == "node_a")
        max_fraction_a = min(1.0, device_a.memory / model_memory_requirement)
        self.assertLessEqual(partition_a.end - partition_a.start, max_fraction_a + 1e-5)

        total_layers = sum(p.end - p.start for p in partitions)
        self.assertAlmostEqual(total_layers, 1.0, places=4)

    # Processing Time Balance with Different FLOPS
    def test_processing_time_balance(self):
        topology = Topology()
        devices = [("node_1", 10, 8000), ("node_2", 20, 8000), ("node_3", 30, 8000),]

        for node_id, flops, memory in devices:
            device_cap = DeviceCapabilities(model=f"Device_{node_id}", chip="VariedChip", memory=memory, flops=DeviceFlops(fp32=flops, fp16=flops * 2, int8=flops * 4))
            topology.update_node(node_id, device_cap)

        strategy = AdvancedStrategy()
        partitions = strategy.partition(topology)

        # Calculate processing times T_i = L_i / f_i
        processing_times = []
        for partition in partitions:
            node_id = partition.node_id
            device_cap = topology.get_node(node_id)
            flops = device_cap.flops.fp32
            L_i = partition.end - partition.start
            T_i = L_i / flops
            processing_times.append(T_i)

        # 5% difference for calculator deviation
        avg_time = sum(processing_times) / len(processing_times)
        for T_i in processing_times:
            self.assertAlmostEqual(T_i, avg_time, delta=0.05 * avg_time)

        # Print for visualization
        for idx, T_i in enumerate(processing_times):
            print(f"Node {idx + 1}: T_i = {T_i:.6f}, L_i = {partitions[idx].end - partitions[idx].start:.6f}")

    # Test Partition Rounding
    def test_partition_rounding(self):
        topology = Topology()
        topology.update_node("node1", DeviceCapabilities(model="MacBook Pro", chip="test1", memory=128000, flops=DeviceFlops(fp32=10, fp16=0, int8=0),),)
        topology.update_node("node2", DeviceCapabilities(model="Mac Studio", chip="test2", memory=192000, flops=DeviceFlops(fp32=20, fp16=0, int8=0),),)
        topology.update_node("node3", DeviceCapabilities(model="MacBook Pro", chip="test3", memory=128000, flops=DeviceFlops(fp32=10, fp16=0, int8=0),),)

        strategy = AdvancedStrategy()
        model_memory_requirement = 400000
        partitions = strategy.partition(topology, model_memory_requirement=model_memory_requirement)

        self.assertEqual(len(partitions), 3)

        expected_partitions = [Partition("node1", 0.0, 0.26), Partition("node2", 0.26, 0.74), Partition("node3", 0.74, 1.0),]

        for expected_partition in expected_partitions:
            actual_partition = next(p for p in partitions if p.node_id == expected_partition.node_id)
            self.assertAlmostEqual(actual_partition.start, expected_partition.start, places=2)
            self.assertAlmostEqual(actual_partition.end, expected_partition.end, places=2)

        # Ensure that the partitions cover the whole range [0, 1]
        total_fraction = sum(p.end - p.start for p in partitions)
        self.assertAlmostEqual(total_fraction, 1.0, places=4)

        # Print the partitions for visual
        for partition in partitions:
            print(f"Partition for {partition.node_id}: start = {partition.start}, end = {partition.end}")

    # Latency-Aware Partitioning
    def test_latency_aware_partitioning(self):
        topology = Topology()

        devices = [("node_1", DeviceCapabilities("Device_A", "Chip_A", 8000, DeviceFlops(fp32=10, fp16=20, int8=40))), ("node_2", DeviceCapabilities("Device_B", "Chip_B", 8000, DeviceFlops(fp32=10, fp16=20, int8=40))),("node_3", DeviceCapabilities("Device_C", "Chip_C", 8000, DeviceFlops(fp32=10, fp16=20, int8=40))),]
        for node_id, cap in devices:
            topology.update_node(node_id, cap)

        # Set inter-node latencies (in milliseconds)
        topology.set_latency("node_1", "node_2", 5)
        topology.set_latency("node_2", "node_3", 100)
        topology.set_latency("node_1", "node_3", 5)

        strategy = AdvancedStrategy(mode='balanced')
        partitions = strategy.partition(topology)

        expected_order = sorted([p.node_id for p in partitions])
        actual_order = [p.node_id for p in partitions]
        self.assertEqual(actual_order, expected_order, "Partitions are not assigned in sorted node ID order")

        total_layers = sum(p.end - p.start for p in partitions)
        self.assertAlmostEqual(total_layers, 1.0, places=4)

        # Print partitions for visual
        for partition in partitions:
            print(f"Partition for {partition.node_id}: start = {partition.start}, end = {partition.end}")

    # Latency Optimization Mode
    def test_latency_optimization(self):
        topology = Topology()

        devices = [("node_fast", DeviceCapabilities("FastDevice", "FastChip", 8000, DeviceFlops(fp32=30, fp16=60, int8=120))), ("node_slow", DeviceCapabilities("SlowDevice", "SlowChip", 8000, DeviceFlops(fp32=10, fp16=20, int8=40))),]

        for node_id, cap in devices:
            topology.update_node(node_id, cap)

        strategy = AdvancedStrategy(mode='latency')
        partitions = strategy.partition(topology)

        partition_fast = next(p for p in partitions if p.node_id == "node_fast")
        partition_slow = next(p for p in partitions if p.node_id == "node_slow")

        # The fast node should have been assigned all layers if possible
        self.assertAlmostEqual(partition_fast.end - partition_fast.start, 1.0, places=4)
        self.assertAlmostEqual(partition_slow.end - partition_slow.start, 0.0, places=4)

        # Print partitions for visual
        for partition in partitions:
            print(f"Partition for {partition.node_id}: start = {partition.start}, end = {partition.end}")

    # Throughput Optimization Mode
    def test_throughput_optimization(self):
        topology = Topology()

        devices = [("node_1", DeviceCapabilities("Device_A", "Chip_A", 8000, DeviceFlops(fp32=10, fp16=20, int8=40))), ("node_2", DeviceCapabilities("Device_B", "Chip_B", 8000, DeviceFlops(fp32=20, fp16=40, int8=80))), ("node_3", DeviceCapabilities("Device_C", "Chip_C", 8000, DeviceFlops(fp32=30, fp16=60, int8=120))),]
        for node_id, cap in devices:
            topology.update_node(node_id, cap)

        strategy = AdvancedStrategy(mode='throughput')
        partitions = strategy.partition(topology)

        total_flops = sum(topology.get_node(node_id).flops.fp32 for node_id in topology.nodes)
        for partition in partitions:
            node_id = partition.node_id
            device_flops = topology.get_node(node_id).flops.fp32
            expected_fraction = device_flops / total_flops
            actual_fraction = partition.end - partition.start
            self.assertAlmostEqual(actual_fraction, expected_fraction, places=4)

        # Print partitions for visual
        for partition in partitions:
            print(f"Partition for {partition.node_id}: start = {partition.start}, end = {partition.end}")

if __name__ == '__main__':
    unittest.main()
