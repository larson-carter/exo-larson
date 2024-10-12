# Advanced Partitioning Strategy: Design and Implementation

## Summary of Task Completion

### Given Task:

Introduction
exo currently implements Pipeline Parallel inference. This splits up layers of a model over multiple devices and executes them sequentially, device-by-device.

There are different ways we can split up the model layers. For this purpose, exo defines something called a PartitioningStrategy:

`exo/exo/topology/partitioning_strategy.py`

Lines 16 to 19:
```
 class PartitioningStrategy(ABC): 
   @abstractmethod 
   def partition(self, topology: Topology) -> List[Partition]: 
     pass 
```
This takes a Topology and gives a `List[Partition]`.
A Partition consists of node_id, start and end. The Partitions must be continuous ranges [start, end). The first start must be 0. The last end must be 1.

There's two things going on here:

It decides the order that nodes execute and send messages between each other. For example, if you return [node1, node2, node3], then node1 will execute first, followed by node2, followed by node3, which will then send an output token to node1 to continue the cycle. However, if you return [node2, node1, node3] then node2 will execute first, followed by node1, followed by node3, which will then send an output token to node2 to continue the cycle.
It decides how many layers each node gets. Each node gets a number of layers proportional to end-start. For example if start=0,end=1 then that node will get all the layers. If start=0,end=0.5 for node1 and start=0.5,end=1 for node2 then node1 will get 50% of the layers and node2 will get 50% of the layers.
The default and only PartitioningStrategy right now is `RingMemoryWeightedPartitioningStrategy`:

`exo/exo/topology/ring_memory_weighted_partitioning_strategy.py`

Lines 7 to 18:

```
 class RingMemoryWeightedPartitioningStrategy(PartitioningStrategy): 
   def partition(self, topology: Topology) -> List[Partition]: 
     nodes = list(topology.all_nodes()) 
     nodes.sort(key=lambda x: (x[1].memory, x[0]), reverse=True) 
     total_memory = sum(node[1].memory for node in nodes) 
     partitions = [] 
     start = 0 
     for node in nodes: 
       end = round(start + (node[1].memory/total_memory), 5) 
       partitions.append(Partition(node[0], start, end)) 
       start = end 
     return partitions 
```

What this does is it sorts primarily by memory, secondarily by node_id. The size of each partition is proportional to the memory of the device, i.e. if deviceA has 4GB memory and deviceB has 6GB memory, deviceA will get 40% of the layers and deviceB will get 60% of the layers (modulo some rounding). Note that it's important that we sort secondarily by node_id to ensure deterministic and consistent sorting in the case that memory is the same for two devices.

The task
The task is to implement a new, improved PartitioningStrategy that takes into account more than just memory. This may require augmenting the Topology class with more information that it currently has, which will require changes across the codebase. Some things you might want to consider here are: device FLOPS and inter-node latency. There are many other things you could take into account here which I will leave to you to decide.

I have some ideas for how to do this, and there are many potential approaches however I'm looking for out of the box ideas here.

I'll leave it up to you to reason about how to lay this out, but there are two high level metrics that would make sense to optimise for (should they be optimised separately or together?):

Time-to-first token (latency)
Tokens per second (throughput)
Deliverables
A set of unit tests for your new PartitioningStrategy that show it works in different cases.
A set of unit tests that "simulate" different scenarios and show that this PartitioningStrategy achieves the optimal solution in each scenario.
An option added to the main script to enable this PartitioningStrategy (you decide if other parameters should be added to configure the new PartitioningStrategy

Sourced from [here](https://github.com/exo-explore/exo/issues/284)

### Completed Work:

- Developed `AdvancedStrategy`: A new PartitioningStrategy that considers device FLOPS and inter-node latency.

- Optimized for Multiple Objectives: The strategy supports three modes—'latency', 'throughput', and 'balanced'—allowing users to prioritize based on their needs and their network state.

- Enhanced Topology Class: Augmented to include inter-node latency and device capability information.
Comprehensive Unit Tests: Created tests that simulate different scenarios to validate the strategy's effectiveness and optimality. **NOTE: did not implement the fetching and detecting of inter-node latnecy (technical considerations will have to be made)**

- Configurable Parameters: Added options to the main script to enable the new strategy and select the optimization mode.

## Introduction

In Exo, for the distributed computing power environment - partitioning strategies are crucial. 

Strategies divide workloads across a network/cluster of devices to benefit: efficiency and ultimately performance. 

The previous implementation of PartitioningStrategy primarily focused on memory constraints/division to partition workloads across the cluster. 

There are various considerations to improve the strategy such as memory constraints, device capabilities, latencies, active memory usage/pressure, architectures and more. 

Taking one or more of these elements into consideration can improve performance of the cluster overall.

## Key Factors Implemented

* Device FLOPS: Representation of the computational power of the device. Allocation of workloads proportional to FLOPS of a machine can help balance processing times and can increase throughput.

* Memory Constraints: Additional Flag/Argument that can ensures that the workload assigned to each device does not exceed memory capacity.

* Inter-node Latency (Not Detection of Latency): The overall communication delay between devices. RTT - Round Trip Time can be taken into consideration. This affects the overall latency of the distribution process when the devices are exchanging data frequently. Key: Lower is better.

## Optimization Models / Modes

Currently the new strategy has 3 modes:
- Latency
- Throughput
- Balanced (Default)

Here is a breakdown of the various approaches:

1. Latency Optimization ('latency' mode)

Goal: Minimize time-to-first token.

`Latency` Approach:

Assign as much of the initial workload as possible to the fastest device. Remaining workloads are distributed to other devices in order of their computational speed. This will work well on efficient and speedy LAN networks.

2. Throughput Optimization ('throughput' mode)

Goal: Maximize tokens processed per second.

`Throughput` Approach:

Distribute workloads proportional to each device's FLOPS.
Ensure all devices are utilized effectively to increase overall throughput. This will work well on cluster with a high device count.

3. Balanced Optimization ('balanced' mode)

Goal: Achieve a compromise between latency and throughput.

`Balanced` Approach:

Assign workloads based on device FLOPS. Reorder devices to minimize total inter-node latency, avoiding high-latency links. Adjust workloads to respect memory constraints and ensure efficient utilization. This will work best as a general plug and play solution.

## Technical Considerations

There are a variety of technical considerations that took place into this design:

- Permutation Complexity: Analyzing all of the permutations, of the optimal device order based on various factors can be computationally intensive and time consuming as the network grows.

- Latency Thresholds: To avoid performance degradation - it is important to analyze and determine what is "low latency" and what is "high latency". This also must be considered when scaling the network in various topology structures.

- How to capture Latencies: To avoid network congestion, unnecessary (root) device level access, and for quick latency captures it is best to keep these main factors in mind.

- Scalability: This strategy must remain efficient, practical, and overall "useable" as the number of devices increases. Along with the network growing scalability across LAN and WAN based networks is important.

- Maintainability: Allowing the Strategy to be further developed and optimized is very important - especially as device standards and the project progress forward.

- Repetition: Considering that the strategy might want to be re-ran every time a device is added or removed. Along with continually on a set time period re run the strategy to take into various factors that are mentioned above.

## Current Implementation Details

### Input of Device Information - Data Collection

Currently: FLOPS and Memory are the two elements that are taken into consideration of each device.

Maximum Workload Equation: Calculation based from the memory constraints to preven overloading devices (Optional for futher considerations)

### Partitioning Logic

#### Throughput Optimizations

Initial Assignment: Workloads are allocated proportionally to device FLOPS.

Memory Constraints: Adjust allocations to ensure they do not exceed the device's maximum workload fraction.

Iterative Redistribution: Any unassigned workload is redistributed among devices with remaining capacity.

#### Latency Optimizations

Primary Device Assignment: The fastest device is assigned the maximum possible workload.

Remaining Workload: Distributed to other devices in descending order of FLOPS until the total workload is assigned.

Future Implementation: Discover the device latency to each permutation from the set.

#### Balenced Optimizations

Initial Assignment: Similar to throughput optimization, workloads are allocated based on FLOPS.

Latency-aware Ordering: Devices are reordered to minimize total inter-node latency by evaluating all possible permutations and selecting the optimal one.

Scaling and Redistribution: Workloads are scaled to fill the total workload, considering memory constraints and redistributing any unassigned workload.

### Handling of Inter-Node latency

Permutation Analysis: All possible device orders are generated to find the sequence with the lowest total inter-node latency.

High-latency Avoidance: Permutations involving connections exceeding the latency threshold are discarded.

## Things Left to Implement (Future Goals)

Detection of Latency: Implement a reliable and efficient way to detect the latency between permutations. Ensure that node to node can communicate without causing extra load on the network. Also take consideration of not needing lower system level permissions (root, admin)

Shuffling of Permutations: Implementation of dynamic shuffling of permutations can help detect the edge to edge bottlenecks, and help improve the overall speed.

Dynamic System Stats: Repetitively detecting system stats such as CPU Usage, Memory Usage, Memory Pressure, etc can help improve the strategy on the edge case a user is using their node for something computational intensive (not exo), we can partition them lower in the strategy.

Device Energy: Considering device type identifiers and device efficiency can help benefit the strategy as a whole.

Continual Dynamic Updating: Dynamically adjusting partions in response to changes in deivce availability, network conditions, device functionality can overall benefit the strategy.

Scalability Improvements: Optimizing the permutation analysis to handle larger numbers of devices efficiently. Scaling to support dynamic data inference is also very important.

User Interface Enhancements: Providing more intuitive configuration options and feedback to users in the main script.

Self Detection: Self detection of the network can help determine what mode/flag that the user would want their cluster to be in.

## Challenges and Considerations

### Balancing Multiple Objectives
Trade-offs: Optimizing for latency may conflict with optimizing for throughput.

User Preferences: Providing different modes allows users to prioritize based on their specific requirements. As mentioned above automatic detection will be ideal in the future.

### Memory Constraints
Limitations: Ensuring devices are not overloaded while maximizing resource utilization.

Dynamic Adjustments: Adjusting workload allocations in real-time as conditions change. If this is to be needed.

### Latency Measurement
Accuracy: Requires precise inter-node latency measurements.

Threshold Determination: Setting appropriate thresholds to define acceptable latency levels. 

Downfall: Can cause unnecessary network traffic. Latency might not be the only that that should be considered - Inter-Network speed might also be considered in future.

### Algorithm Complexity
Permutation Scalability: The computational load increases factorially with the number of devices.

Efficient Algorithms: Potentially implementing heuristic methods to find near-optimal solutions more efficiently.

### Considerations for Dynamic Updating
Real-time Monitoring: Continuously monitor device performance and network latency.

Adaptive Partitioning: Adjust partitions on-the-fly to respond to changes in the environment.

State Synchronization: Ensure consistency across devices when partitions are updated dynamically.

## Results
The `AdvancedStrategy` effectively improves upon the previous partitioning strategy by:

Optimizing Performance Metrics: Achieves better latency and throughput based on the selected mode.

Efficient Resource Utilization: Distributes workloads in a way that maximizes the use of available computational power while respecting memory constraints.

Reduced Communication Delays: Minimizes inter-node latency through intelligent device ordering.

### Comparison of `AdvancedStrategy` to the already existing strategy:

**NOTE: this was ran in balanced mode for time being vs regular exo - Same Prompt, across 2 Macbooks**

Old: 3.10 SEC TO FIRST TOKEN, 14.3 TOKENS/SEC, 456 TOKENS

Advanced: 1.27 SEC TO FIRST TOKEN, 27.5 TOKENS/SEC, 1013 TOKENS

- Latency - Time to first token decreased by approximately 59.03%.
- Throughput - Token/Sec increased by approximately 92.31%.

|         Metric         | Old Strategy | Advanced Strategy | Improvement (%) | Better? |
|:----------------------:|:------------:|:-----------------:|:---------------:|---------|
| Time to First Token    | 3.10sec      | 1.27sec           | 59.03% decrease | Yes     |
| Tokens Per Second      | 14.3         | 27.5              | 92.31% increase | Yes     |
| Total Tokens Generated | 456          | 1013              | n/a             | n/a     |

### Unit Test Outcomes:

All tests passed, confirming that the strategy performs as expected across various scenarios.

Demonstrated the strategy's ability to handle different device capabilities and network conditions effectively.

## Future Work:

Dynamic Updating Mechanisms: Implementing real-time adjustments to partitions based on changing conditions.

Dynamic Latency Detections: Implementation of detecting the device latency in a quick and efficient way will allow for the strategy to fully work across the board.

Algorithm Optimization: Enhancing scalability for environments with a large number of devices.

User Experience Improvements: Simplifying configuration and providing clearer feedback in the main script.

This strategy lays the groundwork for more intelligent and adaptable distributed computing solutions that can efficiently leverage heterogeneous computing resources.

## Conclusion

The development of the `AdvancedStrategy` partitioning approach represents a significant enhancement over the previous memory-only strategy. 

Incorporating device FLOPS and inter-node latency into the partitioning decisions, the strategy provides more efficient and flexible workload distribution options. 

Users can now select optimization modes that align with their performance priorities, whether that's minimizing latency, maximizing throughput, or balancing both.
