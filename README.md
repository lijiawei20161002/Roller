# Roller: A Lightweight RL Exploration Tool for Queueing Systems
Roller is a lightweight Reinforcement Learning (RL) exploration tool designed to assist system operators in selecting, debugging, and interpreting diverse scheduling algorithms for queueing systems. Roller aims to bridge the gap between the promising theory of RL and its practical implementation in real-world queueing systems.

## Key Features
- Customizable queueing systems: Users can configure their own system architecture and workload.
- Algorithm recommendation: Roller automatically recommends the most suitable algorithms based on performance ranking in the built-in simulator.
- Decision tracking: Roller tracks decisions at commonly encountered states, helping users understand the behavior of different algorithms.
- Tree-based algorithm interpretation: Roller visualizes scheduling algorithms as decision trees, providing a clear understanding of the rules followed by the algorithms.
- Evaluation and testing: Roller can evaluate the impact of algorithm training parameters and running time, aiding in algorithm selection and debugging.

## Use Cases
- Testing algorithm robustness: Roller helps users evaluate and understand the performance of different RL algorithms in various scenarios and under different training configurations.
- Algorithm selection: Roller assists in selecting the most suitable algorithms for specific queueing systems based on their performance in the built-in simulator.

## Usage
- test.py gives an example to test the simulator.
- run visualization/app.py to enable user interaction via web front end for Roller. 
