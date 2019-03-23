# Neural Network Skeleton

The basic structure of a neural network implemented in C++. Without any kind of memory leaks, it can learn without damaging your system.

## Installation

```bash
make
```

## Usage

1. Create a new input file (format 0)

```
 <generations>
 <topology_size>
 <topology_vector>
 <input_size>
 <input_vector>
 <target_size>
 <target_vector>
```
Examples available for any file formats in [data folder](https://github.com/ValiDeaconu/NeuralNetworkSkeleton/tree/master/data).

2. Create a new Neural Network.
```bash
./NeuralNetwork train new <path_to_input_file> <path_to_progress_file> 
```
Input file has to be file format 0.
Progress file with be the file where the network will save its progress.

3. Train Neural Network
```bash
./NeuralNetwork train old <path_to_input_file> <path_to_progress_file> 
```
Input file has to be file format 1.
Progress file with be the file where the network will load and save its progress.

4. Compute an input
```bash
./NeuralNetwork response <path_to_input_file> <path_to_progress_file>
```
Input file has to be file format 2.
Progress file with be the file where the network will load its progress.
