#include <cstdio>
#include <vector>
#include "../include/NeuralNetwork.hpp"

using namespace std;

int main() {
    vector<int> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(3);

    vector<double> input;
    input.push_back(1.0f);
    input.push_back(0.0f);
    input.push_back(1.0f);

    NeuralNetwork *nn = new NeuralNetwork(topology);
    //NeuralNetwork *nn = new NeuralNetwork("data/progress1.txt");
    nn->setInput(input);
    nn->setTarget(input);

    // Training process
    unsigned int steps = 10000;
    unsigned int proc = (int)(0.10 * steps);
    for (unsigned int g = 0; g < steps; ++g) {
        nn->feedForward();
        nn->setErrors();
        if (g % proc == 0) {
            printf("Generation: %d\n", g);
            printf("Total error: %.6f\n", nn->getTotalError());
        }
        nn->backPropagation();
    }

    nn->saveProgress("data/progress1.txt");

    delete nn;
    
    return 0;
}