#include <cstdio>
#include <cmath>

#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"

using namespace std;
using namespace OpenNN;

vector< pair< vector<double>, vector<double> > > training_data;

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

int main() {
    srand(time(NULL));

    vector<unsigned int> topology;
    topology.push_back(2);
    topology.push_back(2);
    topology.push_back(1);

    NeuralNetwork nn(topology, sigmoid);

    vector<double> input(2);
    vector<double> target(1);
    
    input[0] = 0;   input[1] = 0;   target[0] = 0;
    training_data.push_back(make_pair(input, target));

    input[0] = 0;   input[1] = 1;   target[0] = 1;
    training_data.push_back(make_pair(input, target));

    input[0] = 1;   input[1] = 0;   target[0] = 1;
    training_data.push_back(make_pair(input, target));

    input[0] = 1;   input[1] = 1;   target[0] = 0;
    training_data.push_back(make_pair(input, target));

    for (unsigned int loop = 0; loop < 400000; ++loop) {
        int r = rand() % training_data.size();
        nn.train(training_data[r].first, training_data[r].second);
        if (loop % 9999 == 0)
            printf("Training index %d\n", loop);
    }

    for (unsigned int idx = 0; idx < training_data.size(); ++idx) {
        vector<double> output = nn.feedForward(training_data[idx].first);
        printf("Input index %d: ", idx);
        for (unsigned int i = 0; i < output.size(); ++i) {
            printf("%10.6lf ", output[i]);
        }
        printf("\n");
    }

    return 0;
}