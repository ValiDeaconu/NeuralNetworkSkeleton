#ifndef __NEURAL_NETWORK_HPP_
#define __NEURAL_NETWORK_HPP_

#include <vector>

#include "../include/Matrix.hpp"

using namespace std;

namespace OpenNN {
    class NeuralNetwork {
        public:
            // Constructor
            NeuralNetwork(vector<unsigned int> topology, double (*fn)(double));

            // Destructor
            ~NeuralNetwork();

            // Feed forward algorithm
            vector<double> feedForward(vector<double> input);

            // Train -> back propagation algorithm using Gradient Descent
            void train(vector<double> input, vector<double> target);

        private:
            double (*activateFunction)(double);
            double learningRate;

            vector<unsigned int> topology;

            vector<Matrix> weights;
            vector<Matrix> bias;
    };
}

#endif