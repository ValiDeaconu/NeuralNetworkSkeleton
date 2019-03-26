#include "../include/NeuralNetwork.hpp"

namespace OpenNN {
    // Constructor
    NeuralNetwork::NeuralNetwork(vector<unsigned int> topology,  double (*fn)(double)) {
        this->topology = topology;
        this->activateFunction = fn;
        this->learningRate = 0.01f;

        // Building weights matrices from input to hidden, ..., hidden to outpit
        for (unsigned int i = 0; i < topology.size() - 1; ++i) {
            Matrix m(topology[i + 1], topology[i]);
            m.randomize();
            weights.push_back(m);
        }

        // Building bias matrices from input to hidden, ..., hidden to outpit
        for (unsigned int i = 0; i < topology.size() - 1; ++i) {
            Matrix m(topology[i + 1], 1);
            m.randomize();
            bias.push_back(m);
        }
    }

    // Destructor
    NeuralNetwork::~NeuralNetwork() {
        topology.clear();
        weights.clear();
        bias.clear();
    }

    // Feed forward algorithm
    vector<double> NeuralNetwork::feedForward(vector<double> input) {
        Matrix inputLayer = Matrix::fromArray(input);
        vector<Matrix> layers;

        // Iterating each layer, computing input to hidden1, hidden1 to hidden2, ..., hiddenN to output
        for (unsigned int i = 0; i < weights.size(); ++i) {
            Matrix m(0, 0);
            if (i != 0)
                m = weights[i] * layers[i - 1] + bias[i];
            else
                m = weights[i] * inputLayer + bias[i];
            
            
            // Passing perceptrons thru activate function
            m.map(activateFunction);
            
            layers.push_back(m);
        }

        // Returning output layer
        return layers[layers.size() - 1].toArray();
    }

    // Train function, back propagation algorithm using Gradient Descent
    void NeuralNetwork::train(vector<double> input, vector<double> target) {
        // Feed forwarding
        vector<Matrix> layers;
        layers.push_back(Matrix::fromArray(input));

        // Iterating each layer, computing input to hidden1, hidden1 to hidden2, ..., hiddenN to output
        for (unsigned int i = 0; i < weights.size(); ++i) {
            Matrix m = weights[i] * layers[i] + bias[i];
            
            // Passing perceptrons thru activate function
            m.map(activateFunction);
            
            layers.push_back(m);
        }

        // Getting output layer
        Matrix output = layers[layers.size() - 1];

        // Computing output errors (ERROR = TARGET - OUTPUT)
        Matrix errors = Matrix::fromArray(target) - output;


        // Going backwards from weight matrix between output and last hidden to weight matrix between index and first hidden
        for (int windex = weights.size() - 1; windex >= 0; --windex) {
            // Computing errors on current hidden layer based on errors we already computed
            Matrix currentErrors = weights[windex].transpose() * errors;

            // Getting gradients by derivating the results based on formula (y * (1 - y)), where y = activateFunction(x)
            Matrix gradients = layers[windex + 1];
            double arg = 0.0f;
            for (unsigned int i = 0; i < gradients.getRows(); ++i) {
                for (unsigned int j = 0; j < gradients.getCols(); ++j) {
                    arg = gradients.getValue(i, j);
                    gradients.setValue(i, j, arg * (1 - arg));
                }
            } 

            // Formula: Delta(L[k], L[p]) = Delta(windex) = learningRate * E(windex) ^ (Gradients) x L[k].transpose()
            gradients = (gradients ^ errors) * learningRate;

            // Tweaking weight matrices, adding deltas to them

            Matrix delta = gradients * layers[windex].transpose();
            this->weights[windex] = this->weights[windex] + delta;
            this->bias[windex] = this->bias[windex] + gradients;

            // Moving backwards with next errors
            errors = currentErrors;
        }
    }
}