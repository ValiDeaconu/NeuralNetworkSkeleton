#ifndef _NEURL_NETWORK_HPP_
#define _NEURL_NETWORK_HPP_

#include <vector>
#include <algorithm>
#include "../include/Layer.hpp"
#include "../include/Matrix.hpp"

using namespace std;

class NeuralNetwork {
    public:
        // Constructor
        NeuralNetwork(vector<int> topology);

        // Destructor
        ~NeuralNetwork();

        // Load progression from file
        NeuralNetwork(const char * path);

        // Download progression to file
        void saveProgress(const char * path);

        // Getters
        Matrix *getNeuronMatrix(int index);
        Matrix *getActivatedNeuronMatrix(int index);
        Matrix *getDerivedNeuronMatrix(int index);
        Matrix *getWeightMatrix(int index);

        double getTotalError();
        vector<double> getErrors();

        // Setter
        void setInput(vector<double> input);
        void setTarget(vector<double> target);
        void setNeuronValue(int layerIndex, int neuronIndex, double value);
        void setErrors();

        // Learning
        void feedForward();
        void backPropagation();

    private:
        vector<int> topology;
        vector<Layer *> layers;
        vector<Matrix *> weightMatrix;

        vector<double> input;
        
        vector<double> target;
        double error;
        vector<double> errors;
        vector<double> historicalErrors;
};

#endif