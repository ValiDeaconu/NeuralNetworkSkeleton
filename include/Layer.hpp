#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <vector>
#include "../include/Neuron.hpp"
#include "../include/Matrix.hpp"

using namespace std;

class Layer {
    public:
        // Constructor
        Layer(unsigned int size);

        // Destructor
        ~Layer();

        // Getters
        unsigned int getSize();
        vector<Neuron *> getNeurons();

        // Setters
        void setValue(int index, double value);
        void setNeurons(vector<Neuron *> neurons);

        // Matrix conversion
        Matrix * toMatrix();
        Matrix * toMatrixActivatedValues();
        Matrix * toMatrixDerivedValues();

    private:
        unsigned int size;
        vector<Neuron *> neurons;

};

#endif