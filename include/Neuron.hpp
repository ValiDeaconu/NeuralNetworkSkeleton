#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#include <cmath>

using namespace std;

class Neuron {
    public:
        // Constructor
        Neuron(double value);

        // Fast Sigmoid Function
        // f(x) = x / (1 + |x|)
        void activate();

        // Derivate for Fast Sigmoid Function
        // f'(x) = f(x) * (1 - f(x))
        void derive();

        // Getters
        double getValue();
        double getActivatedValue();
        double getDerivedValue();

        // Setter
        void setValue(double value);

    private:
        double value;
        double activatedValue;
        double derivedValue;
};

#endif