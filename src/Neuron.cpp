#include "../include/Neuron.hpp"

// Constructor
Neuron::Neuron(double value) {
    setValue(value);
}

// Getters
double Neuron::getValue() {
    return this->value;
}
double Neuron::getActivatedValue() {
    return this->activatedValue;
}
double Neuron::getDerivedValue() {
    return this->derivedValue;
}

// Setter
void Neuron::setValue(double value) {
    this->value = value;
    activate();
    derive();
}

// Fast Sigmoid Function
// f(x) = x / (1 + |x|)
void Neuron::activate() {
    this->activatedValue = this->value / (1 + abs(this->value));
}

// Derivate for Fast Sigmoid Function
// f'(x) = f(x) * (1 - f(x))
void Neuron::derive() {
    this->derivedValue = this->activatedValue * (1 - this->activatedValue);
}