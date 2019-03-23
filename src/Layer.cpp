#include "../include/Layer.hpp"

// Constructor
Layer::Layer(unsigned int size) {
    this->size = size;
    for (unsigned int i = 0; i < size; ++i)
        this->neurons.push_back(new Neuron(0.0f));
}

Layer::Layer(vector<Neuron *> neurons) {
    this->size = neurons.size();
    this->neurons = neurons;
}

// Destructor
Layer::~Layer() {
    for (unsigned int i = 0; i < this->neurons.size(); ++i)
        delete this->neurons.at(i);
    this->neurons.clear();
}

// Getters
unsigned int Layer::getSize() {
    return this->size;
}

vector<Neuron *> Layer::getNeurons() {
    return this->neurons;
}

// Setters
void Layer::setValue(int index, double value) {
    this->neurons.at(index)->setValue(value);
}

void Layer::setNeurons(vector<Neuron *> neurons) {
    this->neurons = neurons;
}

// Matrix conversion
Matrix * Layer::toMatrix() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (unsigned int i = 0; i < neurons.size(); ++i) {
        m->setValue(0, i, this->neurons.at(i)->getValue());
    }

    return m;
}
Matrix * Layer::toMatrixActivatedValues() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (unsigned int i = 0; i < neurons.size(); ++i) {
        m->setValue(0, i, this->neurons.at(i)->getActivatedValue());
    }
    
    return m;
}

Matrix * Layer::toMatrixDerivedValues() {
    Matrix *m = new Matrix(1, this->neurons.size(), false);
    for (unsigned int i = 0; i < neurons.size(); ++i) {
        m->setValue(0, i, this->neurons.at(i)->getDerivedValue());
    }
    
    return m;
}
