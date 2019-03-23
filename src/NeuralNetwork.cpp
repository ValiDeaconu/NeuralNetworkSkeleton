#include "../include/NeuralNetwork.hpp"

using namespace std;

// Constructor
NeuralNetwork::NeuralNetwork(vector<int> topology) {
    this->topology = topology;

    for (unsigned int i = 0; i < topology.size(); ++i) {
        Layer *l = new Layer(topology.at(i));
        this->layers.push_back(l);
    }

    for (unsigned int i = 0; i < topology.size() - 1; ++i) {
        Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);
        this->weightMatrix.push_back(m);
    }

    for (int i = 0; i < topology.at(topology.size() - 1); ++i)
        this->errors.push_back(0);  
}

// Destructor
NeuralNetwork::~NeuralNetwork() {
    topology.clear();
    for (unsigned int i = 0; i < layers.size(); ++i)
        delete layers.at(i);
    layers.clear();
    for (unsigned int i = 0; i < weightMatrix.size(); ++i)
        delete weightMatrix.at(i);
    weightMatrix.clear();
    input.clear();
    target.clear();
    errors.clear();
    historicalErrors.clear();

}

// Load progression from file
NeuralNetwork::NeuralNetwork(const char * path) {

}

// Download progression to file
void NeuralNetwork::saveProgress(const char * path) {

}

// Getters
Matrix * NeuralNetwork::getNeuronMatrix(int index) {
    return this->layers.at(index)->toMatrix();
}

Matrix * NeuralNetwork::getActivatedNeuronMatrix(int index) {
    return this->layers.at(index)->toMatrixActivatedValues();
}

Matrix * NeuralNetwork::getDerivedNeuronMatrix(int index) {
    return this->layers.at(index)->toMatrixDerivedValues();
}

Matrix * NeuralNetwork::getWeightMatrix(int index) {
    return this->weightMatrix.at(index);
}

double NeuralNetwork::getTotalError() {
    return this->error;
}

vector<double> NeuralNetwork::getErrors() {
    return this->errors;
}

// Setter
void NeuralNetwork::setInput(vector<double> input) {
    this->input = input;

    for (unsigned int i = 0; i < input.size(); ++i) {
        this->layers.at(0)->setValue(i, input.at(i));
    }
}

void NeuralNetwork::setTarget(vector<double> target) {
    this->target = target;
}

void NeuralNetwork::setNeuronValue(int layerIndex, int neuronIndex, double value) {
    this->layers.at(layerIndex)->setValue(neuronIndex, value);
}

void NeuralNetwork::setErrors() {
    assert(this->target.size() != 0);
    unsigned int targetSize = this->target.size();

    unsigned int outputLayerIndex = this->layers.size() - 1;
    assert(targetSize == this->layers.at(outputLayerIndex)->getNeurons().size());

    this->error = 0.0f;
    vector<Neuron *> outputNeurons = this->layers.at(outputLayerIndex)->getNeurons();

    for (unsigned int i = 0; i < targetSize; ++i) {
        double tmp = pow(outputNeurons.at(i)->getActivatedValue() - target.at(i), 2);
        this->errors.at(i) = tmp;
        this->error += tmp;
    }

    this->error *= 0.5;

    historicalErrors.push_back(this->error);
}

// Learning
void NeuralNetwork::feedForward() {
    // Loop into each layer, excepting output layer, and multiply matrices
    for (unsigned int i = 0; i < this->layers.size() - 1; ++i) {
        Matrix * A = (i == 0) ? this->getNeuronMatrix(i) : this->getActivatedNeuronMatrix(i);
        Matrix * B = this->getWeightMatrix(i);
        Matrix * C = (*A) * B;

        for (unsigned k = 0; k < C->getCols(); ++k) {
            this->setNeuronValue(i + 1, k, C->getValue(0, k));
        }

        delete A;
        delete C;
    }
}

void NeuralNetwork::backPropagation() {
    vector<Matrix *> newWeights;
    
    // Output layer to hidden layer
    unsigned int outputLayerIndex = this->layers.size() - 1;
    Matrix * derivedValues = this->layers.at(outputLayerIndex)->toMatrixDerivedValues();
    Matrix * gradientsOutput = new Matrix(1, this->layers.at(outputLayerIndex)->getSize(), false);

    for (unsigned int i = 0; i < this->errors.size(); ++i) {
        double d = derivedValues->getValue(0, i);
        double e = this->errors.at(i);
        gradientsOutput->setValue(0, i, d * e);
    }

    unsigned int lastHiddenLayerIndex = outputLayerIndex - 1;
    Layer *lastHiddenLayer = this->layers.at(lastHiddenLayerIndex);
    Matrix *weightsOutput = this->weightMatrix.at(lastHiddenLayerIndex);
    Matrix *gradientsOutputTranspose = gradientsOutput->transpose();
    Matrix *lastHiddenLayerActivatedValues = lastHiddenLayer->toMatrixActivatedValues();
    Matrix *deltaOutput = (*gradientsOutputTranspose) * lastHiddenLayerActivatedValues;
    Matrix *deltaOutputTranspose = deltaOutput->transpose();

    Matrix *weightsOutputMinusDeltaOutput = *weightsOutput - deltaOutputTranspose;

    newWeights.push_back(weightsOutputMinusDeltaOutput);

    delete deltaOutput;
    delete deltaOutputTranspose;
    delete lastHiddenLayerActivatedValues;
    delete gradientsOutputTranspose;
    delete derivedValues;

    // Going back from lastHiddenLayer to firstHiddenLayer
    for (unsigned int i = lastHiddenLayerIndex; i > 0; --i) {
        Layer *l = this->layers.at(i);
        Matrix *activatedHidden = l->toMatrixActivatedValues();
        Matrix *derivedGradients  = new Matrix(1, l->getSize(), false);
        Matrix *weightMatrix = this->weightMatrix.at(i);
        Matrix *originalWeight = this->weightMatrix.at(i - 1);
        if (i == lastHiddenLayerIndex) {
            for (unsigned int r = 0; r < weightMatrix->getRows(); ++r) {
                double sum = 0.0f;
                for (unsigned int c = 0; c < weightMatrix->getCols(); ++c) {
                    sum += gradientsOutput->getValue(0, c) * weightMatrix->getValue(r, c);
                }

                derivedGradients->setValue(0, r, sum * activatedHidden->getValue(0, r));
            }
        }

        Matrix *leftNeurons;
        if (i - 1 == 0) {
            leftNeurons = this->layers.at(i-1)->toMatrix();
        } else {
            leftNeurons = this->layers.at(i-1)->toMatrixActivatedValues();
        }

        Matrix *derivedGradientsTranspose = derivedGradients->transpose();        
        Matrix *deltaWeights = (*derivedGradientsTranspose) * leftNeurons; 
        Matrix *deltaWeightsTranspose = deltaWeights->transpose(); 

        Matrix *computedWeights = (*originalWeight) - deltaWeightsTranspose; 

        newWeights.push_back(computedWeights);
    
        delete derivedGradientsTranspose;
        delete activatedHidden;
        delete leftNeurons;
        delete deltaWeights;
        delete deltaWeightsTranspose;
        delete derivedGradients;
    }

    delete gradientsOutput;    

    for (unsigned int i = 0; i < this->weightMatrix.size(); ++i)
        delete this->weightMatrix.at(i);
    this->weightMatrix.clear();

    reverse(newWeights.begin(), newWeights.end());
    this->weightMatrix = newWeights;
}