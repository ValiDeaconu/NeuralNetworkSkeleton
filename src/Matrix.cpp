#include "../include/Matrix.hpp"
 
 // Constructor
Matrix::Matrix(unsigned int rows, unsigned int cols, bool isRandom) {
    this->rows = rows;
    this->cols = cols;

    double r = 0.0f;
    for (unsigned int i = 0; i < rows; ++i) {
        vector<double> columnValues;
        for (unsigned int j = 0; j < cols; ++j) {
            r = 0.0f;
            if (isRandom) {
                r = generateRandomValue();
            }

            columnValues.push_back(r);
        }

        this->values.push_back(columnValues);
    }
}

// Destructor
Matrix::~Matrix() {
    // Clear all values
    for (unsigned int i = 0; i < this->values.size(); ++i)
        this->values.at(i).clear();
    this->values.clear(); 
}

// Transpose
Matrix *Matrix::transpose() {
    Matrix *m = new Matrix(this->cols, this->rows, false);
    for (unsigned int i = 0; i < rows; ++i) {
        for (unsigned int j = 0; j < cols; ++j) {
            m->setValue(j, i, this->getValue(i, j));
        }
    }

    return m;
}

// Conversion
vector<double> Matrix::toVector() {
    vector<double> result;
    for (unsigned int i = 0; i < this->rows; ++i) {
        for (unsigned int j = 0; j < this->cols; ++j) {
            result.push_back(this->getValue(i, j));
        }
    }
    return result;
}

// Getters
unsigned int Matrix::getRows() {
    return this->rows;
}

unsigned int Matrix::getCols() {
    return this->cols;
}

double Matrix::getValue(unsigned int row, unsigned int col) {
    return this->values.at(row).at(col);
}

// Setters
void Matrix::setValue(unsigned int row, unsigned int col, double value) {
    this->values.at(row).at(col) = value;
}

// Random value between 0 and 1 generator
double Matrix::generateRandomValue() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);

    return dis(gen);
}

// Operators
Matrix& Matrix::operator =(Matrix* srcMatrix) {
    // Clear all values
    for (unsigned int i = 0; i < rows; ++i)
        this->values.at(i).clear();
    this->values.clear();

    // Copy metadata
    this->rows = srcMatrix->rows;
    this->cols = srcMatrix->cols;

    for (unsigned int i = 0; i < this->rows; ++i) {
        this->values.push_back(srcMatrix->values.at(i));
    }

    return (*this);
}

Matrix* Matrix::operator *(Matrix* srcMatrix) {
    assert(this->cols == srcMatrix->rows);

    Matrix *m = new Matrix(this->rows, srcMatrix->cols, false);
    for (unsigned int i = 0; i < this->rows; ++i) {
        for (unsigned int j = 0; j < srcMatrix->cols; ++j) {
            for (unsigned int k = 0; k < this->cols; ++k) {
                m->setValue(i, j, m->getValue(i, j) + (this->getValue(i, k) * srcMatrix->getValue(k, j)));
            }
        }
    }

    return m;
}

Matrix* Matrix::operator -(Matrix* srcMatrix) {
    assert(this->rows == srcMatrix->rows && this->cols == srcMatrix->cols);

    Matrix *m = new Matrix(this->rows, this->cols, false);
    for (unsigned int i = 0; i < this->rows; ++i) {
        for (unsigned int j = 0; j < srcMatrix->cols; ++j) {
            m->setValue(i, j, (this->getValue(i, j) - srcMatrix->getValue(i, j)));
        }
    }

    return m;
}