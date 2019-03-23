#ifndef _MATRIX_HPP_
#define _MATRIX_HPP_

#include <cstdio>
#include <random>
#include <vector>
#include <cassert>

using namespace std;

class Matrix {
    public:
        // Constructor
        Matrix(unsigned int rows, unsigned int cols, bool isRandom);

        // Destructor
        ~Matrix();

        // Transpose
        Matrix *transpose();

        // Conversion
        vector<double> toVector();

        // Getters
        unsigned int getRows();
        unsigned int getCols();

        double getValue(unsigned int row, unsigned int col);
        
        // Operators
        Matrix& operator =(Matrix* srcMatrix);
        Matrix* operator *(Matrix* srcMatrix);
        Matrix* operator -(Matrix* srcMatrix);

        // Setters
        void setValue(unsigned int row, unsigned int col, double value);
    private:
        unsigned int rows;
        unsigned int cols;

        vector< vector<double> > values;

        double generateRandomValue();
};

#endif