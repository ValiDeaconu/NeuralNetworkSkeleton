#ifndef __MATRIX_HPP_
#define __MATRIX_HPP_

#include <cstdio>
#include <vector>
#include <cstdlib>
#include <ctime>

namespace OpenNN {
    using namespace std;
    
    class Matrix {
        public:
            // Constructor
            Matrix(unsigned int rows, unsigned int cols);

            // Destructor
            ~Matrix();

            // Print function
            void print();

            // Randomize
            void randomize();

            // Operators
            Matrix& operator =(const Matrix& srcMatrix);
            Matrix operator +(const double value);
            Matrix operator +(const Matrix& srcMatrix);
            Matrix operator -(const double value);
            Matrix operator -(const Matrix& srcMatrix);
            Matrix operator *(const double scalar);
            Matrix operator *(const Matrix& srcMatrix);
            Matrix operator ^(const Matrix& srcMatrix); // element wise multiplication

            // Transpose
            Matrix transpose();

            // Map function
            void map(double (*fn)(double));

            // To array conversion
            vector<double> toArray();

            // Static from array conversion
            static Matrix fromArray(double * arr, unsigned int arr_length);
            static Matrix fromArray(vector<double> arr);

            // Getters
            unsigned int getRows();
            unsigned int getCols();
            double getValue(unsigned int r, unsigned int c);

            // Setters
            void setValue(unsigned int r, unsigned int c, double value);

        private:
            unsigned int rows;
            unsigned int cols;
            vector< vector<double> > data;
    };
}

#endif