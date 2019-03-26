#include "../include/Matrix.hpp"
#include "../include/NeuralNetwork.hpp"

using namespace std;

namespace OpenNN {
    // Constructor
    Matrix::Matrix(unsigned int rows, unsigned int cols) {
        this->rows = rows;
        this->cols = cols;

        for (unsigned int i = 0; i < rows; ++i) {
            vector<double> col;
            for (unsigned int j = 0; j < cols; ++j) {
                col.push_back(0.0f);
            }
            data.push_back(col);
        }
    }

    // Destructor
    Matrix::~Matrix() {
        for (unsigned int i = 0; i < rows; ++i) {
            data[i].clear();
        }
        data.clear();
    }

    // Print function
    void Matrix::print() {
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                printf("%10.6lf ", data[i][j]);
            }
            printf("\n");
        }
    }

    // Randomize
    void Matrix::randomize() {
        srand(time(NULL));
        
        double r = 0.0f;
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                r = ((double)rand() / RAND_MAX) * 2 - 1;
                data[i][j] = r;
            }
        }

    }

    // Operators
    Matrix& Matrix::operator =(const Matrix& srcMatrix) {
        if (rows != srcMatrix.rows || cols != srcMatrix.cols) {
            // if dimensions are not equals, we destroy first matrix and create a new one
            for (unsigned int i = 0; i < rows; ++i) {
                data[i].clear();
            }
            data.clear();

            rows = srcMatrix.rows;
            cols = srcMatrix.cols;

            for (unsigned int i = 0; i < rows; ++i) {
                vector<double> col;
                for (unsigned int j = 0; j < cols; ++j) {
                    col.push_back(0.0f);
                }
                data.push_back(col);
            }
        }

        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                data[i][j] = srcMatrix.data[i][j];
            }
        }

        return (*this);
    }

    Matrix Matrix::operator +(const double value) {
        Matrix m(rows, cols);

        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                m.data[i][j] = data[i][j] + value;
            }
        }

        return m;
    }

    Matrix Matrix::operator +(const Matrix& srcMatrix) {
        if (rows != srcMatrix.rows || cols != srcMatrix.cols) {
            printf("Operator + failed: different matrix dimensions.\n");
            exit(1);
        }

        Matrix m(rows, cols);
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                m.data[i][j] = data[i][j] + srcMatrix.data[i][j];
            }
        }

        return m;
    }

    Matrix Matrix::operator -(const double value) {
        Matrix m(rows, cols);

        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                m.data[i][j] = data[i][j] - value;
            }
        }

        return m;
    }

    Matrix Matrix::operator -(const Matrix& srcMatrix) {
        if (rows != srcMatrix.rows || cols != srcMatrix.cols) {
            printf("Operator - failed: different matrix dimensions.\n");
            exit(1);
        }

        Matrix m(rows, cols);
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                m.data[i][j] = data[i][j] - srcMatrix.data[i][j];
            }
        }

        return m;
    }

    Matrix Matrix::operator *(const double scalar) {
        Matrix m(rows, cols);

        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                m.data[i][j] = data[i][j] * scalar;
            }
        }

        return m;
    }

    Matrix Matrix::operator *(const Matrix& srcMatrix) {
        if (cols != srcMatrix.rows) {
            printf("Operator * failed: different matrix dimensions.\n");
            exit(1);
        }

        Matrix m(rows, srcMatrix.cols);
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < srcMatrix.cols; ++j) {
                for (unsigned int k = 0; k < cols; ++k) {
                    m.data[i][j] += data[i][k] * srcMatrix.data[k][j];
                }
            }
        }

        return m;
    }
    
    // element wise multiplication
    Matrix Matrix::operator ^(const Matrix& srcMatrix) {
        if (rows != srcMatrix.rows || cols != srcMatrix.cols) {
            printf("Operator ^ (element wise multiplication) failed: different matrix dimensions.\n");
            exit(1);
        }

        Matrix m(rows, srcMatrix.cols);
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < srcMatrix.cols; ++j) {
                m.data[i][j] = data[i][j] * srcMatrix.data[i][j];
            }
        }

        return m;
    }

    // Transpose
    Matrix Matrix::transpose() {
        Matrix m(cols, rows);
        for (unsigned int i = 0; i < cols; ++i) {
            for (unsigned int j = 0; j < rows; ++j) {
                m.data[i][j] = data[j][i];
            }
        }

        return m;
    }

    // Map function
    void Matrix::map(double (*fn)(double)) {
        double arg = 0.0f;
        for (unsigned int i = 0; i < rows; ++i) {
            for (unsigned int j = 0; j < cols; ++j) {
                arg = data[i][j];
                data[i][j] = fn(arg);
            }
        }          
    }

    // To array conversion
    vector<double> Matrix::toArray() {
        vector<double> arr;
        
        for (unsigned int i = 0; i < cols; ++i) {
            for (unsigned int j = 0; j < rows; ++j) {
                arr.push_back(data[i][j]);
            }
        }

        return arr;
    }

    // Static from array conversion
    Matrix Matrix::fromArray(double * arr, unsigned int arr_length) {
        Matrix m(arr_length, 1);

        for (unsigned int i = 0; i < arr_length; ++i)
            m.data[i][0] = arr[i];

        return m;
    }

    Matrix Matrix::fromArray(vector<double> arr) {
        Matrix m(arr.size(), 1);

        for (unsigned int i = 0; i < arr.size(); ++i)
            m.data[i][0] = arr[i];

        return m;
    }

    // Getters
    unsigned int Matrix::getRows() {
        return rows;
    }

    unsigned int Matrix::getCols() {
        return cols;
    }

    double Matrix::getValue(unsigned int r, unsigned int c) {
        return data[r][c];
    }

    // Setters
    void Matrix::setValue(unsigned int r, unsigned int c, double value) {
        data[r][c] = value;
    }

}