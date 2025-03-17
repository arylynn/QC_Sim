#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>

using namespace std;

typedef vector<vector<complex<double>>> Matrix;
typedef vector<complex<double>> Vector;
typedef complex<double> Complex;

// 2.1.3
/**
 * @brief Coverts a complex number from general form to exponential form.
 * @param a is a complex number in the general form
 * Take the absolute value of the complex number and the phase angle of the complex number.
 * @return We return it as r * e^(theta(i))
 */
Complex Gen2Exp(Complex a) {
    double r = abs(a);
    double theta = arg(a);

    return r * exp(Complex(theta));
}

/**
 * @brief Coverts a complex number from general form to polar form.
 * @param a is a complex number in the general form
 * Take the absolute value of the complex number and the phase angle of the complex number.
 * @return We return it as (r, theta).
 */
Complex Gen2Pol(Complex a) {
    double r = abs(a);
    double theta = arg(a);

    return polar(r, theta);
}

/**
 * @brief Coverts a complex number from polar form to general form.
 * @param a is a complex number in the polar form
 * Take the real number of the polar form r (r, theta) and the imaginary theta.
 * @return complex number of the real and imaginary r + theta * i.
 */
Complex Pol2Gen(Complex a) {
    double r = a.real();
    double i = a.imag();

    return Complex(r, i);
}

/**
 * @brief Coverts a complex number from polar form to exponential form.
 * @param a is a complex number in the polar form
 * Take the real number of the polar form r (r, theta) and the imaginary theta.
 * @return complex number of the real and imaginary in exponential form r * e^i(theta)
 */
Complex Pol2Exp(Complex a) {
    double r = a.real();
    double theta = a.imag();

    return r * exp(Complex());
}

/**
 * @brief Coverts a complex number from exponential form to polar form.
 * @param a is a complex number in the exponential form
 * Take the real number of the polar form r, r * e^i(theta) and the imaginary theta.
 * @return complex number of the real and imaginary in polar form (r, i).
 */
Complex Exp2Pol(Complex a) {
    double r = abs(a);
    double theta = arg(a);

    return polar(r, theta);
}

/**
 * @brief Coverts a complex number from exponential form to general form.
 * @param a is a complex number in the exponential form
 * Take the real number of the exponential form r, r * e^i(theta) and the imaginary theta.
 * Takes the cosine and the sin of the theta multiply by the real.
 * @return complex number of the real and imaginary for the general form a + bi.
 */
Complex Exp2Gen(Complex a) {
    double r = a.real();
    double theta = a.imag();

    return Complex (r * cos(theta), r * sin(theta));
}

// 2.1.6
Complex add(Complex a, Complex b) {
    return a + b;
}

Complex sub(Complex a, Complex b) {
    return a - b;
}

Complex mul(Complex a, Complex b) {
    return a * b;
}

Complex div(Complex a, Complex b) {
    return a / b;
}

// 2.1.12
Complex conjugate(Complex a) {
    return Complex(a.real(), -a.imag());
}

Complex modulus(Complex a) {
    return abs(a);
}

Complex norm(Complex a) {
    return Exp2Gen(a);
}

Complex diff(Complex a, Complex b) {
    return abs(a) + abs(b) - (abs(a) + abs(b));
}

// 2.1.15
Matrix sumMatrix(Matrix a, Matrix b) {
    int row = a.size();
    int col = a[0].size();

    Matrix sum(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sum[i][j] = a[i][j] + b[i][j];
        }
    }

    return sum;
}

Matrix scalarMultiply(Matrix a, Complex b) {
    int row = a.size();
    int col = a[0].size();

    Matrix scalar(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            scalar[i][j] = b * a[i][j];
        }
    }

    return scalar;
}

Matrix mulMatrix(Matrix a, Matrix b) {
    int rowa = a.size();
    int cola = a[0].size();
    int rowb = b.size();
    int colb = b[0].size();

    Matrix product(rowa, Vector(colb));

    for (int i = 0; i < rowa; i++) {
        for (int j = 0; j < colb; j++) {
            for (int k = 0; k < cola; k++) {
                product[i][j] = a[i][k] * b[k][j];
            }
        }
    }

    return product;
}

Matrix transposeMatrix(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix transpose(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            transpose[j][i] = a[i][j];
        }
    }

    return transpose;
}

// 2.1.20
Matrix conjugateMatrix(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix conjugateM(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            conjugateM[i][j] = conjugate(a[i][j]);
        }
    }

    return conjugateM;
}

Matrix daggerMatrix(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix matrix(row, Vector(col));

    matrix = transposeMatrix(a);
    matrix = conjugateMatrix(matrix);

    return matrix;
}

Matrix traceMatrix(Matrix a) {
    int row = a.size();
    int col = a[0].size();
    Complex total = 0;

    Matrix trace(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (i == j) {
                total = add(trace[i][j], total);
            }
        }
    }

    return trace;
}

// 2.2.18
Complex normalize(Vector a) {
    int row = a.size();
    Complex normValue = 0;

    for (int i = 0; i < row; i++) {
        normValue = normValue + mul(a[i], conjugate(a[i]));
    }

    normValue = sqrt(normValue);

    return normValue;
}

Vector normMatrix(Vector a) {
    int row = a.size();
    Complex total = 0;

    Vector normv(row);

    for (int i = 0; i < row; i++) {
        normv[i] = normv[i] / normalize(a);
    }

    return normv;
}

bool isNormal(Vector a) {
    Complex b = 1;
    return normalize(a) == b;
}

bool isOrthogonal(Vector a) {
    int row = a.size();
    Complex total = a[0];

    for (int i = 0; i < row; i++) {
        if (total == a[i]) {
            total = a[i];
        } else {
            return false;
        }
    }

    return true;
}

bool isOrthonormal(Vector a) {
    return isNormal(a) && isOrthogonal(a);
}

// 2.3.5
bool isEqual(Vector a, Vector b) {
    int row = a.size();

    for (int i = 0; i < row; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

bool isEqual(Matrix a, Matrix b) {
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(a[i][j] != b[i][j]) {
                return false;
            }
        }
    }

    return true;
}

bool isSymmetric(Matrix a) {
    Matrix b = transposeMatrix(a);
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(a[i][j] != b[j][i]) {
                return false;
            }
        }
    }

    return true;
}

bool isOrthogonalMatrix(Matrix a) {
    Matrix b = transposeMatrix(a);
    a = mulMatrix(a, b);
    Complex x = 1.0;
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(a[i][i] != x) {
                return false;
            }
        }
    }

    return true;
}

bool isHermitian(Matrix a) {
    Matrix b = daggerMatrix(a);
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(a[i][i] != b[j][i]) {
                return false;
            }
        }
    }

    return true;
}

bool isUnitary(Matrix a) {
    Matrix b = daggerMatrix(a);
    a = mulMatrix(a, b);
    Complex x = 1.0;
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if(a[i][i] != x) {
                return false;
            }
        }
    }

    return true;
}

// 2.3.10
bool isEigen(Matrix a, Vector b, Complex c) {
    return true;
}

// 2.4.7
Matrix tensor(Matrix a, Matrix b) {
    int row = a.size();
    int col = a[0].size();

    Matrix tensorProduct;
    Complex t;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            t = a[i][j];
            tensorProduct = scalarMultiply(b, t);
        }
    }

    return tensorProduct;
}

// 3.1.7
bool isEigen(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (a[i][j] != (Complex) 0 || a[i][j] != (Complex) 1) {
                return false;
            }
        }
    }

    return true;
}

Matrix matrixEx(Matrix a, int x) {
    for (int i = 0; i < x; i++) {
        a = mulMatrix(a, a);
    }

    return a;
}


Matrix vectorEx(Matrix a, Matrix b, int x) {
    int row = b.size();
    int col = b[0].size();

    Matrix c(row, Vector(1));

    for (int i = 0; i < x; i++) {
        c = mulMatrix(b, a);
    }

    return c;
}

Matrix vectorMoreEx(Matrix a, Matrix b, Matrix c, int x) {
    int row = c.size();
    int col = c[0].size();

    Matrix d(row, Vector(1));

    for (int i = 0; i < x; i++) {
        d = mulMatrix(c, a);
    }

    int z = x;
    z = 0;

    for (int i = 0; i < z; i++) {
        d = mulMatrix(c, b);
    }

    return d;
}

// 3.2.4
bool isCol(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Complex total;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            total += a[i][j];
        }
        if (total != (Complex) 1) {
            return false;
        }
        total = 0;
    }

    return true;
}

Matrix colTime(Matrix a, int x) {
    int row = a.size();
    int col = a[0].size();

    Matrix b = mulMatrix(a, a);
    for (int i = 0; i < x - 1; i++) {
        b = mulMatrix(b, a);
    }

    return b;
}

Matrix colState(Matrix a, int x) {
    for (int i = 0; i < x; i++) {
        a = colTime(a, x);
    }

    return a;
}

Matrix vecState(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix b(row, Vector(1));

    for (int i = 0; i < row; i++) {
        a = mulMatrix(a, b);
    }

    return a;
}

// 3.3.1
bool isColUnitary(Matrix a) {
    return isUnitary(a);
}

Matrix colUnitaryTime(Matrix a, int x) {
    return colTime(a, x);
}

Matrix colUnitaryState(Matrix a, int x) {
    return colState(a, x);
}

Matrix colUnitaryVecState(Matrix a) {
    return vecState(a);
}

// 4.1.6
/**
 * @brief Takes a state vector and normalizes the vector.
 * @param a is a state vector of a quantuam system.
 * Calls normalize functions to get the normvalue of the vector and then normalizes 
 * the vector with that value.
 * @return the normalized vector after applying the normvalue.
 */
Vector stateNorm(Vector a) {
    Complex x = normalize(a);

    for (int i = 0; i < a.size(); i++) {
        a[i] = a[i] / x;
    }

    return a;
}

/**
 * @brief Finds the phase of a state vector.
 * @param a, r takes in a state vector and a real number.
 * Call stateNorm to get the normalized vector and then we multiply it by the phase
 * e^(ir) to get the state vector with the specified phase.
 * @return return the state vector with phase e^(ir).
 */
Vector statePhase(Vector a, double r) {
    a = stateNorm(a);
    Complex phase = exp(Complex(r));

    for (int i = 0; i < a.size(); i++) {
        a[i] = a[i] * phase;
    }

    return a;
}

/**
 * @brief Takes in a ket vector and find its associated bra vector.
 * @param a a finite deminensional ket vector. 
 * A ket vector is a column vector and we take the dagger so we transpose and conjuagate it
 * to a row vector. we can skip the transpose part since transpose of a transpose are equal
 * so we just take the conjugate and return its as a row vector. 
 * @return the finite dimensional bra vector assiociated with it. 
 */
Vector ket2Bra(Vector a) {
    for (int i = 0; i < a.size(); i++) {
        a[i] = conj(a[i]); 
    }

    return a;
}

/**
 * @brief Takes in a bra vector and find its associated bra vector.
 * @param a a finite deminensional bra vector. 
 * A ket vector is a column vector and we take the dagger so we transpose and conjuagate it
 * to a row vector. we can skip the transpose part since transpose of a transpose are equal
 * so we just take the conjugate and return its as a row vector. 
 * @return the finite dimensional ket vector assiociated with it. 
 */
Vector bra2Ket(Vector a) {
    for (int i = 0; i < a.size(); i++) {
        a[i] = conj(a[i]); 
    }

    return a;
}

/**
 * @brief function takes in a ket and bra vectors and find the matrix associated with it
 * @param a, b are vectors of finite dimensions
 * calls ket2Bra to one of the matrix since the bra is the dagger of the ket and
 * multiplies the elements of the matrix togeter
 * @return the results of the 2d matrix from mulitplying a ket and bra vector
 */
Matrix ketBraMatrix(Vector a, Vector b) {
    int rows = a.size();
    int cols = b.size();
    b = ket2Bra(b);

    Matrix ketbra(rows, Vector(cols));

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < b.size(); j++) {
            ketbra[i][j] = a[i] * b[j];
        }
    }

    return ketbra;
}

// 4.2.8
/**
 * @brief function finds the expected value of the state of the system.
 * @param a, b, c are an observable and vectors of finite dimensions
 * Calls stateNorm function to normalize the vector. Goes through the vector to find the probabilty for each state.
 * @return the Expected value
 */
Complex expectedValue(Matrix a, Vector b, Vector c) {
    Complex ev = 0.0;
    c = stateNorm(c);

    Vector matrix(c.size(), 0.0);

    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            matrix[i] += a[i][j] * c[j];
        }
    }
    
    for (size_t i = 0; i < c.size(); i++) {
        ev += conjugate(c[i]) * matrix[i];
    }

    return ev;
}

/**
 * @brief function finds the dispersion of the system.
 * @param a, b, c are an observable and vectors of finite dimensions
 * Follows the formula for variance ev^2 - ev^2. Subtracts expected value sqaured by 
 * every element added together when they are squared.
 * @return the dispersion of the system.
 */
Complex dispersion(Matrix a, Vector b, Vector c) {
    Complex ev = expectedValue(a, b, c);

    Matrix a_squared(a.size(), Vector(a.size(), 0.0));
    for (size_t i = 0; i < a.size(); i++) {
        for (size_t j = 0; j < a.size(); j++) {
            for (size_t k = 0; k < a.size(); k++) {
                a_squared[i][j] += a[i][k] * a[k][j];
            }
        }
    }

    Complex ev_squared = expectedValue(a_squared, b, c);

    return ev_squared - (ev * ev);
}

int main() {
    cout << "hello world" << endl;
}