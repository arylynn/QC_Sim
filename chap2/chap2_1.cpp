#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <string>

using namespace std;
using namespace std::complex_literals;

using Matrix = vector<vector<complex<double>>>;
using Vector = vector<complex<double>>;
using Complex = complex<double>;

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
/**
 * @brief Adds two complex numbers
 * @param a, b two complex numbers in the same form.
 * @return The sum of two complex numbers.
 */
Complex add(Complex a, Complex b) {
    return a + b;
}

/**
 * @brief Subtracts two complex numbers
 * @param a, b two complex numbers in the same form.
 * @return The difference of two complex numbers.
 */
Complex sub(Complex a, Complex b) {
    return a - b;
}

/**
 * @brief Multiply two complex numbers
 * @param a, b two complex numbers in the same form.
 * @return The product of two complex numbers.
 */
Complex mul(Complex a, Complex b) {
    return a * b;
}

/**
 * @brief Divide two complex numbers
 * @param a, b two complex numbers in the same form.
 * @return The quotient of two complex numbers.
 */
Complex div(Complex a, Complex b) {
    return a / b;
}

// 2.1.12
/**
 * @brief Conjugates a complex number (the inverse)
 * @param a complex number in general form
 * Keeps the real part of the complex number and inverses the imaginary part
 * @return The conjugate or inverse of the complex number
 */
Complex conjugate(Complex a) {
    return Complex(a.real(), -a.imag());
}

/**
 * @brief Finds the modulus of the complex number by finding its absolute value
 * @param a complex number in general form
 * Calls abs to find the absolute value of complex number
 * @return modulus of the complex number
 */
Complex modulus(Complex a) {
    return abs(a);
}

/**
 * @brief Finds the normal or normalized form of a complex number
 * @param a complex number in general form
 * takes the real and imaginary part of the complex number and add the square of them together 
 * a^2 + b^2 and then we take the sqrt of them sqrt(a^2 + b^2) 
 * and we set this as the denominator for the real and imaginary part of the complex number
 * @return normal or normalized form of the complex number
 */
Complex norm(Complex a) {
    double magnitude = sqrt(a.real() * a.real() + a.imag() * a.imag());
    return Complex(a.real() / magnitude, a.imag() / magnitude);
}

/**
 * @brief Finds the difference between the sum of the moduli and the modulus of the sum
 * @param a, b two complex numbers in general form
 * gets the abs value of both a and b complex number sums them together
 * then get the sum of a + b and the abs value of that sum subtract them together
 * @return the difference of the moduli and modulus of the two complex number
 */
double diff(Complex a, Complex b) {
    return abs(a) + abs(b) - (abs(a + b));
}

// 2.1.15
/**
 * @brief Performs matrix addition on two matrix of the same size
 * @param a, b are matrix of the same dimensions
 * @return the sum of the two matrix being added together
 */
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

/**
 * @brief Performs scalar multiplication on the entire matrix
 * @param a, b are matrix of some dimension or vector and a complex number to
 * multiply it by
 * @return the product of the scalar multiplication on the matrix or vector
 */
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

/**
 * @brief Multiplies two matrix together
 * @param a, b are matrix with the same number of columns to rows
 * @return the product of the two matrix multplied together
 */
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

/**
 * @brief Rotates the matrix to the right 
 * @param a is a matrix of some dimension
 * @return the rotate matrix that has been rotated to the right
 */
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
/**
 * @brief Performs mconjugation on the entire matrix
 * @param a are matrix of some dimension
 * Calls conjugate function to inverse the complex number in the matrix
 * if the number is real it does nothing. As the conjugate of a real is itself
 * @return conjugatation of the entire matrix
 */
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

/**
 * @brief Finds the dagger of the matrix. Which is the tranpose
 * and the conjugate of the matrix
 * @param a are matrix of some dimension
 * Calls the transpose and conjugate function on the matrix to get the dagger
 * @return the dagger of the matrix after it has been transposed and conjugated
 */
Matrix daggerMatrix(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix matrix(row, Vector(col));

    matrix = transposeMatrix(a);
    matrix = conjugateMatrix(matrix);

    return matrix;
}

/**
 * @brief Finds the trace of a matrix
 * @param a are matrix of some dimension
 * adds all the value of the diagonal of the matrix [x,x]
 * @return the trace of the matrix
 */
Complex traceMatrix(Matrix a) {
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

    return total;
}

// 2.2.18
/**
 * @brief Finds the normalized value of the vector
 * @param a vector of some size. We find its normalized value of the vector by
 * adding all the elements in the vecotr to its conjugate.
 * Normalized value = (complex # * -complex #). Adding this togther for all
 * elements in the vector and square root the sum to get the value. 
 * @return the normalized value of the vector
 */
Complex normalize(Vector a) {
    int row = a.size();
    Complex normValue = 0;

    for (int i = 0; i < row; i++) {
        normValue = normValue + mul(a[i], conjugate(a[i]));
    }

    normValue = sqrt(normValue);

    return normValue;
}

/**
 * @brief Finds the normalized vector
 * @param a vector of some size. We find its normalized value of the vector by
 * adding all the elements in the vecotr to its conjugate.
 * Normalized value = (complex # * -complex #). Adding this togther for all
 * elements in the vector and square root the sum to get the value. 
 * divide all elements of the vector by the noramlized value
 * @return the normalized value of the vector
 */
Vector normVector(Vector a) {
    int row = a.size();
    Complex total = 0;

    Vector normv(row);

    for (int i = 0; i < row; i++) {
        normv[i] = normv[i] / normalize(a);
    }

    return normv;
}

/**
 * @brief Checks to see if a vector is normal
 * @param a Vector of some size. We check to see if it is normal
 * by seeing if its normalzied value if equal to 1. 
 * @return True if the noramalized value is equal 1. Otherwise False. 
 */
bool isNormal(Vector a) {
    Complex b = 1;
    return normalize(a) == b;
}

/**
 * @brief Checks to see if the vector is orthogonal
 * @param a vector of some size. It is orthogonal when the vector is if the
 * next element in the vector is equal to the previous element. 
 * @return True if vector is orthogonal and if otherwise False
 */
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

/**
 * @brief Checks to see if the vector is orthonormal
 * @param a vector of some size. It is orthonormal when the vector is both normal and orthogonal. 
 * @return True if vector is orthonormal and if otherwise False
 */
bool isOrthonormal(Vector a) {
    return isNormal(a) && isOrthogonal(a);
}

// 2.3.5
/**
 * @brief Checks to see if the two vectors are equal
 * @param a,b two vectors of some size. Checks to see if the elements
 * in the vectors are equal and if the size is equal
 * @return True if the two vectors are equal and false otherwise.
 */
bool isEqual(Vector a, Vector b) {
    int row = a.size();

    for (int i = 0; i < row; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

/**
 * @brief Checks to see if the two matrix are equal
 * @param a,b two matrix of some size. Checks to see if the elements
 * in the matrix are equal and if the size is equal
 * @return True if the two matrix are equal and false otherwise.
 */
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

/**
 * @brief Checks to see if the a matrix is symetric
 * @param a matrix of some size. 
 * Checks to see if the see if the diagonal of the matrix is equal
 * so at [0,1] we check to see if it is the same as [1,0]
 * @return True if the the matrix is symetric when all elements are symetrix 
 * false otherwise.
 */
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

/**
 * @brief Checks to see if the a matrix is orthongonal
 * @param a matrix of some size. 
 * Checks to see if the see if the diagonal of the matrix is equal is equal to 1
 * @return True if the the matrix is symetric when all elements all equal to 1 false otherwise.
 */
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

/**
 * @brief Checks to see if the a matrix is hermintian
 * @param a matrix of some size. 
 * Checks to see if the see if the diagonal of the matrix is equal to the mirror
 * so at [1,1] we check if [1,0] are equal
 * @return True if the the matrix is hermitian false otherwise.
 */
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

/**
 * @brief Checks to see if the a matrix is unitary
 * @param a matrix of some size. 
 * Takes the dagger of the matrix and check sto see if the diagonal of the matrix is equal to 1
 * @return True if the the matrix is unitary false otherwise.
 */
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
/**
 * @brief Checks to see if the complex number and complex vector
 * is the eigenvalue and eigen vector of a complex matrix
 * @param a,b,c takes in a matrix of some size
 * takes in a complex vector of some size
 * takes in a complex number
 * Checks to see if the complex number is the eigen value and checks to see if the vector
 * is the eigenvector to the matrix a.
 * @return True if the the the vector and complex number are the eigenvalue and eigenvalue.
 */
bool isEigen(Matrix a, Vector b, Complex c) {
    size_t n = a.size();
    Vector ab(n, Complex(0.0, 0.0));
    for (size_t i = 0; i < n; ++i) {
        if (a[i].size() != n) {
            return false;
        }
        for (size_t j = 0; j < n; ++j) {
            ab[i] += a[i][j] * b[j];
        }
    }
    
    Vector cb(n);
    for (size_t i = 0; i < n; ++i) {
        cb[i] = c * b[i];
    }
    
    double epsilon = 1e-9;
    for (size_t i = 0; i < n; ++i) {
        if (abs(ab[i] - cb[i]) > epsilon) {
            return false;
        }
    }
    
    return true;
}

// 2.4.7
/**
 * @brief Finds the tensor of two matrix of either 1d or 2d
 * @param a,b two matrix that are either 1d or 2d
 * Takes each element in matrix and multiply it by the entire matrix b
 * @return Tensor value of the matrix
 */
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
/**
 * @brief Checks to see if a 2d matrix is a boolean column stochastic matrix (matrix with only 1 and 0)
 * @param a a 2d matrix of some dimension 
 * Checks the entire matrix to see if it is either a 0 or a 1 only
 * @return returns true if it is a boolean column stochastic matrix and false if not
 */
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

/**
 * @brief Multiplies a square matrix by a postive integer value
 * @param a,x a square of some dimension and a postive integer
 * Takes each element of the matrix multiply for the integer passed in
 * @return product of the matrix multiplye by the interger
 */
Matrix matrixEx(Matrix a, int x) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[0].size(); j++) {
            a[i][j] = a[i][j] * (Complex) x;
        }
    }

    return a;
}

/**
 * @brief Multiplies a square matrix by a vector x amount of times
 * @param a,b,x a matrix of some size m x m, vectors of some size m , a postive integer 
 * multiplies the matrix by a vector x amount of times
 * @return the state of the matrix after multiplying by a vector x amount of time
 */
Matrix vectorEx(Matrix a, Matrix b, int x) {
    int row = b.size();
    int col = b[0].size();

    Matrix c(row, Vector(1));

    for (int i = 0; i < x; i++) {
        c = mulMatrix(b, a);
    }

    return c;
}

/**
 * @brief Multiplies a matrix and a vector together to find the state of the vector
 * @param a,b,a swaure boolean column stohastic matrix and a vector of m x 1 size with postive integers 
 * multiplies the matrix by a vector to get the state of the vector
 * @return the state of the vector after mutlplugin a vector by a matrix.
 */
Matrix vectorMoreEx(Matrix a, Matrix b) {
    int row = a.size();
    int col = a[0].size();

    Matrix d(row, Vector(1));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < row; j++) {
            Complex count = 0;
            for (int k = 0; k < row; k++) {
                if (b[k][j] == (Complex) 1) count+= 1;
            }

            if (b[i][j] == (Complex) 1) {
                d[i][0] += a[j][0] / count;
            }
        }
    }

    return d;
}

// 3.2.4
/**
 * @brief Checks to see if a matrix is a column stohastic matrix
 * @param a a matrix of some size
 * Checks to see if the columns of the matrix add up to 1
 * @return true if it is a column stohastic matrix false otherwise
 */
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

/**
 * @brief multiplies a column stocastic matrix by itself x times
 * @param a, x  a column stocastich matrix and a postive integer
 * multiplies a column stohastic matri by itself x amount of times m^x
 * @return the product of the matrix after m^x
 */
Matrix colTime(Matrix a, int x) {
    int row = a.size();
    int col = a[0].size();

    Matrix b = mulMatrix(a, a);
    for (int i = 0; i < x; i++) {
        b = mulMatrix(b, a);
    }

    return b;
}

/**
 * @brief finds the state of the system after n times
 * @param a, x a column stoahstic matrix and a postive integer
 * returns the first n state sof the system by calling colTime()
 * @return the first n states of the system 
 */
Matrix colState(Matrix a, int x) {
    return colTime(a, x);
}

/**
 * @brief finds the state of a matrix through a corresponding vector
 * @param a m x m column stoahstic matrix 
 * the m x m square column stohastic matrix multiplies by a vector [1, 0, ... 0] transpose
 * to get the state of the system after mutlplyiing these two and expressing it as a vector
 * @return the vector tht represents the state of the system
 */
Matrix vecState(Matrix a) {
    int row = a.size();
    int col = a[0].size();

    Matrix b(row, Vector(1));
    b[0][1] = (Complex) 1;
    
    for (int i = 0; i < row; i++) {
        b = mulMatrix(a, b);
    }

    return b;
}

// 3.3.1
/**
 * @brief Checks to see if a matrix is a column unitary matrix
 * @param a is a matrix of some size
 * Calls isUnitary() to see if a matrix is unitary. if it is unitary it 
 * is also a column untiary matrix
 * @return true if it is a column unitary matrix false otherwise
 */
bool isColUnitary(Matrix a) {
    return isUnitary(a);
}

/**
 * @brief Multiplies a column untiary matrix by itself n times
 * @param a, x is a column unitary matrix, and a postive integer
 * Calls colTime() to mulitply the amtrix to itself n times a^n
 * @return the matrix after the multiplication is done
 */
Matrix colUnitaryTime(Matrix a, int x) {
    return colTime(a, x);
}

/**
 * @brief Finds the first n states of a matrix
 * @param a, x is a column unitary matrix, and a postive integer
 * Calls colState() to find the first n states by multiplying the matrix
 * n times to get its states
 * @return the frist n states of the matrix
 */
Matrix colUnitaryState(Matrix a, int x) {
    return colState(a, x);
}

/**
 * @brief Multiplies a column untiary matrix by a vector to get a vector to describe its state
 * @param a is a column unitary matrix
 * Calls vecState() to get the state of the matrix that can be describes by a vector.
 * multplies the matrix by a vector of [1,0...0]^t
 * @return the vector that corresppond to the state of the matrix
 */
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

// 2.6.4
/**
 * @brief find the bit in the nth position of the string
 * @param bit, nth takes in a string bit which is the entire string
 * takes in an integer to find the bit in that position
 * @return the bit in that spot
 */
int nthBit(string bit, int nth) {
    int x = 0;
    for (int x = 0; x < nth + 1; x++) {
        x = (int) bit[x];
    }

    return x;
}

/**
 * @brief finds the compliment of a binary string
 * @param bit takes in a string of binary numbers
 * loop through the binary string and change it from 0 -> 1 or from 1 -> 0
 * @return thes the compliment of binary string
 */
string bitCompliment(string bit) {
    for (int x = 0; x < bit.length(); x++) {
        if (bit[x] == 0) {
            bit[x] = 1;
        } else {
            bit[x] = 0;
        }
    }

    return bit;
}

/**
 * @brief find the parity Prt(x) of the binary string
 * @param bit takes in a string of binary numbers
 * loop through the binary string and add all the numbers together
 * @return if the number is even return 0 and odd return 1
 */
int parity(string bit) {
    int x = 0;
    for (int x = 0; x < bit.length(); x++) {
        x += bit[x]; 
    }

    return (x % 2 == 0) ? 0 : 1;
}

/**
 * @brief finds the exculsive or of two binary string
 * @param a,b takes in two binary strings
 * loop the strings of binary numbers and check to see if the numbers are equal at nth position
 * @return 1 if the bits are not the same and 0 when the bits are the same
 */
string exor(string a, string b) {
    string s = "";
    for (int x = 0; x < a.length(); x++) {
        s = (a[x] != b[x]) ? '1' : '0';
    }

    return s;
}

/**
 * @brief finds the conjucnction or and of two binary strings
 * @param a,b takes in two binary strings
 * loop the strings of binary numbers and check to see if the numbers are equal at nth position
 * if both bits at the nth position are both 1 
 * @return 1 if the bits are both 1 in the nth posiiton and 0 otherwise
 */
string bitconjuction(string a, string b) {
    string s = "";
    for (int x = 0; x < a.length(); x++) {
        if (a[x] == '1' && b[x] == '1') {
            s += "1";
        } else {
            s += "0";
        }
    }

    return s;
}

/**
 * @brief finds the inner product of two binary string
 * @param a,b takes in two binary strings
 * calls bit conjuction to find the AND of two binary string
 * the uses the return from bitconjuction() function call we call parity()
 * which finds the parity of the string  
 * @return 1 if the sum of the bits is off and 0 if it is even
 */
int innerProduct(string a, string b) {
    return parity(bitconjuction(a, b));
}

/**
 * @brief converts binary numbers to base 10 numbers
 * @param a takes in one binary string
 * we start from the left of the string and we take takes the length - 1 - x to get the exponet
 * @return the converted number in base 10
 */
int binaryToNormal(string a) {
    int sum = 0;
    for (int x = 0; x < a.length(); x++) {
        sum += (int) pow(a[x], (a.length() - 1) - x);
    }

    return sum;
}

/**
 * @brief converts integers base 10 to binary numbers
 * @param a takes in one binary string
 * repeatly divide the integer by 2 and find the remainder and continue
 * @return the binary string 
 */
string normalToBinary(int a) {
    string s = "";
    while (a > 0) {
        if (a % 2 == 0) {
            s = '0' + s;
        } else {
            s = '1' + s;
        }
        a /= 2;
    }

    return s;
}

// 5.1.9
/**
 * @brief Takes in a binary string creates a boolean matrix where it is all 0 but at the nth spot is a 1
 * @param s takes in one binary string
 * converts the binary string to deciaml numbers and at that spot in the matrix is a 1. 0101 -> 5
 * The matrix will be 15x1 and at the 5th position it will be 1 and the rest is 0
 * @return the boolean matrix.
 */
Matrix boolMatrix(string s) {
    int x = binaryToNormal(s);
    int max = (int) pow(2, s[0]);

    int row = x;
    int col = 1;

    Matrix m(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (i == x) {
                m[i][j] = 1;
            }

            m[i][j] = 0;
        }
    }

    return m;
}

/**
 * @brief Creates a matrix of uniform probabilty of all the same number for every element in the matrix
 * @param n takes in an integer to do 2^n
 * Takes in an integer performs 2^n. The matrix will be 2^n x 1 size. Every element in the matrix 
 * will be 1/2^n
 * @return the uniform probabilty matrix.
 */
Matrix uniformProbMatrix(int n) {
    int max = pow(2, n);
    int row = max;
    int col = 1;

    Matrix m(row, Vector(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            m[i][j] = 1 / max;
        }
    }

    return m;
}


/**
 * @brief Finds the theta and the phi for the angle of a bloch sphere from a qubit
 * @param a,b are two complex numbers to descirbe a qubit
 * Takes in a complex number and calls norm to find the normailied value for the number for both complex numbers
 * Finds the longitude and latiude by using 2 * arccose(a) for long
 * and latitude is arctan^2(imaginary part of b, and imaginary part for a)
 * @return the the pair of number {theta, phi}
 */
pair<Complex, Complex> qubitAngle(Complex a, Complex b) {
    a = norm(a);
    b = norm(b);

    Complex longitude = (complex<double>) 2 * acos(a);
    Complex latitude = atan2(b.imag(), a.imag());

    return {a, b};
}

/**
 * @brief Finds the probability that the qubit is in that state of the vector
 * @param a,b are a vector of complex number and a binary string
 * Takes in a binary string and converts it to decimal 10 -> 2
 * the state we are checking will be at v[2] of the vector
 * to find the probabilty it is v[2] * v[2]
 * @return the the probabilty of the that it is found in that state.
 */
Complex probQubit(Vector v, string s) {
    int x = binaryToNormal(s);
    Complex prob =  v[x] * v[x];
    return prob;
}

/**
 * @brief Assumes that the first bit measured is |1> and we calculate the state of the vector
 * @param v takes in a vectors of some size
 * We calculate the probabilty of the state of the vector. Given that the the probabilty is 1/sqrt(length of the vector)
 * Since we assumed that the bit is at state |1> we know that it is not in state |0> so half of the bit can we eleminated
 * So it will become 1/length of vector. and the first half of the new vector will be 0 and the later half is 1/length of vector
 * @return the new vector of the state probabilty.
 */
Vector probState(Vector v) {
    Complex prob = 1 / v.size();

    for (int x = 0; x < v.size(); x++) {
        if (x < v.size() / 2) {
            v[x] = 0;
        } else {
            v[x] = prob;
        }
    }

    return v;
}

/**
 * @brief Has the user input which bit to measure and return the state of the vector
 * @param v, j are a vector of some size and a user input of an integer for which bit to measure
 * Takes in user input for which bit to measure. Given that a bit can be anywhere we take the user input if the bit is in 
 * state |1> we recalaute the probabilty and state |0> will 0. if th bit is in state |0> we recaluate the probabilty and state
 * |1> will be 0.
 * @return the the new probabilty for the state of the vector
 */
Vector probStateInput(Vector v, int j) {
    Complex prob = 1 / v.size();

    if (j >= v.size() / 2) {
        for (int x = 0; x < v.size(); x++) {
            if (x < v.size() / 2) {
                v[x] = 0;
            } else {
                v[x] = prob;
            }
        }
    } else {
        for (int x = 0; x < v.size(); x++) {
            if (x < v.size() / 2) {
                v[x] = 1;
            } else {
                v[x] = 0;
            }
        }
    }

    return v;
}

// 5.2.4
Matrix identityMatrix(int size) {
    Matrix id(size, Vector(size, {0, 0}));
    for (int i = 0; i < size; ++i) {
        id[i][i] = {1, 0};
    }
    return id;
}

Matrix hadamardGate(int num, int control, int target) {
    Matrix hadamard = {{1/sqrt(2), 1/sqrt(2)}, 
                    {1/sqrt(2), -1/sqrt(2)}};

    for (int x = 0; x < num; x++) {
        hadamard = tensor(hadamard, hadamard);
    }

    return hadamard;
}

int main() {
    cout << "hello world" << endl;
    return 0;
}