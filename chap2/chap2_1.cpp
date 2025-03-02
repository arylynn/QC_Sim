#include <iostream>
#include <vector>
#include <array>
#include <complex>
#include <cmath>

using namespace std;

// 2.1.3
complex<double> Gen2Exp(complex<double> a) {
    double r = abs(a);
    double theta = arg(a);

    return r * exp(complex<double>(theta));
}

complex<double> Gen2Pol(complex<double> a) {
    double r = abs(a);
    double theta = arg(a);

    return polar(r, theta);
}

complex<double> Pol2Gen(complex<double> a) {
    double r = a.real();
    double i = a.imag();

    return complex<double>(r, i);
}

complex<double> Pol2Exp(complex<double> a) {
    double r = a.real();
    double theta = a.imag();

    return r * exp(complex<double>());
}

complex<double> Exp2Pol(complex<double> a) {
    double r = abs(a);
    double theta = arg(a);

    return polar(r, theta);
}

complex<double> Exp2Gen(complex<double> a) {
    double r = a.real();
    double theta = a.imag();

    return complex<double> (r * cos(theta), r * sin(theta));
}

// 2.1.6
complex<double> add(complex<double> a, complex<double> b) {
    return a + b;
}

complex<double> sub(complex<double> a, complex<double> b) {
    return a - b;
}

complex<double> mul(complex<double> a, complex<double> b) {
    return a * b;
}

complex<double> div(complex<double> a, complex<double> b) {
    return a / b;
}

// 2.1.12
complex<double> conjugate(complex<double> a) {
    return complex<double>(a.real(), -a.imag());
}

complex<double> modulus(complex<double> a) {
    return abs(a);
}

complex<double> norm(complex<double> a) {
    return Exp2Gen(a);
}

complex<double> diff(complex<double> a, complex<double> b) {
    return abs(a) + abs(b) - (abs(a) + abs(b));
}

// 2.1.15
vector<vector<complex<double>>> sumMatrix(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> sum(row, vector<complex<double>>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            sum[i][j] = a[i][j] + b[i][j];
        }
    }

    return sum;
}

vector<vector<complex<double>>> scalarMultiply(vector<vector<complex<double>>> a, complex<double> b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> scalar(row, vector<complex<double>>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            scalar[i][j] = b * a[i][j];
        }
    }

    return scalar;
}

vector<vector<complex<double>>> mulMatrix(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b) {
    int rowa = a.size();
    int cola = a[0].size();
    int rowb = b.size();
    int colb = b[0].size();

    vector<vector<complex<double>>> product(rowa, vector<complex<double>>(colb));

    for (int i = 0; i < rowa; i++) {
        for (int j = 0; j < colb; j++) {
            for (int k = 0; k < cola; k++) {
                product[i][j] = a[i][k] * b[k][j];
            }
        }
    }

    return product;
}

vector<vector<complex<double>>> transposeMatrix(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> transpose(row, vector<complex<double>>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            transpose[j][i] = a[i][j];
        }
    }

    return transpose;
}

// 2.1.20
vector<vector<complex<double>>> conjugateMatrix(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> conjugateM(row, vector<complex<double>>(col));

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            conjugateM[i][j] = conjugate(a[i][j]);
        }
    }

    return conjugateM;
}

vector<vector<complex<double>>> daggerMatrix(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> matrix(row, vector<complex<double>>(col));

    matrix = transposeMatrix(a);
    matrix = conjugateMatrix(matrix);

    return matrix;
}

vector<vector<complex<double>>> traceMatrix(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();
    complex<double> total = 0;

    vector<vector<complex<double>>> trace(row, vector<complex<double>>(col));

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
complex<double> normalize(vector<complex<double>> a) {
    int row = a.size();
    complex<double> normValue = 0;

    for (int i = 0; i < row; i++) {
        normValue = normValue + mul(a[i], conjugate(a[i]));
    }

    normValue = sqrt(normValue);

    return normValue;
}

vector<complex<double>> normMatrix(vector<complex<double>> a) {
    int row = a.size();
    complex<double> total = 0;

    vector<complex<double>> normv(row);

    for (int i = 0; i < row; i++) {
        normv[i] = normv[i] / normalize(a);
    }

    return normv;
}

bool isNormal(vector<complex<double>> a) {
    complex<double> b = 1;
    return normalize(a) == b;
}

bool isOrthogonal(vector<complex<double>> a) {
    int row = a.size();
    complex<double> total = a[0];

    for (int i = 0; i < row; i++) {
        if (total == a[i]) {
            total = a[i];
        } else {
            return false;
        }
    }

    return true;
}

bool isOrthonormal(vector<complex<double>> a) {
    return isNormal(a) && isOrthogonal(a);
}

// 2.3.5
bool isEqual(vector<complex<double>> a, vector<complex<double>> b) {
    int row = a.size();

    for (int i = 0; i < row; i++) {
        if (a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

bool isEqual(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b) {
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

bool isSymmetric(vector<vector<complex<double>>> a) {
    vector<vector<complex<double>>> b = transposeMatrix(a);
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

bool isOrthogonalMatrix(vector<vector<complex<double>>> a) {
    vector<vector<complex<double>>> b = transposeMatrix(a);
    a = mulMatrix(a, b);
    complex<double> x = 1.0;
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

bool isHermitian(vector<vector<complex<double>>> a) {
    vector<vector<complex<double>>> b = daggerMatrix(a);
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

bool isUnitary(vector<vector<complex<double>>> a) {
    vector<vector<complex<double>>> b = daggerMatrix(a);
    a = mulMatrix(a, b);
    complex<double> x = 1.0;
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
bool isEigen(vector<vector<complex<double>>> a, vector<complex<double>>b, complex<double> c) {
    return true;
}

// 2.4.7
vector<vector<complex<double>>> tensor(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> tensorProduct;
    complex<double> t;
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            t = a[i][j];
            tensorProduct = scalarMultiply(b, t);
        }
    }

    return tensorProduct;
}

// 3.1.7
bool isEigen(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            if (a[i][j] != (complex<double>) 0 || a[i][j] != (complex<double>) 1) {
                return false;
            }
        }
    }

    return true;
}

vector<vector<complex<double>>> matrixEx(vector<vector<complex<double>>> a, int x) {
    for (int i = 0; i < x; i++) {
        a = mulMatrix(a, a);
    }

    return a;
}


vector<vector<complex<double>>> vectorEx(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b, int x) {
    int row = b.size();
    int col = b[0].size();

    vector<vector<complex<double>>> c(row, vector<complex<double>>(1));

    for (int i = 0; i < x; i++) {
        c = mulMatrix(b, a);
    }

    return c;
}

vector<vector<complex<double>>> vectorMoreEx(vector<vector<complex<double>>> a, vector<vector<complex<double>>> b, vector<vector<complex<double>>> c, int x) {
    int row = c.size();
    int col = c[0].size();

    vector<vector<complex<double>>> d(row, vector<complex<double>>(1));

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
bool isCol(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    complex<double> total;

    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++) {
            total += a[i][j];
        }
        if (total != (complex<double>) 1) {
            return false;
        }
        total = 0;
    }

    return true;
}

vector<vector<complex<double>>> colTime(vector<vector<complex<double>>> a, int x) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> b = mulMatrix(a, a);
    for (int i = 0; i < x - 1; i++) {
        b = mulMatrix(b, a);
    }

    return b;
}

vector<vector<complex<double>>> colState(vector<vector<complex<double>>> a, int x) {
    for (int i = 0; i < x; i++) {
        a = colTime(a, x);
    }

    return a;
}

vector<vector<complex<double>>> vecState(vector<vector<complex<double>>> a) {
    int row = a.size();
    int col = a[0].size();

    vector<vector<complex<double>>> b(row, vector<complex<double>>(1));

    for (int i = 0; i < row; i++) {
        a = mulMatrix(a, b);
    }

    return a;
}

// 3.3.1
bool isColUnitary(vector<vector<complex<double>>> a) {
    return isUnitary(a);
}

vector<vector<complex<double>>> colUnitaryTime(vector<vector<complex<double>>> a, int x) {
    return colTime(a, x);
}

vector<vector<complex<double>>> colUnitaryState(vector<vector<complex<double>>> a, int x) {
    return colState(a, x);
}

vector<vector<complex<double>>> colUnitaryVecState(vector<vector<complex<double>>> a) {
    return vecState(a);
}

int main() {
    cout << "hello world" << endl;
}