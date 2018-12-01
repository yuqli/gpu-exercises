// 20181010
// Yuqiong Li
// Implement a convolutional neural net in cpp
#include <iostream>

using std::cout;
using std::endl;
using std::cin;

int main() {
    int m, s;  // m is the length of a, s is the length of mask
    float * a, * mask;
    float * b;  // result array

    m = 5;
    s = 3;

    a = new float[m];
    b = new float[m];
    mask = new float[s];

    // initialize the array
    for (int i = 0; i < m; i++){
        a[i] = (float) i;
        b[i] = 0;
    }

    for (int i = 0; i < s; i++){
        mask[i] = (float) i / 1.34 + 0.32;
    }

    for (int i = 0; i < m; i++)
        cout << a[i] << '\t';
    cout << endl;

    for (int i = 0; i < s; i++)
        cout << mask[i] << '\t';
    cout << endl;

    // perform convolution
    for (int i = 0; i < m; i++){
        float buff = 0.0;
        int left = (i-s/2) >= 0 ? (i-s/2) : 0;  // left boundary
        int right = (i+s/2) <= (m-1) ? (i+s/2) : (m-1);   // right boundary
        for (int j = left; j <= right; j++){
            buff += a[j] * mask[j-left];
        }
        b[i] = buff;
    }

    for (int i = 0; i < m; i++)
        cout << b[i] << '\t';
    cout << endl;

    delete [] a;
    delete [] b;
    delete [] mask;
    return 0;
}