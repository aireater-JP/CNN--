#include "CppNN/Layer/Softmax.hpp"

int main()
{
    Softmax<float> s;

    Array<float> x({2,9});
    x[0] = -4;
    x[1] = -3;
    x[2] = -2;
    x[3] = -1;
    x[4] = 0;
    x[5] = 1;
    x[6] = 2;
    x[7] = 3;
    x[8] = 4;

    x[9] = 8;
    x[10] = 7;
    x[11] = 6;
    x[12] = 5;
    x[13] = 4;
    x[14] = 3;
    x[15] = 2;
    x[16] = 1;
    x[17] = 0;

    out(s.forward(x));
    out(s.backward(x));
}