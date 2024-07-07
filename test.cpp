#include "CppNN/Layer/Tanh.hpp"

int main()
{
    Tanh<float> layer;
    Array<float> a({2, 3});
    a[0] = 0;
    a[1] = 1;
    a[2] = 2;
    a[3] = 3;
    a[4] = 4;
    a[5] = 5;
    Array<float> b({2, 3}, 1);
    out(layer.forward(a));
    out(layer.backward(b));
}