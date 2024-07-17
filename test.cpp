#include "CppNN/Layer/Tanh.hpp"

int main()
{
    Tanh<float> layer;
    Array<float> a({2, 2, 5});
    for (size_t i = 0; i < a.size(); ++i)
    {
        a[i] = i;
    }

    out(dot(reshape({0,2},a),reshape({2,0},a)).Transpose());
}