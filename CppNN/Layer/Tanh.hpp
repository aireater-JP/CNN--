#pragma once
#include "../Layer.hpp"

template <typename T>
class Tanh : public Layer<T>
{
    Array<T> _output_cash;

public:
    Index initialize(const Index &input_dimension) override
    {
        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> y = tanh(x);
        _output_cash = y;
        return y;
    }
    Array<T> backward(const Array<T> &dy) override
    {
        Array<T> dx(dy.dimension());
        dx = dy * (1.0f - (_output_cash ^ 2.0f));
        return dx;
    }
};