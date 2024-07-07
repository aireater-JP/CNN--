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
    Array<T> backward(const Array<T> &x) override
    {
        Array<T> y(x.dimension());
        y = x * (1.0f - (_output_cash ^ 2.0f));
        return y;
    }
};