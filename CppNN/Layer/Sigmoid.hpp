#pragma once
#include "../Layer.hpp"

template <typename T>
class Sigmoid : public Layer<T>
{
    Array<T> _output_cash;

public:
    Index initialize(const Index &input_dimension) override
    {
        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> y = 1.0f / (1.0f + exp(-x));
        _output_cash = y;
        return y;
    }
    Array<T> backward(const Array<T> &dy) override
    {
        return dy * (1.0f - _output_cash) * _output_cash;
    }
};

template <typename T>
Array<T> sigmoid(const Array<T> &x)
{
    return 1.0f / (1.0f + exp(-x));
}