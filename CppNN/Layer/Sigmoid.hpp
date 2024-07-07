#pragma once
#include "../Layer.hpp"

template <typename T>
class Sigmoid : public Layer<T>
{
    Array<T> _output_cash;

public:
    Index initialize(const Index &input_size) override
    {
        return input_size;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> y = 1.0f / (1.0f + exp(-x));
        _output_cash = y;
        return y;
    }
    Array<T> backward(const Array<T> &x) override
    {
        Array<T> y = x * (1.0f - _output_cash) * _output_cash;
        return y;
    }
};