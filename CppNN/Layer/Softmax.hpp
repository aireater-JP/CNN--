#pragma once
#include "../Layer.hpp"

template <typename T>
class Softmax : public Layer<T>
{
    Array<T> _output_cash;

public:
    Index initialize(const Index &input_dimension) override
    {
        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> exp_temp = exp(x - x.max(0));
        _output_cash = exp_temp / exp_temp.sum(0);
        return _output_cash;
    }
    Array<T> backward(const Array<T> &x) override
    {
        Array<T> y = _output_cash * x;
        y -= _output_cash * y.sum(0);
        return y;
    }
};