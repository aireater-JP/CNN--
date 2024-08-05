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
        _output_cash = softmax(x);
        return _output_cash;
    }
    Array<T> backward(const Array<T> &dy) override
    {
        Array<T> dx = _output_cash * dy;
        Array<T> s = dx.sum(0);
        return dx - _output_cash * s;
    }

private:
    Array<T> softmax(const Array<T> &x)
    {
        Array<T> exp_temp = exp(x - x.max(0));
        return exp_temp / exp_temp.sum(0);
    }
};