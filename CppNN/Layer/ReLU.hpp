#pragma once
#include "../Layer.hpp"

template <typename T>
class ReLU : public Layer<T>
{
    Array<T> _input_cash;

public:
    Index initialize(const Index &input_dimension) override
    {
        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        _input_cash = x;
        Array<T> y(x.dimension());
        for (size_t i = 0; i < x.size(); ++i)
            y[i] = x[i] * (0.0 < x[i]);

        return y;
    }
    Array<T> backward(const Array<T> &dy) override
    {
        Array<T> dx(dy.dimension());
        for (size_t i = 0; i < dy.size(); ++i)
            dx[i] = dy[i] * (0.0 < _input_cash[i]);

        return dx;
    }
};