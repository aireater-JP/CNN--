#pragma once
#include "../Layer.hpp"

template <typename T>
class Affine : public Layer<T>
{
    Array<T> W;
    Array<T> dW;
    Array<T> B;
    Array<T> dB;

    size_t _output_size;

    Array<T> _input_cash;

public:
    Affine(const size_t output_size) : _output_size(output_size) {}

    Index initialize(const Index &input_dimension) override
    {
        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        _input_cash = x;
        Array<T> y = dot(x, W) + B;
        return y;
    }

    Array<T> backward(const Array<T> &x) override
    {
        dW = dot(_input_cash.Transpose(), x);
        dB = sum(x, 0);

        return dot(x, W.Transpose());
    }
};