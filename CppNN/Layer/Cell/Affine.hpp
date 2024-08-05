#pragma once
#include "../../Layer.hpp"

template <typename T>
class Cell_Affine
{
    Array<T> W, dW;
    Array<T> B, dB;

    Array<T> _input_cash;

public:
    Cell_Affine(size_t input_size, size_t output_size, size_t time_size) : W({input_size, output_size}), dW({input_size, output_size}),
                                                                           B({output_size}), dB({output_size}),
                                                                           _input_cash({time_size, input_size})
    {
        Random<std::uniform_real_distribution<>> r(-1.0, 1.0);
        for (auto &i : W)
            i = r();
    }

    Array<T> forward(const Array<T> &x, size_t current)
    {
        _input_cash.copy(x, {current});
        Array<T> y = dot(x, W) + B;
        return y;
    }

    Array<T> backward(const Array<T> &dy, size_t current)
    {
        dW += dot(_input_cash.cut({current}).Transpose(), dy);
        dB += dy.sum(1);

        return dot(dy, W.Transpose());
    }

    void update(const T lr) override
    {
        W -= dW * lr;
        B -= dB * lr;

        dW.clear();
        dB.clear();
    }
};