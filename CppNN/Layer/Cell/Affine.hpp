#pragma once
#include "../../Layer.hpp"

template <typename T>
class Cell_Affine
{
    Array<T> &W;
    Array<T> &dW;
    Array<T> &B;
    Array<T> &dB;

    Array<T> _input_cash;

public:
    Cell_Affine(Array<T> &W, Array<T> &dW, Array<T> &B, Array<T> &dB) : W(W), dW(dW), B(B), dB(dB) {}

    Array<T> forward(const Array<T> &x)
    {
        _input_cash = x;
        Array<T> y = dot(x, W) + B;
        return y;
    }

    Array<T> backward(const Array<T> &dy)
    {
        dW += dot(_input_cash.Transpose(), dy);
        dB += dy.sum(1);

        return dot(dy, W.Transpose());
    }
};