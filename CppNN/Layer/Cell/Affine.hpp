#pragma once
#include "../Layer.hpp"

template <typename T>
class Affine : public Layer<T>
{
    Array<T> &W;
    Array<T> &dW;
    Array<T> &B;
    Array<T> &dB;

    Array<T> _input_cash;

public:
    Affine(Array<T> &W, Array<T> &dW, Array<T> &B, Array<T> &dB) : W(W), dW(dW), B(B), dB(dB) {}

    Array<T> forward(const Array<T> &x) override
    {
        _input_cash = x;
        Array<T> y = dot(x, W) + B;
        return y;
    }

    Array<T> backward(const Array<T> &x) override
    {
        dW += dot(_input_cash.Transpose(), x);
        dB += x.sum(1);

        return dot(x, W.Transpose());
    }

    void update(const T learning_rate) override
    {
        W -= dW * learning_rate;
        B -= dB * learning_rate;

        dW.clear();
        dB.clear();
    }
};