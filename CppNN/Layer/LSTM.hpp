#pragma once
#include "../Layer.hpp"

#include "Tanh.hpp"
#include "Sigmoid.hpp"

template <typename T>
class LSTM : public Layer<T>
{
    Array<T> Wfx;
    Array<T> Wfh;
    Array<T> Bf;

    Array<T> Wgx;
    Array<T> Wgh;
    Array<T> Bg;

    Array<T> Wix;
    Array<T> Wih;
    Array<T> Bi;

    Array<T> Wox;
    Array<T> Woh;
    Array<T> Bo;

    Array<T> h;
    Array<T> c;

    Tanh<T> _tanh;
    Sigmoid<T> _sigmoid;

public:
    LSTM() : {}

    Index initialize(const Index &input_dimension) override
    {
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> f = _sigmoid.forward(dot(x, Wfx) + dot(h, Wfh) + Bf);
        Array<T> g = _tanh.forward(dot(x, Wgx) + dot(h, Wgh) + Bg);
        Array<T> i = _sigmoid.forward(dot(x, Wix) + dot(h, Wih) + Bi);
        Array<T> o = _sigmoid.forward(dot(x, Wox) + dot(h, Woh) + Bo);

        c = f * c + g * i;
        h = o * _tanh.forward(c);
    }

    Array<T> backward(const Array<T> &x) override
    {
        dc=dh*o;
    }
};