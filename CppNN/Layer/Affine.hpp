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
        W = Array<T>({input_dimension.back_access(0), _output_size});
        B = Array<T>({_output_size});

        Random<std::uniform_real_distribution<>> r(-1.0, 1.0);
        for (auto &i : W)
            i = r();

        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        _input_cash = x;
        Array<T> y = dot(x, W) + B;
        return y;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        dW += dot(_input_cash.Transpose(), dy);
        dB += dy.sum(1);

        return dot(dy, W.Transpose());
    }

    void update(const T learning_rate) override
    {
        W -= dW * learning_rate;
        B -= dB * learning_rate;

        dW.clear();
        dB.clear();
    }
};