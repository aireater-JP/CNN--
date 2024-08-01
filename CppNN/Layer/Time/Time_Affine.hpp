#pragma once
#include "../../Layer.hpp"
#include "../Cell/Affine.hpp"

template <typename T>
class Time_Affine : public Layer<T>
{
    Array<T> W;
    Array<T> dW;
    Array<T> B;
    Array<T> dB;

    std::vector<Cell_Affine<T>> affines;

    size_t _input_size, _output_size;

    size_t current;

public:
    Time_Affine(const size_t output_size) : _output_size(output_size), current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        affines = std::vector<Cell_Affine<T>>(input_dimension[0], Cell_Affine<T>(W, dW, B, dB));

        _input_size = input_dimension[1];
        W = Array<T>({_input_size, _output_size});
        dW = Array<T>({_input_size, _output_size});
        B = Array<T>({_output_size});
        dB = Array<T>({_output_size});

        Random<std::uniform_real_distribution<>> r(-1.0, 1.0);
        for (auto &i : W)
            i = r();

        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> X = reshape({0, _input_size}, x);
        Array<T> y = affines[current].forward(X);
        current++;
        return y;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        Array<T> dY = reshape({0, _output_size}, dy);
        Array<T> dx = affines[current].backward(dY);
        return dx;
    }

    void update(const T lr) override
    {
        current = 0;

        W -= dW * lr;
        B -= dB * lr;

        dW.clear();
        dB.clear();
    }
};