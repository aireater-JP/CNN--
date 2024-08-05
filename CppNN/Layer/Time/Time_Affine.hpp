#pragma once
#include "../../Layer.hpp"
#include "../Cell/Affine.hpp"

template <typename T>
class Time_Affine : public Layer<T>
{
    Cell_Affine<T> affines;

    size_t _output_size;

    size_t current;

public:
    Time_Affine(const size_t output_size) : _output_size(output_size), current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        affines = Cell_Affine<T>(input_dimension[1], _output_size, input_dimension[0]);

        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> y = affines.forward(x, current);
        current++;
        return y;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        return affines.backward(dy, current);
    }

    void update(const T lr) override
    {
        affines.update(lr);
    }

    void reset() override
    {
        current = 0;
    }
};