#pragma once
#include "../../Layer.hpp"

template <typename T>
class Time_Sigmoid : public Layer<T>
{
    std::vector<Sigmoid<T>> Sigmoids;

    size_t current;

public:
    Time_Sigmoid() : current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        Sigmoids = std::vector<Sigmoid<T>>(input_dimension[0], Sigmoid<T>());

        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> t = Sigmoids[current].forward(x);
        current++;
        return t;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        return Sigmoids[current].backward(dy);
    }

    void reset(const T lr) override
    {
        current = 0;
    }
};