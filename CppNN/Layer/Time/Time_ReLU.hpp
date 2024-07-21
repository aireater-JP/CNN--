#pragma once
#include "../../Layer.hpp"

template <typename T>
class Time_ReLU : public Layer<T>
{
    std::vector<ReLU<T>> ReLUs;

    size_t current;

public:
    Time_ReLU() : current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        ReLUs = std::vector<ReLU<T>>(input_dimension[0], ReLU<T>());

        return input_dimension;
    }

    Array<T> forward(const Array<T> &x) override
    {
        current++;
        return ReLUs[current].forward(x);
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        return ReLUs[current].backward(dy);
    }

    void update(const T lr) override
    {
        current = 0;
    }
};