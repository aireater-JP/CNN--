#pragma once
#include "Array/Array.hpp"
#include "Random.hpp"

template <typename T>
class Time_Layer
{
public:
    virtual Index initialize(const Index &input_dimension) = 0;

    virtual Array<T> forward(const Array<T> &x) = 0;
    virtual Array<T> backward(const Array<T> &x) = 0;

    virtual void update(const T learning_rate) {};
};