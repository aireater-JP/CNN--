#pragma once
#include <Array/Array.hpp>

template <typename T>
class Layer
{
    virtual Index initialize(const Index &input_size) = 0;

    virtual Array<T> forward(const Array<T> &x) = 0;
    virtual Array<T> backward(const Array<T> &x) = 0;

    virtual void update(const T learning_rate) {};

protected:
    Index input_size;
};