#pragma once
#include "../Layer.hpp"

template <typename T>
class Convolutional2D : public Layer<T>
{

public:
    Index initialize(const Index &input_dimension) override
    {
    }

    Array<T> forward(const Array<T> &x) override
    {
    }
    Array<T> backward(const Array<T> &x) override
    {
    }
private:

};
