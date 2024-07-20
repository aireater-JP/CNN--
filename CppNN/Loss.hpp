#pragma once

#include "Array/Array.hpp"

template <typename T>
class Loss
{
public:
    virtual T forward(const Array<T> &x, const Array<T> &t) = 0;
    virtual Array<T> backward() = 0;
};

#include "Loss/Identity.hpp"
#include "Loss/Softmax.hpp"