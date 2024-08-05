#pragma once
#include "Array/Array.hpp"
#include "Random.hpp"

template <typename T>
class Layer
{
public:
    virtual Index initialize(const Index &input_dimension) = 0;

    virtual Array<T> forward(const Array<T> &x) = 0;
    virtual Array<T> backward(const Array<T> &dy) = 0;

    virtual void update(const T lr) {};
    virtual void reset() {};
};

#include "Layer/ReLU.hpp"
#include "Layer/Sigmoid.hpp"
#include "Layer/Affine.hpp"
#include "Layer/Tanh.hpp"
#include "Layer/Softmax.hpp"