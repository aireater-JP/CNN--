#pragma once

#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Layer/Time/Time_Affine.hpp"
#include "../CppNN/Layer/Time/Time_Attention.hpp"
#include "../CppNN/Layer/Time/Time_ReLU.hpp"

#include "Loss.hpp"

class CppNN_Time
{
    std::vector<std::unique_ptr<Layer<float>>> _layer;

    bool is_initialized = false;

public:
    template <class T>
    void add_Layer(T &&l)
    {
        _layer.emplace_back(std::make_unique<T>(std::move(l)));
        is_initialized = false;
    }

    Index initialize(const Index &input_size)
    {
        Index x = input_size;
        for (auto &i : _layer)
        {
            x = i->initialize(x);
        }

        is_initialized = true;

        return x;
    }

    Array<float> predict(const Array<float> &x)
    {
        Array<float> y = x;
        for (auto &i : _layer)
        {
            y = i->forward(y);
        }
        return y;
    }

    Array<float> gradient(const Array<float> &dy)
    {
        Array<float> dx = dy;
        for (auto &i : _layer)
        {
            dx = i->forward(dx);
        }
        return dx;
    }

    void update(const float lr)
    {
        for (size_t i = 0; i < _layer.size(); ++i)
        {
            _layer[i]->update(lr);
        }
    }
};