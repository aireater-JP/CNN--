#pragma once

#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Layer/Time/Time_Affine.hpp"

#include "Loss.hpp"

class CppNN_Time
{
    std::vector<std::unique_ptr<Layer<float>>> _layer;

    bool is_initialized = false;

    Array<float> res;

    size_t output_size;

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
            x = i->initialize(x);

        output_size = x.back_access(0);

        is_initialized = true;

        return x;
    }

    Array<float> predict(const Array<float> &x)
    {
        res.clear();
        Array<float> y;
        for (size_t i = 0; i < x.dimension()[0]; ++i)
        {
            y = x.cut({i});
            for (size_t j = 0; j < _layer.size(); ++j)
            {
                y = _layer[i]->forward(y);
            }
            std::copy(y.begin(), y.end(), (res.begin() + i * output_size));
        }
        return res;
    }

    Array<float> gradient(const Array<float> &dy)
    {
        Array<float> dx;
        for (size_t i = dy.dimension()[0] - 1; i < dy.dimension()[0]; --i)
        {
            dx = dy.cut({i});
            for (size_t j = _layer.size() - 1; j < _layer.size(); --i)
            {
                dx = _layer[i]->backward(dx);
            }
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