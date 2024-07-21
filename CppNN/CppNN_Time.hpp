#pragma once

#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Layer/Time/Time_Affine.hpp"
#include "../CppNN/Layer/Time/Time_Attention.hpp"

#include "Loss.hpp"

class CppNN_Time
{
    std::vector<std::unique_ptr<Layer<float>>> _layer;

    bool is_initialized = false;

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
        Array<float> res(x.dimension()[0], output_size);
        Array<float> y;
        for (size_t i = 0; i < x.dimension()[0]; ++i)
        {
            y = x.cut({i});
            for (size_t j = 0; j < _layer.size(); ++j)
            {
                y = _layer[j]->forward(y);
            }
            std::copy(y.begin(), y.end(), (res.begin() + i * output_size));
        }
        return res;
    }

    Array<float> gradient(const Array<float> &dy)
    {
        Array<float> res(dy.dimension()[0], output_size);
        Array<float> dx;
        for (size_t i = dy.dimension()[0] - 1; i < dy.dimension()[0]; --i)
        {
            dx = dy.cut({i});
            for (size_t j = _layer.size() - 1; j < _layer.size(); --j)
            {
                dx = _layer[j]->backward(dx);
            }
            std::copy(dx.begin(), dx.end(), (res.begin() + i * output_size));
        }
        return res;
    }

    void update(const float lr)
    {
        for (size_t i = 0; i < _layer.size(); ++i)
        {
            _layer[i]->update(lr);
        }
    }
};