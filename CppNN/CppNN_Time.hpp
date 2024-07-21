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

    void initialize(const Index &input_size)
    {
        Index x = input_size;
        for (auto &i : _layer)
            x = i->initialize(x);

        output_size = x.back_access(0);

        is_initialized = true;
    }

    Array<float> predict(const Array<float> &e)
    {
        res.clear();
        Array<float> y;
        for (size_t i = 0; i < e.dimension()[0]; ++i)
        {
            y = e.cut({i});
            for (size_t j = 0; j < _layer.size(); ++j)
            {
                y = _layer[i]->forward(y);
            }
            std::copy(y.begin(), y.end(), (res.begin() + i * output_size));
        }
        return res;
    }

    Array<float> gradient(const Array<float> &x)
    {
        Array<float> y;
        for (size_t i = x.dimension()[0] - 1; i < x.dimension()[0]; --i)
        {
            y = x.cut({i});
            for (size_t j = _layer.size() - 1; j < _layer.size(); --i)
            {
                y = _layer[i]->backward(y);
            }
        }

        return y;
    }

    void update(const float lr)
    {
        for (size_t i = 0; i < _layer.size(); ++i)
        {
            _layer[i]->update(lr);
        }
    }
};