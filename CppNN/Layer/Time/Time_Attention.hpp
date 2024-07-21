#pragma once
#include "../../Layer.hpp"
#include "../Cell/AttentionWeight.hpp"
#include "../Cell/WeightSum.hpp"

template <typename T>
class Time_Attention : public Layer<T>
{
    Array<T> hs_en;

    Array<T> res;

    std::vector<AttentionWeight<T>> att_wei;
    std::vector<WeightSum<T>> wei_sum;

    size_t current;

public:
    Time_Attention() : current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        return input_dimension;
    }

    Array<T> forward(const Array<T> &hs_de) override
    {
        Array<T> y = wei_sum[current].forward(hs_en, att_wei[current].forward(hs_en, hs_de));
        current++;
        return y;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        wei_sum[current].backward(dy);
        att_wei[current].backward();
        return ;
    }

    void update(const T lr) override
    {
    }
};