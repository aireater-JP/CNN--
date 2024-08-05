#pragma once
#include "../../Layer.hpp"
#include "../Cell/AttentionWeight.hpp"
#include "../Cell/WeightSum.hpp"

template <typename T>
class Time_Attention : public Layer<T>
{
    Array<T> hs;
    Array<T> dhs;

    std::vector<AttentionWeight<T>> att_wei;
    std::vector<WeightSum<T>> wei_sum;

    Index hs_size;

    size_t current;

public:
    Time_Attention(const Index &hs_size) : dhs(hs_size), hs_size(hs_size) {}

    Index initialize(const Index &input_dimension) override
    {
        att_wei = std::vector<AttentionWeight<T>>(input_dimension[0], AttentionWeight<T>(hs_size, input_dimension[1]));
        wei_sum = std::vector<WeightSum<T>>(input_dimension[0], WeightSum<T>(hs_size, input_dimension[1]));
        return {input_dimension[0], input_dimension[1] * 2};
    }

    Array<T> forward(Array<T> &h) override
    {
        Array<T> res({h.dimension()[0], h.dimension()[1] * 2});
        Array<T> y = wei_sum[current].forward(hs, att_wei[current].forward(hs, h));

        std::copy(y.begin(), y.end(), res.begin());
        std::copy(h.begin(), h.end(), res.begin() + y.size());

        current++;
        return res;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;

        Array<T> dx = att_wei[current].backward(hs, wei_sum[current].backward(hs, dy)) + dy;
        dhs += att_wei[current].get_dhs() + wei_sum[current].get_dhs();
        return dx;
    }

    Array<T> get_dhs() { return dhs; }

    void set_hs(const Array<T> &hs) { hs = hs; }

    void reset() { dhs_en.clear(); }
};