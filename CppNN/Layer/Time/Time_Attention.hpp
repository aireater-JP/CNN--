#pragma once
#include "../../Layer.hpp"
#include "../Cell/AttentionWeight.hpp"
#include "../Cell/WeightSum.hpp"

template <typename T>
class Time_Attention : public Layer<T>
{
    Array<T> hs_en;
    Array<T> dhs_en;

    std::vector<AttentionWeight<T>> att_wei;
    std::vector<WeightSum<T>> wei_sum;

    Index hs_s;

public:
    Time_Attention(const Index &hs_s) : dhs_en(hs_s), hs_s(hs_s) {}

    Index initialize(const Index &input_dimension) override
    {
        att_wei = std::vector<AttentionWeight<T>>(input_dimension[0], AttentionWeight<T>(hs_s, input_dimension[1]));
        wei_sum = std::vector<WeightSum<T>>(input_dimension[0], WeightSum<T>(hs_s, input_dimension[1]));
        return {input_dimension[0], input_dimension[1] * 2};
    }

    Array<T> forward(const Array<T> &hs_de) override
    {
        Array<float> res({hs_de.dimension()[0], hs_de.dimension()[1] * 2});
        Array<float> y;
        for (size_t i = 0; i < hs_de.dimension()[0]; ++i)
        {
            y = wei_sum[i].forward(hs_en, att_wei[i].forward(hs_en.cut({i}), hs_de));
            std::copy(y.begin(), y.end(), (res.begin() + i * hs_de.dimension()[1] * 2));
            std::copy(hs_de.begin(), hs_de.end(), (res.begin() + i * hs_de.dimension()[1] * 2 + hs_de.dimension()[1]));
        }
        return res;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        Array<float> res({dy.dimension()[0], dy.dimension()[1]});
        Array<float> dh;
        for (size_t i = 0; i < dy.dimension()[0]; ++i)
        {
            dh = att_wei[i].backward(hs_en, wei_sum[i].backward(hs_en, dy));

            std::copy(dh.begin(), dh.end(), (res.begin() + i * dy.dimension()[1]));
            dhs_en += wei_sum[i].get_dhs() + att_wei[i].get_dhs();
        }
        res += dy;

        return res;
    }

    Array<T> get_dhs_en() { return dhs_en; }

    void set_hs_en(const Array<T> &hs) { hs_en = hs; }

    void reset() { dhs_en.clear(); }
};