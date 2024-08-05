#pragma once

#include "encoder.hpp"
#include "decoder.hpp"

#include "../CppNN/Loss.hpp"

class seq2seq
{
    encoder en;
    decoder de;

    std::unique_ptr<Loss<float>> _loss;

    size_t _output_size;

public:
    seq2seq() : en(10, 10), de(10, 10, 10) {}

    void initialize(const Index &input_size)
    {
        en.initialize(input_size);
        de.initialize(input_size);
    }

    Array<float> predict(const Array<float> &e, const Array<float> &x)
    {
        for (size_t i = 0; i < e.dimension()[0]; ++i)
            en.predict(e.cut({i}));

        de.set_h(en.get_h());

        Array<float> y({x.dimension()[0], _output_size});

        for (size_t i = 0; i < x.dimension()[0]; ++i)
            y.copy(de.gradient(x.cut({i})), i);

        return y;
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        return _loss->forward(predict(e, x), t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        float y = loss(e, x, t);

        Array<float> dloss = _loss->backward();

        for (size_t i = 0; i < x.dimension()[0]; ++i)
            de.gradient(dloss.cut(i));

        en.set_dh(de.get_dh());

        for (size_t i = 0; i < e.dimension()[0]; ++i)
            en.gradient(Array<float>({e.dimension()[1]}));

        return y;
    }

    void update(const float lr)
    {
        en.update(lr);
        de.update(lr);
    }
};