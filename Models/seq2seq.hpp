#pragma once

#include "encoder.hpp";
#include "decoder.hpp";

#include "../Loss.hpp"

class seq2seq
{
    encoder en;
    decoder de;

    std::unique_ptr<Loss<float>> _loss;

public:
    void initialize(const Index &input_size)
    {
        en.initialize(input_size);
        de.initialize(input_size);
    }

    Array<float> predict(const Array<float> &e, const Array<float> &x)
    {
        en.predict(e);

        // enのhをdeに移植

        de.gradient(x);
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        return _loss->forward(predict(e, x), t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        float y = loss(e, x, t);

        de.gradient(_loss->backward());

        // enにdeのhを移植

        en.gradient();

        return y;
    }

    void update(const float lr)
    {
    }
};