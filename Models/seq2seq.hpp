#pragma once

#include "CppNN_Time.hpp"

class seq2seq
{
    CppNN_Time e_in;
    CppNN_Time e_out;
    CppNN_Time d_in;
    CppNN_Time d_out;

    Array<float> res;
    size_t e_t, d_t, data_size, hidden_size;
    size_t en_out_size;

    Time_LSTM<float> Encoder;
    Time_LSTM<float> Decoder;

    std::unique_ptr<Loss<float>> _loss;

    float batch_size = 0;

public:
    seq2seq(const size_t e_t, const size_t d_t, const size_t &data_size, const size_t hidden_size, CppNN_Time e_in, CppNN_Time e_out, CppNN_Time d_in, CppNN_Time d_out)
        : e_in(e_in), e_out(e_out), d_in(d_in), d_out(d_out),
          e_t(e_t), d_t(d_t), data_size(data_size), hidden_size(hidden_size),
          Encoder(hidden_size), Decoder(hidden_size)
    {
    }

    void initialize()
    {
        en_out_size = e_out.initialize(Encoder.initialize(e_in.initialize({e_t, data_size}))).back_access(0);
        d_out.initialize(Decoder.initialize(d_in.initialize({d_t, data_size})));
    }

    Array<float> predict(const Array<float> &e, const Array<float> &x)
    {
        Encoder.reset();
        Decoder.reset();

        e_out.predict(Encoder.forward(e_in.predict(e)));

        Decoder.set_h(Encoder.get_h());

        return d_out.predict(Decoder.forward(d_in.predict(x)));
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        return _loss->forward(predict(e, x), t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        float y = loss(e, x, t);
        Array<float> g = _loss->backward();

        d_in.gradient(Decoder.backward(d_out.gradient(g)));

        Encoder.set_dh(Decoder.get_dh());

        e_in.gradient(Encoder.backward(e_out.gradient(Array<float>({e_t, en_out_size}))));

        batch_size++;

        return y;
    }

    void update(const float lr)
    {
        batch_size = 0;

        float LR = lr / batch_size;

        e_in.update(LR);
        e_out.update(LR);
        d_in.update(LR);
        d_out.update(LR);
        Decoder.update(LR);
        Encoder.update(LR);
    }
};