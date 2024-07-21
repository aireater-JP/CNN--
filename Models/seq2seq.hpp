#pragma once

#include "CppNN_Time.hpp"

class seq2seq
{
    CppNN_Time e_in;
    CppNN_Time e_out;
    CppNN_Time d_in;
    CppNN_Time d_out;

    Array<float> res;
    size_t _e_t, _d_t, en_out_size;
    Index en_in_size, de_out_size;

    Time_LSTM<float> Encoder;
    Time_LSTM<float> Decoder;

    std::unique_ptr<Loss<float>> _loss;

    float batch_size = 0;

public:
    seq2seq(const size_t e_t, const size_t d_t, const Index &en_in_size, const Index &de_out_size, const size_t hidden_size)
        : _e_t(e_t), _d_t(d_t), en_in_size(en_in_size), de_out_size(de_out_size),
          Encoder(e_t, hidden_size), Decoder(d_t, hidden_size)
    {
    }

    void initialize(const Index &input_size)
    {
    }

    Array<float> predict(const Array<float> &e, const Array<float> &x)
    {
        Encoder.reset();
        Decoder.reset();

        e_out.predict(Encoder.forward(e_in.predict(e)));

        Decoder.set_h(Encoder.get_h());
        Decoder.set_c(Encoder.get_c());

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
        Encoder.set_dc(Decoder.get_dc());

        e_in.gradient(Encoder.backward(e_out.gradient(Array<float>({_e_t, en_out_size}))));

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