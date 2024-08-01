#pragma once

#include "../CppNN/CppNN_Time.hpp"

class Attention
{
    CppNN_Time &e_in;
    CppNN_Time &e_out;
    CppNN_Time &d_in;
    CppNN_Time &d_out;

    Time_LSTM<float> Encoder;
    Time_LSTM<float> Decoder;

    Time_Attention<float> attention;

    std::unique_ptr<Loss<float>> _loss;

    size_t e_t, d_t, data_size, hidden_size;
    size_t en_out_size;

    float batch_size = 0;

public:
    Attention(CppNN_Time &e_in, CppNN_Time &e_out, CppNN_Time &d_in, CppNN_Time &d_out,
              const size_t e_t, const size_t d_t, const size_t &data_size, const size_t hidden_size)
        : e_in(e_in), e_out(e_out), d_in(d_in), d_out(d_out),
          Encoder(hidden_size), Decoder(hidden_size),
          attention({e_t, hidden_size}),
          e_t(e_t), d_t(d_t), data_size(data_size), hidden_size(hidden_size) {}

    void initialize()
    {
        en_out_size = e_out.initialize(Encoder.initialize(e_in.initialize({e_t, data_size}))).back_access(0);
        d_out.initialize(attention.initialize(Decoder.initialize(d_in.initialize({d_t, data_size}))));
    }

    Array<float> predict(const Array<float> &e, const Array<float> &x)
    {
        Encoder.reset();
        Decoder.reset();
        attention.reset();

        Array<float> hs = Encoder.forward(e_in.predict(e));
        e_out.predict(hs);

        Decoder.set_h(Encoder.get_h());
        attention.set_hs_en(hs);

        return d_out.predict(attention.forward(Decoder.forward(d_in.predict(x))));
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        return _loss->forward(predict(e, x), t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        float y = loss(e, x, t);
        Array<float> g = _loss->backward();

        //Decoder.set_dhs(Array<float>(Decoder.);
        //Decoder.set_dhs();
        d_in.gradient(attention.backward(Decoder.backward(d_out.gradient(g))));

        Encoder.set_dh(Decoder.get_dh());
        Encoder.set_dhs(attention.get_dhs_en());

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

    template <class T>
    void set_Loss(T &&l)
    {
        _loss = std::make_unique<T>(std::move(l));
    }
};