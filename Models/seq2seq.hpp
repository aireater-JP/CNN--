#pragma once

#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Layer/Time/Time_Affine.hpp"
#include "../CppNN/Loss/Identity.hpp"

class seq2seq
{
    Time_Affine<float> e_in;
    Time_LSTM<float> Encoder;

    Time_Affine<float> d_in;
    Time_LSTM<float> Decoder;
    Time_Affine<float> d_out;

    Softmax_with_Loss<float> _loss;

    Array<float> res;
    size_t _e_t, _d_t;
    size_t _output_size;

public:
    seq2seq(const size_t e_t, const size_t d_t, const size_t input_size, const size_t output_size)
        : e_in(e_t, input_size), Encoder(e_t, input_size, output_size),
          d_in(d_t, input_size), Decoder(d_t, input_size, output_size), d_out(d_t, output_size),
          _e_t(e_t), _d_t(d_t),
          _output_size(output_size)
    {
    }

    void initialize(const Index &input_size)
    {
        Index idx = input_size;
        idx = e_in.initialize(idx);
        idx = Encoder.initialize(idx);

        idx = d_in.initialize(idx);
        idx = Decoder.initialize(idx);
        idx = d_out.initialize(idx);

        res = Array<float>({_d_t, _output_size});
    }

    void predict_e(const Array<float> &e)
    {
        Array<float> y;
        for (size_t i = 0; i < _e_t; ++i)
        {
            y = e.cut({i});
            y = e_in.forward(y);
            Encoder.forward(y);
        }
    }

    Array<float> predict_d(const Array<float> &x)
    {
        Decoder.set_h(Encoder.get_h());
        Decoder.set_c(Encoder.get_c());
        Array<float> y({12});
        y[11] = 1;
        for (size_t i = 0; i < _d_t; ++i)
        {
            y = d_in.forward(y);
            y = Decoder.forward(y);
            y = d_out.forward(y);
            std::copy(y.begin(), y.end(), (res.begin() + i * _output_size));
        }
        return res;
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        predict_e(e);
        return _loss.forward(predict_d(x), t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        Encoder.reset();
        Decoder.reset();
        res.clear();

        float y = loss(e, x, t);
        Array<float> g = _loss.backward();
        Array<float> z;
        for (size_t i = g.dimension()[0] - 1; i < g.dimension()[0]; --i)
        {
            z = g.cut({i});
            z = d_out.backward(z);
            z = Decoder.backward(z);
            z = d_in.backward(z);
        }

        Encoder.set_dh(Decoder.get_dh());
        Encoder.set_dc(Decoder.get_dc());

        for (size_t i = e.dimension()[0] - 1; i < e.dimension()[0]; --i)
        {
            z = Encoder.backward(Array<float>({1, _output_size}));
            e_in.backward(z);
        }

        return y;
    }

    void update(const float lr)
    {
        Decoder.update(lr);
        Encoder.update(lr);
    }
};