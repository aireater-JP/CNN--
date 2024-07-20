#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Loss/Identity.hpp"

class seq2seq
{
    Time_LSTM<float> Encoder;

    Time_LSTM<float> Decoder;

    Softmax_with_Loss<float> _loss;

    Array<float> res;
    size_t _e_t, _d_t;
    size_t _output_size;

public:
    seq2seq(const size_t e_t, const size_t d_t, const size_t input_size, const size_t output_size)
        : Encoder(e_t, input_size, output_size), Decoder(d_t, input_size, output_size),
          _e_t(e_t), _d_t(d_t),
          _output_size(output_size)
    {
    }

    void initialize(const Index &input_size)
    {
        Encoder.initialize(input_size);
        Decoder.initialize(input_size);
        res = Array<float>({_d_t, _output_size});
    }

    void predict_e(const Array<float> &e)
    {
        for (size_t i = 0; i < _e_t; ++i) 
            Encoder.forward(e.cut({i}));
    }

    Array<float> predict_d(const Array<float> &x)
    {
        Decoder.set_h(Encoder.get_h());
        Decoder.set_c(Encoder.get_c());
        Array<float> y({12});
        y[11] = 1;
        for (size_t i = 0; i < _d_t; ++i)
        {
            y = Decoder.forward(y);
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
        float y = loss(e, x, t);
        Array<float> g = _loss.backward();
        for (size_t i = g.dimension()[0] - 1; i < g.dimension()[0]; --i)
            Decoder.backward(g.cut({i}));

        Encoder.set_dh(Decoder.get_dh());
        Encoder.set_dc(Decoder.get_dc());

        for (size_t i = e.dimension()[0] - 1; i < e.dimension()[0]; --i)
            Encoder.backward(Array<float>({1,_output_size}));

        return y;
    }

    void update(const float lr)
    {
        Decoder.update(lr);
        Encoder.update(lr);
    }
};