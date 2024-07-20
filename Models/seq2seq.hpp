#include "../CppNN/Layer/Time/Time_LSTM.hpp"
#include "../CppNN/Loss/Identity.hpp"

class seq2seq
{
    Time_LSTM<float> Encoder;

    Time_LSTM<float> Decoder;

    Identity_with_Loss<float> _loss;

    Array<float> res;

public:
    seq2seq() : Encoder(10000, 256), Decoder(10000, 256)
    {
    }

    void initialize(const Index &input_size)
    {
        Encoder.initialize(input_size);
        Decoder.initialize(input_size);
        res = Array<float>({10000, input_size.back_access(0)});
    }

    void predict_e(const Array<float> &e)
    {
        for (size_t i = 0; i < e.dimension()[0]; ++i)
            Encoder.forward(e.cut({i}));
    }

    Array<float> predict_d(const Array<float> &x)
    {
        Array<float> y = x;
        for (size_t i = 0; i < x.dimension()[0]; ++i)
        {
            y = Decoder.forward(x);
            std::copy(y.begin(), y.end(), (res.begin() + i * 256));
        }
        return y;
    }

    float loss(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        predict_e(e);
        predict_d(x);
        return _loss.forward(res, t);
    }

    float gradient(const Array<float> &e, const Array<float> &x, const Array<float> &t)
    {
        float y = loss(e, x, t);
        Array<float> g = _loss.backward();
        for (size_t i = g.dimension()[0] - 1; i < g.dimension()[0]; --i)
            Decoder.backward(g.cut({i}));

        for (size_t i = e.dimension()[0] - 1; i < e.dimension()[0]; --i)
            Encoder.backward(Array<float>());

        return y;
    }

    void update(const float lr)
    {
        Decoder.update(lr);
        Encoder.update(lr);
    }
};