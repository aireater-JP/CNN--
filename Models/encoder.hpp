#pragma once
#include "../CppNN/CppNN_Time.hpp"

class encoder
{
    CppNN_Time NN;
    Time_LSTM<float> rnn;

    size_t current;

public:
    encoder(size_t affine_size, size_t hidden_size) : rnn(hidden_size)
    {
        NN.add_Layer(Time_Affine<float>(affine_size));
        NN.add_Layer(Time_ReLU<float>());
    }

    Index initialize(const Index &input_size)
    {
        return rnn.initialize(NN.initialize(input_size));
    }

    Array<float> predict(const Array<float> &x)
    {
        Array<float> y = rnn.forward(NN.predict(x));

        current++;
        return y;
    }

    Array<float> gradient(const Array<float> &dy)
    {
        current--;

        return NN.gradient(rnn.backward(dy));
    }

    void update(const float lr)
    {
        NN.update(lr);
        rnn.update(lr);
    }

    void reset()
    {
        NN.reset();
        rnn.reset();
    }

    Array<float> get_h()
    {
        return rnn.get_h();
    }

    void set_dh(const Array<float> &dh)
    {
        rnn.set_dh(dh);
    }
};