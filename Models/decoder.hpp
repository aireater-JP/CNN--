#pragma once
#include "../CppNN/CppNN_Time.hpp"

class decoder
{
    CppNN_Time NN_in;
    Time_LSTM<float> rnn;
    CppNN_Time NN_out;

    size_t current;

public:
    decoder(size_t hidden_size) : rnn(hidden_size)
    {}

    Index initialize(const Index &input_size)
    {
        return NN_out.initialize(rnn.initialize(NN_in.initialize(input_size)));
    }

    Array<float> predict(const Array<float> &x)
    {
        Array<float> y = NN_out.predict(rnn.forward(NN_in.predict(x)));

        current++;
        return y;
    }

    Array<float> gradient(const Array<float> &dx)
    {
        current--;

        return NN_in.gradient(rnn.backward(NN_out.gradient(dx)));
    }

    void update(const float lr)
    {
        NN_in.update(lr);
        rnn.update(lr);
        NN_out.update(lr);
    }
};