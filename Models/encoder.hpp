#pragma once
#include "../CppNN/CppNN_Time.hpp"

class encoder
{
    CppNN_Time NN;
    Time_LSTM<float> rnn;

    Array<float> hs;
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

        std::copy(y.begin(), y.end(), hs.begin() + current * y.size());

        current++;
        return y;
    }

    Array<float> gradient(const Array<float> &dh)
    {
        current--;

        return NN.gradient(rnn.backward(dh));
    }

    void update(const float lr)
    {
        NN.update(lr);
        rnn.update(lr);
    }
};