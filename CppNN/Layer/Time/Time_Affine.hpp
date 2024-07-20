#include "../../Layer.hpp"
#include "../Cell/LSTM.hpp"

template <typename T>
class Time_Affine : public Layer<T>
{
    Array<T> W;
    Array<T> dW;
    Array<T> B;
    Array<T> dB;

    size_t _output_size;

    std::vector<Affine<T>> affines;

    size_t current;

public:
    Time_Affine(const size_t time, const size_t output_size) : affines(time, (W, dW, B, dB)), current(0) {}

    Index initialize(const Index &input_dimension) override
    {
        W = Array<T>({input_dimension.back_access(0), _output_size});
        B = Array<T>({_output_size});

        Random<std::uniform_real_distribution<>> r(-1.0, 1.0);
        for (auto &i : W)
            i = r();

        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> X = reshape({0, _input_size}, x);
        Array<T> y = lstms[current].forward(X);
        current++;
        return y;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        Array<T> dx = affines[current].backward(dy);
        return dx;
    }

    void update(const T lr) override
    {
        W -= dW * lr;
        B -= dB * lr;

        dW.clear();
        dB.clear();
    }
};