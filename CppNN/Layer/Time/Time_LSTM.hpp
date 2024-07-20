#include "../../Time_Layer.hpp"
#include "../Cell/LSTM.hpp"

template <typename T>
class Time_LSTM
{
    std::unique_ptr<LSTM<T>[]> lstms;

    size_t current;

    Array<T> h, dh;
    Array<T> c, dc;

    Array<T> Wfx, Wfh, Bf, dWfx, dWfh, dBf;
    Array<T> Wgx, Wgh, Bg, dWgx, dWgh, dBg;
    Array<T> Wix, Wih, Bi, dWix, dWih, dBi;
    Array<T> Wox, Woh, Bo, dWox, dWoh, dBo;

    size_t _output_size;

public:
    Time_LSTM(const size_t time, const size_t output_size)
        : lstms(new LSTM<T>[time](Wfx, Wfh, Bf, dWfx, dWfh, dBf, Wgx, Wgh, Bg, dWgx, dWgh, dBg, Wix, Wih, Bi, dWix, dWih, dBi, Wox, Woh, Bo, dWox, dWoh, dBo)),
          _output_size(output_size)
    {
    }

    Index initialize(const Index &input_dimension)
    {
        Wfx = Array<T>({input_dimension.back_access(0), _output_size}), Wfh = Array<T>({_output_size, _output_size}), Bf = Array<T>({_output_size});
        Wgx = Array<T>({input_dimension.back_access(0), _output_size}), Wgh = Array<T>({_output_size, _output_size}), Bg = Array<T>({_output_size});
        Wix = Array<T>({input_dimension.back_access(0), _output_size}), Wih = Array<T>({_output_size, _output_size}), Bi = Array<T>({_output_size});
        Wox = Array<T>({input_dimension.back_access(0), _output_size}), Woh = Array<T>({_output_size, _output_size}), Bo = Array<T>({_output_size});

        dWfx = Array<T>({input_dimension.back_access(0), _output_size}), dWfh = Array<T>({_output_size, _output_size}), dBf = Array<T>({_output_size});
        dWgx = Array<T>({input_dimension.back_access(0), _output_size}), dWgh = Array<T>({_output_size, _output_size}), dBg = Array<T>({_output_size});
        dWix = Array<T>({input_dimension.back_access(0), _output_size}), dWih = Array<T>({_output_size, _output_size}), dBi = Array<T>({_output_size});
        dWox = Array<T>({input_dimension.back_access(0), _output_size}), dWoh = Array<T>({_output_size, _output_size}), dBo = Array<T>({_output_size});

        return {input_dimension[0], _output_size};
    }

    Array<T> forward(const Array<T> &x)
    {
        h = lstms[current]->forward(x, h, c);
        c = lstms[current]->get_c();
        current++;

        return h;
    }

    Array<T> backward(const Array<T> &dy)
    {
        Array<T> dx = lstms[current]->backward(dy + dh, dc);
        dh = lstms[current]->get_dh();
        dc = lstms[current]->get_dc();
        current--;
        
        return dx;
    }

    void update(const T lr)
    {
        Wfh -= dWfh * lr;
        Wih -= dWih * lr;
        Woh -= dWoh * lr;
        Wgh -= dWgh * lr;

        Wfx -= dWfx * lr;
        Wix -= dWix * lr;
        Wox -= dWox * lr;
        Wgx -= dWgx * lr;

        Bf -= dBf * lr;
        Bi -= dBi * lr;
        Bo -= dBo * lr;
        Bg -= dBg * lr;
    }
};