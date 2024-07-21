#include "../../Layer.hpp"
#include "../Cell/LSTM.hpp"

template <typename T>
class Time_LSTM : public Layer<T>
{
    size_t current;

    Array<T> h, dh;
    Array<T> c, dc;

    Array<T> Wfx, Wfh, Bf;
    Array<T> Wgx, Wgh, Bg;
    Array<T> Wix, Wih, Bi;
    Array<T> Wox, Woh, Bo;

    Array<T> dWfx, dWfh, dBf;
    Array<T> dWgx, dWgh, dBg;
    Array<T> dWix, dWih, dBi;
    Array<T> dWox, dWoh, dBo;

    std::vector<LSTM<T>> lstms;

    size_t _input_size, _hidden_size;

public:
    Time_LSTM(const size_t time, const size_t hidden_size)
        : current(0),
          lstms(time, LSTM<T>(Wfx, Wfh, Bf, dWfx, dWfh, dBf, Wgx, Wgh, Bg, dWgx, dWgh, dBg, Wix, Wih, Bi, dWix, dWih, dBi, Wox, Woh, Bo, dWox, dWoh, dBo)),
          _hidden_size(hidden_size)
    {
    }

    Index initialize(const Index &input_dimension) override
    {
        _input_size = input_dimension.back_access(0);
        init();
        return {1, _hidden_size};
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> X = reshape({0, _input_size}, x);
        h = lstms[current].forward(X, h, c);
        c = lstms[current].get_c();
        current++;

        return h;
    }

    Array<T> backward(const Array<T> &dy) override
    {
        current--;
        Array<T> dx = lstms[current].backward(dy + dh, dc);
        dh = lstms[current].get_dh();
        dc = lstms[current].get_dc();

        return dx;
    }

    Array<T> get_c() { return c; }
    Array<T> get_dc() { return dc; }
    Array<T> get_h() { return h; }
    Array<T> get_dh() { return dh; }

    void set_c(const Array<T> &n) { c = n; }
    void set_dc(const Array<T> &n) { dc = n; }
    void set_h(const Array<T> &n) { h = n; }
    void set_dh(const Array<T> &n) { dh = n; }

    void update(const T lr) override
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

        dWfh.clear();
        dWih.clear();
        dWoh.clear();
        dWgh.clear();

        dWfx.clear();
        dWix.clear();
        dWox.clear();
        dWgx.clear();

        dBf.clear();
        dBi.clear();
        dBo.clear();
        dBg.clear();
    }

    void reset()
    {
        current = 0;
        h.clear();
        c.clear();
        dh.clear();
        dc.clear();
    }

private:
    void init()
    {
        h = Array<T>({1, _hidden_size});
        dh = Array<T>({1, _hidden_size});
        c = Array<T>({1, _hidden_size});
        dc = Array<T>({1, _hidden_size});
        Wfx = Array<T>({_input_size, _hidden_size});
        Wfh = Array<T>({_hidden_size, _hidden_size});
        Bf = Array<T>({_hidden_size});
        Wgx = Array<T>({_input_size, _hidden_size});
        Wgh = Array<T>({_hidden_size, _hidden_size});
        Bg = Array<T>({_hidden_size});
        Wix = Array<T>({_input_size, _hidden_size});
        Wih = Array<T>({_hidden_size, _hidden_size});
        Bi = Array<T>({_hidden_size});
        Wox = Array<T>({_input_size, _hidden_size});
        Woh = Array<T>({_hidden_size, _hidden_size});
        Bo = Array<T>({_hidden_size});

        dWfx = Array<T>({_input_size, _hidden_size});
        dWfh = Array<T>({_hidden_size, _hidden_size});
        dBf = Array<T>({_hidden_size});
        dWgx = Array<T>({_input_size, _hidden_size});
        dWgh = Array<T>({_hidden_size, _hidden_size});
        dBg = Array<T>({_hidden_size});
        dWix = Array<T>({_input_size, _hidden_size});
        dWih = Array<T>({_hidden_size, _hidden_size});
        dBi = Array<T>({_hidden_size});
        dWox = Array<T>({_input_size, _hidden_size});
        dWoh = Array<T>({_hidden_size, _hidden_size});
        dBo = Array<T>({_hidden_size});

        Random<std::uniform_real_distribution<>> r(-1.0, 1.0);
        for (auto &i : Wfh)
            i = r();
        for (auto &i : Wih)
            i = r();
        for (auto &i : Woh)
            i = r();
        for (auto &i : Wgh)
            i = r();
        for (auto &i : Wfx)
            i = r();
        for (auto &i : Wix)
            i = r();
        for (auto &i : Wox)
            i = r();
        for (auto &i : Wgx)
            i = r();
    }
};