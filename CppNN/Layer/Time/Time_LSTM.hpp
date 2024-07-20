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

    size_t _input_size, _output_size;

public:
    Time_LSTM(const size_t time, const size_t input_size, const size_t output_size)
        : current(0),
          h({1, output_size}), dh({1, output_size}), c({1, output_size}), dc({1, output_size}),
          Wfx({input_size, output_size}), Wfh({output_size, output_size}), Bf({output_size}),
          Wgx({input_size, output_size}), Wgh({output_size, output_size}), Bg({output_size}),
          Wix({input_size, output_size}), Wih({output_size, output_size}), Bi({output_size}),
          Wox({input_size, output_size}), Woh({output_size, output_size}), Bo({output_size}),

          dWfx({input_size, output_size}), dWfh({output_size, output_size}), dBf({output_size}),
          dWgx({input_size, output_size}), dWgh({output_size, output_size}), dBg({output_size}),
          dWix({input_size, output_size}), dWih({output_size, output_size}), dBi({output_size}),
          dWox({input_size, output_size}), dWoh({output_size, output_size}), dBo({output_size}),
          lstms(time, LSTM<T>(Wfx, Wfh, Bf, dWfx, dWfh, dBf, Wgx, Wgh, Bg, dWgx, dWgh, dBg, Wix, Wih, Bi, dWix, dWih, dBi, Wox, Woh, Bo, dWox, dWoh, dBo)),
          _input_size(input_size), _output_size(output_size)
    {
    }

    Index initialize(const Index &input_dimension) override
    {
        init();
        return {input_dimension[0], _output_size};
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

    void reset()
    {
        current = 0;
        h.clear();
        c.clear();
        dh.clear();
        dc.clear();
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

private:
    void init()
    {
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