#include "../../Layer.hpp"

template <typename T>
class AttentionWeight
{
    Array<T> h_cash;
    Array<T> dhs, dh;
    
    Softmax<T> s;

public:
    AttentionWeight(const Index &hs_s, const Index &h_s) : dhs(hs_s), dh(h_s) {}

    Array<T> forward(const Array<T> &hs, const Array<T> &h)
    {
        Array<T> a({h.size()});

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
                a[j] += hs[{i, j}] * h[j];

        h_cash = h;

        return s.forward(a);
    }

    Array<T> backward(const Array<T> &hs, const Array<T> &da)
    {
        Array<float> ds = s.backward(da);

        for (size_t i = 0; i < h_cash.dimension()[0]; ++i)
            for (size_t j = 0; j < h_cash.dimension()[1]; ++j)
            {
                dhs[{i, j}] = ds[j] * h_cash[j];
                dh[j] += ds[j] * hs[{i, j}];
            }
        return dh;
    }

    Array<T> get_dhs() { return dhs; }
};