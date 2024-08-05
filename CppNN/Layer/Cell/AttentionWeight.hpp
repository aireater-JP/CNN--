#include "../../Layer.hpp"

template <typename T>
class AttentionWeight
{
    Array<T> h_cash;
    Array<T> dhs;

    Softmax<T> softm;

public:
    AttentionWeight(const Index &hs_size) : dhs(hs_size) {}

    Array<T> forward(const Array<T> &hs, const Array<T> &h)
    {
        Array<T> a(h.dimension());

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
                a[j] += hs[{i, j}] * h[j];

        h_cash = h;

        return softm.forward(a);
    }

    Array<T> backward(const Array<T> &hs, const Array<T> &da)
    {
        Array<T> ds = softm.backward(da);
        Array<T> dh(da.dimension());

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
            {
                dhs[{i, j}] = ds[j] * h_cash[j];
                dh[j] += ds[j] * hs[{i, j}];
            }
        return dh;
    }

    Array<T> get_dhs() { return dhs; }
};