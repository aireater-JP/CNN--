#include "../../Layer.hpp"

template <typename T>
class AttentionWeight
{
    Array<T> hs_cash, h_cash;

    Array<T> dhs(hs_cash.dimension());
    Array<T> dh(h_cash.dimension());

public:
    Array<T> forward(const Array<T> &hs, const Array<T> &h) override
    {
        Array<T> a({h.size()});

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
                a[j] += hs[{i, j}] * h[j];

        return a;
    }

    Array<T> backward(const Array<T> &da) override
    {
        for (size_t i = 0; i < hs_cash.dimension()[0]; ++i)
            for (size_t j = 0; j < hs_cash.dimension()[1]; ++j)
            {
                dhs[{i, j}] = a[j] * dh[j];
                dh[j] += a[j] * dhs[{i, j}];
            }

        return dhs;
    }
};