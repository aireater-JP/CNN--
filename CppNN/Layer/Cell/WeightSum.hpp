#include "../../Layer.hpp"

template <typename T>
class WeightSum
{
    Array<T> hs_cash, a_cash;

    Array<T> da(a_cash.dimension());
    Array<T> dhs(hs_cash.dimension());

public:
    Array<T> forward(const Array<T> &hs, const Array<T> &a) override
    {
        Array<T> c({a.size()});

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
                c[j] += hs[{i, j}] * a[j];

        hs_cash = hs;
        a_cash = a;

        return c;
    }

    Array<T> backward(const Array<T> &dc) override
    {
        for (size_t i = 0; i < hs_cash.dimension()[0]; ++i)
            for (size_t j = 0; j < hs_cash.dimension()[1]; ++j)
            {
                da[j] += hs_cash[{i, j}] * dc[j];
                dhs[{i, j}] = a_cash[j] * dc[j];
            }

        return dhs;
    }
};