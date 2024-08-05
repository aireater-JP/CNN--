#include "../../Layer.hpp"

template <typename T>
class WeightSum
{
    Array<T> a_cash;
    Array<T> dhs;

public:
    WeightSum(const Index &hs_size) : dhs(hs_size) {}

    Array<T> forward(const Array<T> &hs, const Array<T> &a)
    {
        Array<T> c(a.dimension());

        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
                c[j] += hs[{i, j}] * a[j];

        a_cash = a;

        return c;
    }

    Array<T> backward(const Array<T> &hs, const Array<T> &dc)
    {
        Array<T> da(dc.dimension());
        for (size_t i = 0; i < hs.dimension()[0]; ++i)
            for (size_t j = 0; j < hs.dimension()[1]; ++j)
            {
                da[j] += hs[{i, j}] * dc[j];
                dhs[{i, j}] = a_cash[j] * dc[j];
            }

        return da;
    }

    Array<T> get_dhs() { return dhs; }
};