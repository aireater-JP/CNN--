#include "Array.hpp"

template <typename T>
Array<T> exp(const Array<T> &x)
{
    Array<T> res(x.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = std::exp(x[i]);

    return res;
}

template <typename T>
Array<T> operator+(const Array<T> &x)
{
    return x;
}

template <typename T>
Array<T> operator-(const Array<T> &x)
{
    Array<T> res(x.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = -x[i];

    return res;
}