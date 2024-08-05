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
Array<T> tanh(const Array<T> &x)
{
    Array<T> res(x.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = std::tanh(x[i]);

    return res;
}

template <typename T>
Array<T> log(const Array<T> &x)
{
    Array<T> res(x.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = std::log(x[i]);

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

template <typename T>
Array<T> dot(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension().size() != 2 or y.dimension().size() != 2 or x.dimension()[1] != y.dimension()[0])
        throw "計算できません";
    size_t _I = x.dimension()[0];
    size_t _J = x.dimension()[1];
    size_t _K = y.dimension()[1];
    Array<T> res({_I, _K});

    for (size_t i = 0; i < _I; ++i)
        for (size_t k = 0; k < _K; ++k)
            for (size_t j = 0; j < _J; ++j)
                res[{i, k}] += x[{i, j}] * y[{j, k}];

    return res;
}