#include "Array.hpp"

//--------------------------------------------------------------
// ブロードキャスト後の次元を求める
Index broadcast_shape(const Index &x, const Index &y)
{
    size_t r_size = std::max(x.size(), y.size());
    Index res(r_size);

    for (size_t i = 0; i < r_size; ++i)
    {
        size_t x_temp = (i < x.size()) ? x.back_access(i) : 1;
        size_t y_temp = (i < y.size()) ? y.back_access(i) : 1;

        if (x_temp != y_temp and x_temp != 1 and y_temp != 1)
            throw std::logic_error("ブロードキャストできないお");

        res.back_access(i) = std::max(x_temp, y_temp);
    }
    return res;
}

//--------------------------------------------------------------
template <typename T>
Array<T> operator+(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x);
        for (size_t i = 0; i < x.size(); ++i)
            res[i] += y[i];

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = x[x.broadcast_to_Index(idx)] + y[y.broadcast_to_Index(idx)];
    }
    return res;
}

template <typename T>
Array<T> operator-(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x);
        for (size_t i = 0; i < x.size(); ++i)
            res[i] -= y[i];

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = x[x.broadcast_to_Index(idx)] - y[y.broadcast_to_Index(idx)];
    }
    return res;
}

template <typename T>
Array<T> operator*(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x);
        for (size_t i = 0; i < x.size(); ++i)
            res[i] *= y[i];

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = x[x.broadcast_to_Index(idx)] * y[y.broadcast_to_Index(idx)];
    }
    return res;
}

template <typename T>
Array<T> operator/(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x);
        for (size_t i = 0; i < x.size(); ++i)
            res[i] /= y[i];

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = x[x.broadcast_to_Index(idx)] / y[y.broadcast_to_Index(idx)];
    }
    return res;
}

template <typename T>
Array<T> operator%(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x);
        for (size_t i = 0; i < x.size(); ++i)
            res[i] /= y[i];

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = x[x.broadcast_to_Index(idx)] % y[y.broadcast_to_Index(idx)];
    }
    return res;
}

template <typename T>
Array<T> operator^(const Array<T> &x, const Array<T> &y)
{
    if (x.dimension() == y.dimension())
    {
        Array<T> res(x.dimension());
        for (size_t i = 0; i < x.size(); ++i)
            res[i] = std::pow(x[i], y[i]);

        return res;
    }
    // ブロードキャスト適応して計算
    Index shape = broadcast_shape(x.dimension(), y.dimension());
    Array<T> res(shape);

    for (size_t i = 0; i < res.size(); ++i)
    {
        Index idx = res.calculate_one_to_mul(i);
        res[idx] = std::pow(x[x.broadcast_to_Index(idx)], y[y.broadcast_to_Index(idx)]);
    }
    return res;
}

