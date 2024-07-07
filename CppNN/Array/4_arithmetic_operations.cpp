#include "Array.hpp"

//--------------------------------------------------------------
// 行列と行列
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
//--------------------------------------------------------------
// 行列とスカラー
//--------------------------------------------------------------
template <typename T>
Array<T> operator+(const Array<T> &x, const T y)
{
    Array<T> res(x);
    for (const auto &i : res)
        i += y;

    return res;
}

template <typename T>
Array<T> operator-(const Array<T> &x, const T y)
{
    Array<T> res(x);
    for (const auto &i : res)
        i -= y;

    return res;
}

template <typename T>
Array<T> operator*(const Array<T> &x, const T y)
{
    Array<T> res(x);
    for (const auto &i : res)
        i *= y;

    return res;
}

template <typename T>
Array<T> operator/(const Array<T> &x, const T y)
{
    Array<T> res(x);
    for (const auto &i : res)
        i /= y;

    return res;
}

template <typename T>
Array<T> operator%(const Array<T> &x, const T y)
{
    Array<T> res(x);
    for (const auto &i : res)
        i %= y;

    return res;
}

template <typename T>
Array<T> operator^(const Array<T> &x, const T y)
{
    Array<T> res(x.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        i = std::pow(i, y);

    return res;
}
//--------------------------------------------------------------
// スカラーと行列
//--------------------------------------------------------------
template <typename T>
Array<T> operator+(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension(), x);
    for (size_t i = 0; i < res.size(); ++i)
        res[i] += y[i];

    return res;
}

template <typename T>
Array<T> operator-(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension(), x);
    for (size_t i = 0; i < res.size(); ++i)
        res[i] -= y[i];

    return res;
}

template <typename T>
Array<T> operator*(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension(), x);
    for (size_t i = 0; i < res.size(); ++i)
        res[i] *= y[i];

    return res;
}

template <typename T>
Array<T> operator/(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension(), x);
    for (size_t i = 0; i < res.size(); ++i)
        res[i] /= y[i];

    return res;
}

template <typename T>
Array<T> operator%(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension(), x);
    for (size_t i = 0; i < res.size(); ++i)
        res[i] %= y[i];

    return res;
}

template <typename T>
Array<T> operator^(const T x, const Array<T> &y)
{
    Array<T> res(y.dimension());
    for (size_t i = 0; i < res.size(); ++i)
        res[i] = std::pow(x, y[i]);

    return res;
}