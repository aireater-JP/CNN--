#include "Array.hpp"

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
template <typename T>
Array<T>::Array(const Index &dimension, const T &value) : Array(dimension)
{
    std::fill(begin(), end(), value);
}

template <typename T>
Array<T>::Array(Index &&dimension, const T &value) noexcept : Array(dimension)
{
    std::fill(begin(), end(), value);
}

template <typename T>
Array<T>::Array(const Array &array) : _dimension(array._dimension), _stride(array._stride), _size(array._size), _data(new T[_size])
{
    std::copy(array.begin(), array.end(), begin());
}

template <typename T>
Array<T>::Array(Array &&array) noexcept : _dimension(std::move(array._dimension)), _stride(std::move(array._stride)), _size(array._size), _data(std::move(array._data))
{
    array._size = 0;
}

//--------------------------------------------------------------
// 代入演算子
//--------------------------------------------------------------
template <typename T>
Array<T> &Array<T>::operator=(const Array &array)
{
    if (this != &array)
    {
        _dimension = array._dimension;
        _stride = array._stride;
        _size = array._size;
        _data.reset(new T[_size]);
        std::copy(array.begin(), array.end(), begin());
    }
    return *this;
}

template <typename T>
Array<T> &Array<T>::operator=(Array &&array) noexcept
{
    if (this != &array)
    {
        _dimension = std::move(array._dimension);
        _stride = std::move(array._stride);
        _size = array._size;
        _data = std::move(array._data);

        array._size = 0;
    }
    return *this;
}

//--------------------------------------------------------------
// 内部で使用
//--------------------------------------------------------------
// アクセス用の次元の積を取る
template <typename T>
Index Array<T>::calculate_stride(const Index &dimension) const
{
    Index res(dimension.size());

    res.back_access(0) = 1;

    for (size_t i = 1; i <= res.size() - 1; ++i)
        res[res.size() - i - 1] = res[res.size() - i] * dimension[dimension.size() - i];

    return res;
}
//--------------------------------------------------------------
// 要素数計算
template <typename T>
size_t Array<T>::calculate_size(const Index &dimension) const
{
    size_t res = 1;
    for (const size_t &i : dimension)
        res *= i;

    return res;
}
//--------------------------------------------------------------
// 添字から距離
template <typename T>
size_t Array<T>::calculate_mul_to_one(const Index &index) const
{
    size_t res = 0;
    for (size_t i = 0; i < _dimension.size(); ++i)
    {
        if (index[i] >= _dimension[i])
            throw "範囲外アクセス";
        res += index[i] * _stride[i];
    }
    return res;
}
//--------------------------------------------------------------
// 距離から添字
template <typename T>
Index Array<T>::calculate_one_to_mul(const size_t index) const
{
    size_t idx = index;
    if (idx >= _size)
        throw "範囲外アクセス";

    Index res(_dimension.size());

    for (size_t i = 0; i < res.size(); ++i)
    {
        res[i] = idx / _stride[i];
        idx %= _stride[i];
    }
    return res;
}
//--------------------------------------------------------------
// ブロードキャスト
template <typename T>
Index Array<T>::broadcast_to_Index(const Index &index) const
{
    Index res(_dimension.size());
    for (size_t i = 0; i < res.size(); ++i)
        if (_dimension.back_access(i) != 1)
            res.back_access(i) = index.back_access(i);

    return res;
}
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
            throw "ブロードキャストに対応していません";

        res.back_access(i) = std::max(x_temp, y_temp);
    }
    return res;
}
//--------------------------------------------------------------
// 変形
//--------------------------------------------------------------
template <typename T>
void Array<T>::reshape(const Index &index)
{
    // 未知数が1つの場合
    if (std::count(index.begin(), index.end(), 0) == 1)
    {
        Index idx = index;
        size_t temp = 1;
        for (const size_t &i : idx)
            if (i != 0)
                temp *= i;

        if (_size % temp != 0)
            throw "変形できません";

        *std::find(idx.begin(), idx.end(), 0) = _size / temp;
    }
    else if (_size != calculate_size(index))
        throw "変形先と要素数が一致しません";

    _dimension = index;
    _stride = calculate_stride(_dimension);
}

template <typename T>
Array<T> reshape(const Index &index, const Array<T> &array)
{
    Array<T> temp = array;
    temp.reshape({index});
    return temp;
}
//--------------------------------------------------------------
// 変形
//--------------------------------------------------------------
template <typename T>
Array<T> Array<T>::cut(const Index &index) const
{
    size_t start = 0;
    Index idx(_dimension.size() - index.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] >= _dimension[i])
            throw "要素が大きすぎます";
        start += index[i] * _stride[i];
    }
    for (size_t i = 0; i < idx.size(); ++i)
        idx.back_access(i) = _dimension.back_access(i);

    Array<T> res;
    res._dimension = idx;
    res._stride = res.calculate_stride(idx);
    res._size = res.calculate_size(idx);
    res._data.reset(new T[res._size]);
    std::copy((begin() + start), (begin() + start + res._size), res.begin());

    return res;
}

template <typename T>
void Array<T>::copy(const Array<T> &array, const Index &index)
{
    size_t start = 0;
    for (size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] >= _dimension[i])
            throw "要素が大きすぎます";
        start += index[i] * _stride[i];
    }

    if (_dimension.size() - index.size() == array._dimension.size())
        throw "配列が適合していません";

    for (size_t i = 0; i < _dimension.size() - index.size(); ++i)
        if (_dimension[index.size() + i] != array._dimension[i])
            throw "配列が適合してません";

    std::copy(array.begin(), array.end(), begin() + start);
}

template <typename T>
Array<T> Array<T>::Transpose()
{
    if (_dimension.size() != 2)
        throw "要素数が多すぎます";

    Array res({_dimension[1], _dimension[0]});

    for (size_t i = 0; i < _dimension[0]; ++i)
        for (size_t j = 0; j < _dimension[1]; ++j)
            res[{j, i}] = *this[{i, j}];

    return res;
}

template <typename T>
Array<T> Array<T>::sum(const size_t axis) const
{
    Index dim = _dimension;
    dim.back_access(axis) = 1;
    Array<T> res(dim);

    for (size_t i = 0; i < _size; ++i)
    {
        Index idx = calculate_one_to_mul(i);
        res[res.broadcast_to_Index(idx)] += *this[idx];
    }

    return res;
}

template <typename T>
Array<T> Array<T>::max(const size_t axis) const
{
    Index dim = _dimension;
    dim.back_access(axis) = 1;
    Array<T> res(dim);

    for (size_t i = 0; i < _size; ++i)
    {
        Index idx = calculate_one_to_mul(i);
        res[res.broadcast_to_Index(idx)] = std::max(res[res.broadcast_to_Index(idx)], *this[idx]);
    }

    return res;
}

template <typename T>
T sum(const Array<T> &x)
{
    T res = 0;
    for (const auto &i : x)
        res += i;
    return res;
}

template <typename T>
T max(const Array<T> &x)
{
    T res = 0;
    for (const auto &i : x)
        res = std::max(res, i);
    return res;
}