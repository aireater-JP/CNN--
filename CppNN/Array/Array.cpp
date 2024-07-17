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
Array<T>::Array(const Array &array) : _start(0), _dimension(array._dimension), _stride(array._stride), _size(array._size), _data(new T[_size])
{
    std::copy(array.begin(), array.end(), begin());
}

template <typename T>
Array<T>::Array(Array &&array) noexcept : _start(array._start), _dimension(std::move(array._dimension)), _stride(std::move(array._stride)), _size(array._size), _data(std::move(array._data))
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
        _start = 0;
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
        _start = array._start;
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

    // 一番後の次元が0の場合は0
    if (dimension[dimension.size() - 1] == 0)
        return res;

    res[res.size() - 1] = 1;

    for (size_t i = 1; i <= res.size() - 1; ++i)
        res[res.size() - 1 - i] = res[res.size() - i] * dimension[dimension.size() - i];

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
            throw std::out_of_range("Index out of range");
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
        throw std::out_of_range("Index out of range");

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
            throw std::logic_error("ブロードキャストできないお");

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
            throw "計算できないよ";

        *std::find(idx.begin(), idx.end(), 0) = _size / temp;

        _dimension = idx;
        _stride = calculate_stride(_dimension);
        return;
    }
    if (_size != calculate_size(index))
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
Array<T> Array<T>::share(const Index &index) const
{
    if (index.size() == 0)
    {
        Array<T> res;
        res._start = _start;
        res._dimension = _dimension;
        res._stride = _stride;
        res._size = _size;
        res._data = _data;

        return res;
    }

    size_t start = 0;
    Index idx(_dimension.size() - index.size());
    for (size_t i = 0; i < index.size(); ++i)
    {
        if (index[i] >= _dimension[i])
            throw "エラーだお";
        start += index[i] * _stride[i];
    }
    for (size_t i = 0; i < idx.size(); ++i)
        idx.back_access(i) = _dimension.back_access(i);

    Array<T> res;
    res._start = start;
    res._dimension = idx;
    res._stride = res.calculate_stride(idx);
    res._size = res.calculate_size(idx);
    res._data = _data;

    return res;
}

template <typename T>
Array<T> Array<T>::Transpose()
{
    if (_dimension.size() != 2)
        throw "計算できないよ!";

    Array res({_dimension[1], _dimension[0]});

    for (size_t i = 0; i < _dimension[0]; ++i)
        for (size_t j = 0; j < _dimension[1]; ++j)
            res[{j, i}] = operator[]({i, j});

    return res;
}

template <typename T>
Array<T> Array<T>::sum(const size_t axis)
{
    Index dim = _dimension;
    dim[axis] = 1;
    Array<T> res(dim);

    for (size_t i = 0; i < _size; ++i)
    {
        Index idx = calculate_one_to_mul(i);
        res[res.broadcast_to_Index(idx)] += operator[](idx);
    }

    return res;
}