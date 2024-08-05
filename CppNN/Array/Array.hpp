#pragma once

#include "Index.hpp"

template <typename T>
class Array
{
    using iterator = T *;
    using const_iterator = const T *;

private:
    Index _dimension;
    Index _stride;
    size_t _size;
    std::shared_ptr<T[]> _data;

public:
    //--------------------------------------------------------------
    // コンストラクタ
    // デフォルトコンストラクタ
    Array() : _dimension(), _stride(), _size(0), _data(nullptr) {}

    // 初期化リストによる初期化
    Array(std::initializer_list<size_t> dimension) : Array(Index(dimension)) {};

    // 次元を指定して初期化
    Array(const Index &dimension) : _dimension(dimension), _stride(calculate_stride(_dimension)), _size(calculate_size(_dimension)), _data(new T[_size]()) {}
    Array(Index &&dimension) noexcept : _dimension(std::move(dimension)), _stride(calculate_stride(_dimension)), _size(calculate_size(_dimension)), _data(new T[_size]()) {}

    // 次元と初期値を指定して初期化
    Array(const Index &dimension, const T &value);
    Array(Index &&dimension, const T &value) noexcept;

    // コピーコンストラクタ
    Array(const Array &array);
    // ムーブコンストラクタ
    Array(Array &&array) noexcept;

    //--------------------------------------------------------------
    // 代入演算子
    Array &operator=(const Array &array);
    Array &operator=(Array &&array) noexcept;

    //--------------------------------------------------------------
    size_t size() const { return _size; }
    Index dimension() const { return _dimension; }

    //--------------------------------------------------------------
    // イテレータ
    iterator begin() { return _data.get(); }
    iterator end() { return _data.get() + _size; }

    const_iterator begin() const { return _data.get(); }
    const_iterator end() const { return _data.get() + _size; }

    //--------------------------------------------------------------
    // 要素アクセス
    T &operator[](const Index &index) { return _data[calculate_mul_to_one(index)]; }
    const T &operator[](const Index &index) const { return _data[calculate_mul_to_one(index)]; }

    T &operator[](const size_t &index) { return _data[index]; }
    const T &operator[](const size_t &index) const { return _data[index]; }

    //--------------------------------------------------------------
    // 比較演算子
    bool operator==(const Array &array) const { return (_dimension == array._dimension) and (std::equal(begin(), end(), array.begin(), array.end())); };
    bool operator!=(const Array &array) const { return !(*this == array); };

    //--------------------------------------------------------------
    // ブロードキャストを含む四則演算
    template <typename U>
    friend Array<U> operator+(const Array<U> &x, const Array<U> &y);
    template <typename U>
    friend Array<U> operator-(const Array<U> &x, const Array<U> &y);
    template <typename U>
    friend Array<U> operator*(const Array<U> &x, const Array<U> &y);
    template <typename U>
    friend Array<U> operator/(const Array<U> &x, const Array<U> &y);
    template <typename U>
    friend Array<U> operator%(const Array<U> &x, const Array<U> &y);
    template <typename U>
    friend Array<U> operator^(const Array<U> &x, const Array<U> &y);

    Array &operator+=(const Array &x)
    {
        *this = *this + x;
        return *this;
    }
    Array &operator-=(const Array &x)
    {
        *this = *this - x;
        return *this;
    }
    Array &operator*=(const Array &x)
    {
        *this = *this * x;
        return *this;
    }
    Array &operator/=(const Array &x)
    {
        *this = *this / x;
        return *this;
    }
    Array &operator^=(const Array &x)
    {
        *this = *this ^ x;
        return *this;
    }

    // 出力
    template <typename U>
    friend void out(const Array<U> &array);

    void reshape(const Index &index);
    Array cut(const Index &index = {}) const;
    void copy(const Array &array, const Index &index);

    Array Transpose();
    Array sum(const size_t axis) const;
    Array max(const size_t axis) const;

    void clear() { std::fill(begin(), end(), 0); }

private:
    //--------------------------------------------------------------
    // 次元の積を計算
    Index calculate_stride(const Index &dimension) const;
    // 要素数を計算
    size_t calculate_size(const Index &dimension) const;
    // 多次元を1次元に変換
    size_t calculate_mul_to_one(const Index &index) const;
    // 距離から添字
    Index calculate_one_to_mul(const size_t index) const;
    // ブロードキャスト
    Index broadcast_to_Index(const Index &index) const;
};

#include "Array.cpp"
#include "4_arithmetic_operations.cpp"
#include "Array_calculate.cpp"

// 出力関数
template <typename T>
void out(const Array<T> &array)
{
    const Index &str = array._stride;

    // 1次元配列用
    if (str.size() == 1)
    {
        out("[");
        for (size_t i = 0; i < array.size(); ++i)
        {
            out(array[i]);
            if (array.size() == i + 1)
            {
                out("]\n\n");
                return;
            }
            out(",");
        }
    }

    // 2次元以上
    size_t level = 0;
    size_t interval = str.back_access(1);

    for (size_t i = 0; i < array.size(); ++i)
    {
        // 行始め
        if (i % interval == 0)
        {
            size_t temp_level = level;
            for (size_t j = 0; j < str.size(); ++j)
                if (j >= level)
                {
                    out("[");
                    temp_level++;
                }
                else
                {
                    out(" ");
                }

            level = temp_level;
        }

        out(array[i]);

        if (i % interval == interval - 1)
        {
            for (size_t j = 1; j < str.size(); ++j)
                if ((i + 1) % str.back_access(j) == 0)
                {
                    out("]");
                    level--;
                }

            if (i + 1 == array.size())
            {
                out("]\n\n");
                return;
            }
            out(",\n");

            for (size_t j = 2; j < str.size(); ++j)
                if ((i + 1) % str.back_access(j) == 0)
                    out("\n");

            continue;
        }
        out(",");
    }
}