#pragma once

#include <memory>
#include <initializer_list>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "out.hpp"
class Index
{
    using iterator = size_t *;
    using const_iterator = const size_t *;

private:
    size_t _size;
    std::unique_ptr<size_t[]> _data;

public:
    // デフォルトコンストラクタ
    Index() : _size(0), _data() {}

    Index(const size_t size) : _size(size), _data(new size_t[_size]()) {}
    Index(const size_t size, const size_t value);

    // 初期化リストで初期化
    Index(std::initializer_list<size_t> index);

    Index(const Index &index);
    Index(Index &&index) noexcept;

    // 代入演算子
    Index &operator=(const Index &index);
    Index &operator=(Index &&index) noexcept;

    size_t size() const { return _size; }

    // イテレータ
    iterator begin() { return _data.get(); }
    iterator end() { return _data.get() + _size; }

    const_iterator begin() const { return _data.get(); }
    const_iterator end() const { return _data.get() + _size; }

    // アクセス
    size_t &operator[](const size_t index);
    const size_t operator[](const size_t index) const;

    size_t &back_access(const size_t index);
    const size_t back_access(const size_t index) const;

    // 比較演算子
    bool operator==(const Index &index) const { return std::equal(begin(), end(), index.begin(), index.end()); };
    bool operator!=(const Index &index) const { return !(*this == index); };
};

#include "Index.cpp"

inline void out(const Index &x)
{
    out("[");
    for (size_t i = 0; i < x.size(); ++i)
    {
        out(x[i]);
        if (i + 1 == x.size())
        {
            break;
        }
        out(",");
    }
    out("]");
}