#include "Index.hpp"

Index::Index(const size_t size, const size_t value) : Index(size)
{
    std::fill(begin(), end(), value);
}

Index::Index(std::initializer_list<size_t> index) : Index(index.size())
{
    std::copy(index.begin(), index.end(), begin());
}

Index::Index(const Index &index) : Index(index.size())
{
    std::copy(index.begin(), index.end(), begin());
}

Index::Index(Index &&index) noexcept : _size(index._size), _data(std::move(index._data))
{
    index._size = 0;
}

Index &Index::operator=(const Index &index)
{
    if (this != &index)
    {
        _size = index._size;
        _data.reset(new size_t[_size]);
        std::copy(index.begin(), index.end(), begin());
    }
    return *this;
}

Index &Index::operator=(Index &&index) noexcept
{
    if (this != &index)
    {
        _size = index._size;
        _data = std::move(index._data);

        index._size = 0;
    }
    return *this;
}

size_t &Index::operator[](const size_t index)
{
    if (index >= _size)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[index];
}

const size_t Index::operator[](const size_t index) const
{
    if (index >= _size)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[index];
}

size_t &Index::back_access(const size_t index)
{
    if (_size - 1 - index >= _size)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_size - 1 - index];
}

const size_t Index::back_access(const size_t index) const
{
    if (_size - 1 - index >= _size)
    {
        throw std::out_of_range("Index out of range");
    }
    return _data[_size - 1 - index];
}