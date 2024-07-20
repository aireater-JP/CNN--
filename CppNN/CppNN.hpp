#include "Layer.hpp"
#include "Loss.hpp"

template <typename T>
class CppNN
{
    std::vector<std::unique_ptr<Layer<T>[]>> _layer;
    std::unique_ptr<Loss<T>[]> _loss;

    bool is_initialized = false;

public:
    template <class _T>
    void add_Layer(_T &&l)
    {
        _layer.emplace_back(std::make_unique<_T>(std::move(l)));
        is_initialized = false;
    }

    template <class _T>
    void set_Loss(_T &&l)
    {
        _loss = std::make_unique<_T>(std::move(l));
        is_initialized = false;
    }

    void initialize(const Index &input_size)
    {
        Index x = input_size;
        for (auto &i : _layer)
            x = i->initialize(x);

        is_initialized = true;
    }

    Array<T> predict(const Array<T> &x)
    {
        if (is_initialized == false)
            throw std::invalid_argument("initializeを呼んでね");

        Array<T> y = x;
        for (auto &i : _layer)
            y = i->forward(y);
        return y;
    }

    T loss(const Array<T> &x, const Array<T> &t)
    {
        return _loss->forward(predict(x), t);
    }

    T gradient(const Array<T> &x, const Array<T> &t)
    {
        T y = loss(x, t);
        Array<T> g = _loss->backward();

        for (size_t i = _layer.size() - 1; i < _layer.size(); i--)
            g = _layer[i]->backward(g);

        return y;
    }

    void update(const T lr)
    {
        for (auto &i : _layer)
            i->update(lr);
    }
};