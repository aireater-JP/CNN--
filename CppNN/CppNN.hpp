#include "Layer.hpp"
#include "Loss.hpp"
#include "Random.hpp"

class CppNN
{
    std::vector<std::unique_ptr<Layer<float>>> _layer;
    std::unique_ptr<Loss<float>> _loss;

    bool is_initialized = false;

public:
    template <class T>
    void add_Layer(T &&l)
    {
        _layer.emplace_back(std::make_unique<T>(std::move(l)));
        is_initialized = false;
    }

    template <class T>
    void set_Loss(T &&l)
    {
        _loss = std::make_unique<T>(std::move(l));
        is_initialized = false;
    }

    void initialize(const Index &input_size)
    {
        Index x = input_size;
        for (auto &i : _layer)
            x = i->initialize(x);

        is_initialized = true;
    }

    Array<float> predict(const Array<float> &x)
    {
        if (is_initialized == false)
            throw std::invalid_argument("initializeを呼んでね");

        Array<float> y = x;
        for (auto &i : _layer)
            y = i->forward(y);

        return y;
    }

    float loss(const Array<float> &x, const Array<float> &t)
    {
        return _loss->forward(predict(x), t);
    }

    float gradient(const Array<float> &x, const Array<float> &t, const float lr)
    {
        float y = loss(x, t);
        Array<float> g = _loss->backward();

        for (size_t i = _layer.size() - 1; i < _layer.size(); i--)
            g = _layer[i]->backward(g);

        for (auto &i : _layer)
            i->update(lr);

        return y;
    }
};