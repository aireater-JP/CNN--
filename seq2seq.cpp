#include "Models/seq2seq.hpp"

int main()
{
    seq2seq NN(7, 5, 12, 12);

    Random<std::uniform_int_distribution<>> r(1, 9);

    NN.initialize({0});

    for (size_t i = 0; i < 100; ++i)
    {
        Array<float> e({7, 12});
        int ans = 0, p = 100;
        for (size_t i = 0; i < 3; ++i)
        {
            size_t R = r();
            e[{i, R}] = 1;
            ans += R * p;
            p /= 10;
        }
        e[{3, 10}] = 1;
        p = 100;
        for (size_t i = 0; i < 3; ++i)
        {
            size_t R = r();
            e[{i + 4, R}] = 1;
            ans += R * p;
            p /= 10;
        }
        Array<float> a({5, 12});
        p = 1000;
        for (size_t i = 0; i < 4; ++i)
        {
            a[{i, size_t(ans / p)}] = 1;
            ans %= p;
            p /= 10;
        }
        a[{4, 10}] = 1;

        out(NN.gradient(e, Array<float>(), a), "\n");
        NN.update(0.01);
    };
}