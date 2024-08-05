#include "Models/seq2seq.hpp"

int main()
{
    CppNN_Time test;
    test.add_Layer(Time_LSTM<float>(10));
    test.add_Layer(Time_Affine<float>(1));

    test.initialize({100, 1});

    Identity_with_Loss<float> loss;

    Random<std::uniform_real_distribution<>> r(0, 3.14);

    Array<float> res({100, 1});
    Array<float> tea({100, 1});

    for (size_t k = 0; k < 30; ++k)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            float s = r();
            for (size_t j = 0; j < 100; ++j)
            {
                tea[j] = std::sin(j / 100 + s);
                res.copy(test.predict(tea.cut({j})), {j});
            }

            out(loss.forward(res, tea));

            for (size_t j = 99; j < 100; --j)
            {
                test.gradient(loss.backward());
            }
            test.update(0.01);
        }
    }
}