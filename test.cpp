#include "Models/simpleNN.hpp"

int main()
{
    mnist mnist;

    CppNN NN = simpleNN();

    Random<std::uniform_int_distribution<>> r(0, 60000 - 1);

    NN.initialize({10, MNIST_size});

    for (size_t i = 0; i < 100; ++i)
    {
        Array<float> x({10, MNIST_size});
        Array<float> t({10, 10}, 0);

        for (size_t j = 0; j < 10; ++j)
        {
            int R = r();
            Array<float> temp = mnist.get_img(R, TRAI);
            std::copy(temp.begin(), temp.end(), (x.begin() + j * MNIST_size));
            t[{j, mnist.get_label(R, TRAI)}] = 1;
        }

        out(NN.loss(x, t), "\n");

        NN.gradient(x, t, 0.01);
    };
}