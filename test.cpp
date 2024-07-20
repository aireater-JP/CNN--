#include "Models/simpleNN.hpp"

int main()
{
    mnist mnist;

    CppNN NN = simpleNN();

    Random<std::uniform_int_distribution<>> r(0, 60000 - 1);

    NN.initialize({10,MNIST_size});

    for (size_t i = 0; i < 100; ++i)
    {
        Array<float> x({10, MNIST_size});
        Array<float> t({10});

        for (size_t j = 0; j < 10; ++j)
        {
            int R=r();   
            x.share({j})=mnist.get_img(R,TRAI);
            t[j]=mnist.get_label(R,TRAI);
        }

        out(t);

        out(NN.loss(x,t));

        NN.gradient(x,t);
        NN.update(0.01);
    }
}