#include "CppNN/Layer/Tanh.hpp"
#include "MNIST/mnist.hpp"

int main()
{
    mnist mnist;

    Array<float> t = mnist.get_img(0, TRAI);
    out(reshape({28,28},t));
}