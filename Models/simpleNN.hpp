#include "../CppNN/CppNN.hpp"
#include "../MNIST/mnist.hpp"

CppNN simpleNN(){
    CppNN NN;

    NN.add_Layer(Affine<float>(256));
    NN.add_Layer(ReLU<float>());
    NN.add_Layer(Affine<float>(64));
    NN.add_Layer(ReLU<float>());
    NN.add_Layer(Affine<float>(10));

    NN.set_Loss(Softmax_with_Loss<float>());

    return NN;
}