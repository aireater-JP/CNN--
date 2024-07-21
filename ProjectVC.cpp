#include "Models/Attention.hpp"

const size_t f_a_s = 100;

int main()
{
    CppNN_Time e_in;
    CppNN_Time e_out;
    CppNN_Time d_in;
    CppNN_Time d_out;

    e_in.add_Layer(Time_Affine<float>(f_a_s));
    e_in.add_Layer(Time_ReLU<float>());
    d_in.add_Layer(Time_Affine<float>(f_a_s));
    d_in.add_Layer(Time_ReLU<float>());

    d_out.add_Layer(Time_Affine<float>(f_a_s));
    d_out.add_Layer(Time_ReLU<float>());

    Attention attention(e_in, e_out, d_in, d_out, 10, 10, 100, 10);
    attention.set_Loss(Identity_with_Loss<float>());
}