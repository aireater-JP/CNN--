#pragma once
#include "../Loss.hpp"

#include <cfloat>

template <typename T>
class Softmax_with_Loss : public Loss<T>
{
    Array<T> Softmax_output_cash;
    Array<T> teacher_cash;

public:
    T forward(const Array<T> &x, const Array<T> &teacher) override
    {
        teacher_cash = teacher;
        Softmax_output_cash = Softmax(x);

        return cross_entropy_error(Softmax_output_cash, teacher);
    }

    Array<T> backward() override
    {
        return Softmax_output_cash - teacher_cash;
    }

private:
    Array<T> Softmax(const Array<T> &x)
    {
        Array<T> exp_temp = exp(x - x.max(0));
        return exp_temp / exp_temp.sum(0);
    }

    T cross_entropy_error(const Array<T> &x, const Array<T> &teacher)
    {
        return -sum(teacher * log(x + T(FLT_MIN))) / (x.size() / x.dimension().back_access(0));
    }
};