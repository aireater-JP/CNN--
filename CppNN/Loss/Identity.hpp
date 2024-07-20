#pragma once
#include "../Loss.hpp"

template <typename T>
class Identity_with_Loss : public Loss<T>
{
    Array<T> Identity_output_cash;
    Array<T> teacher_cash;

public:
    T forward(const Array<T> &x, const Array<T> &teacher) override
    {
        teacher_cash = teacher;
        Identity_output_cash = x;
        return sum_of_squared_error(x, teacher);
    }

    Array<T> backward() override
    {
        return Identity_output_cash - teacher_cash;
    }

private:
    T sum_of_squared_error(const Array<T> &x, const Array<T> &teacher)
    {
        T y = sum((x - teacher)^2.f);
        return (y * 0.5) / (x.size() / x.dimension().back_access(0));
    }
};