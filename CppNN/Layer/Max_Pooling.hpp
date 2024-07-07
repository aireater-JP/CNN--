#pragma once
#include "../Layer.hpp"

template <typename T>
class Max_Pooling : public Layer<T>
{
    Index _output_size;
    Array<Index> mask;

public:
    Index initialize(const Index &input_dimension) override
    {
        Index dim = input_dimension;
        if (dim.back_access(1) % 2 != 0 or dim.back_access(0) % 2 != 0)
            throw "計算できないよ";
        dim.back_access(1) /= 2;
        dim.back_access(0) /= 2;
        _output_size = dim;
        return dim;
    }

    Array<T> forward(const Array<T> &x) override
    {
        Array<T> reshape_x = reshape({-1, x.back_access(0)}, x);
        Array<T> y({reshape_x.dimension()[0] / 2, reshape_x.dimension()[1] / 2});

        for (size_t i = 0; i < reshape_x.dimension()[0] / 2; ++i)
            for (size_t j = 0; j < reshape_x.dimension()[1] / 2; ++j)
                for (size_t m = 0; m < 2; ++m)
                    for (size_t n = 0; n < 2; ++n)
                    {
                        size_t col = i * 2 + m;
                        size_t row = j * 2 + n;
                        if (y[{i, j}] > reshape_x[{col, row}])
                            continue;
                        y[{i, j}] = reshape_x[{col, row}];
                        mask[{i, j}] = {col, row};
                    }

        y.reshape(_output_size);
        return y;
    }

    Array<T> backward(const Array<T> &x) override
    {
        Array<T> reshape_x = reshape({-1, x.back_access(0)}, x);
        Array<T> y({reshape_x.dimension()[0] * 2, reshape_x.dimension()[1] * 2});

        for (size_t i = 0; i < reshape_x.dimension()[0] / 2; ++i)
            for (size_t j = 0; j < reshape_x.dimension()[1] / 2; ++j)
                y[{mask[{i, j}][0], mask[i, j][1]}] = reshape_x[{i, j}];

        return;
    }
};