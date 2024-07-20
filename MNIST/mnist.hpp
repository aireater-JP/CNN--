#pragma once

#include "../CppNN/Array/Array.hpp"
#include <fstream>

// 784
const int MNIST_size = 28 * 28;

typedef enum
{
    TRAI,
    TEST
} MNIST_GET_TYPE;

class mnist
{
    Array<float> trai_label;
    Array<float> trai_img;

    Array<float> test_label;
    Array<float> test_img;

public:
    mnist()
    {
        trai_label = load_label(60000, "MNIST/train-labels.idx1-ubyte");
        trai_img = load_img(60000, "MNIST/train-images.idx3-ubyte");

        test_label = load_label(10000, "MNIST/t10k-labels.idx1-ubyte");
        test_img = load_img(10000, "MNIST/t10k-images.idx3-ubyte");
    }

    float get_label(const size_t id, const MNIST_GET_TYPE t) const
    {
        if (t == TRAI)
            return trai_label[id];
        return test_label[id];
    }

    Array<float> get_img(const size_t id, const MNIST_GET_TYPE t)
    {
        if (t == TRAI)
            return trai_img.share({id});
        return test_img.share({id});
    }

private:
    Array<float> load_label(const size_t size, const std::string &name)
    {
        std::ifstream ifs;
        ifs.open(name, std::ios::binary);

        ifs.seekg(8, std::ios::beg);

        Array<unsigned char> temp({size});
        Array<float> res({size});

        ifs.read(reinterpret_cast<char *>(&temp[0]), size);

        for (size_t i = 0; i < size; ++i)
        {
            res[i] = float(temp[i]);
        }

        ifs.close();

        return res;
    }

    Array<float> load_img(const size_t size, const std::string &name)
    {
        std::ifstream ifs;
        ifs.open(name, std::ios::binary);

        ifs.seekg(16, std::ios::beg);

        Array<unsigned char> temp({size, MNIST_size});
        Array<float> res({size, MNIST_size});

        ifs.read(reinterpret_cast<char *>(&temp[0]), size * MNIST_size);

        float _255 = 1.f / 255.f;
        for (size_t i = 0; i < size * MNIST_size; ++i)
        {
            res[i] = float(temp[i]) * _255;
        }

        ifs.close();
        return res;
    }
};