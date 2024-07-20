#pragma once

#include <random>

template <class distribution>
class Random
{
    std::mt19937 engine;
    distribution dist;

public:
    Random(double a = 0.0, double b = 0.0) : engine(std::random_device()()), dist(a, b) {}

    double operator()()
    {
        return dist(engine);
    }

    void set(double a, double b)
    {
        class distribution::param_type param(a, b);
        dist.param(param);
    }
};