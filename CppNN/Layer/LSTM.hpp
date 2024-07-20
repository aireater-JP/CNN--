#pragma once
#include "../Layer.hpp"

#include "Tanh.hpp"
#include "Sigmoid.hpp"

template <typename T>
class LSTM
{
    Array<T> Wfx, Wfh, Bf, dWfx, dWfh, dBf;
    Array<T> Wgx, Wgh, Bg, dWgx, dWgh, dBg;
    Array<T> Wix, Wih, Bi, dWix, dWih, dBi;
    Array<T> Wox, Woh, Bo, dWox, dWoh, dBo;

    Array<T> F;
    Array<T> G;
    Array<T> I;
    Array<T> O;

    Array<T> X;
    Array<T> H_prev;
    Array<T> C_prev;
    Array<T> C_next;

public:
    LSTM() : {}

    Index initialize(const Index &input_dimension) override
    {
    }

    Array<T> forward(const Array<T> &x, const Array<T> &h_prev, const Array<T> &c_prev) override
    {
        F = sigmoid(dot(x, Wfx) + dot(h_prev, Wfh) + Bf);
        G = tanh(dot(x, Wgx) + dot(h_prev, Wgh) + Bg);
        I = sigmoid(dot(x, Wix) + dot(h_prev, Wih) + Bi);
        O = sigmoid(dot(x, Wox) + dot(h_prev, Woh) + Bo);

        C_next = F * c_prev + G * I;
        Array<T> H_next = O * tanh(C_next);

        X = x;
        C_prev = c_prev;
    }

    Array<T> backward(const Array<T> &dH_next, const Array<T> &dC_next) override
    {
        Array<T> tanh_C_next = tanh(C_next);

        Array<T> dS = dC_next + (dH_next * O) * (1 - tanh_C_next * tanh_C_next);

        Array<T> dC_prev = dS * F;

        Array<T> dI = dS * G * I * (1 - I);
        Array<T> dF = dS * C_prev * F * (1 - F);
        Array<T> dO = dH_next * tanh_C_next * O * (1 - O);
        Array<T> dG = dS * I * (1 - G * G);

        Array<T> HT = H_prev.Transpose();
        Array<T> XT = X.Transpose();

        dWfh = dot(HT, dF);
        dWfx = dot(XT, dF);
        dBf = dF.sum(0);

        dWih = dot(HT, dI);
        dWix = dot(XT, dI);
        dBi = dI.sum(0);

        dWoh = dot(HT, dO);
        dWox = dot(XT, dO);
        dBo = dO.sum(0);

        dWgh = dot(HT, dG);
        dWgx = dot(XT, dG);
        dBg = dG.sum(0);

        Array<T> dX = dot(dF, Wfx.Transpose()) + dot(dI, Wix.Transpose()) + dot(dO, Wox.Transpose()) + dot(dG, Wgx.Transpose());
        Array<T> dH = dot(dF, Wfh.Transpose()) + dot(dI, Wih.Transpose()) + dot(dO, Woh.Transpose()) + dot(dG, Wgh.Transpose());
    }

    void update(const T lr)
    {
        Wfh = Wfh - dWfh * lr;
        Wih = Wih - dWih * lr;
        Woh = Woh - dWoh * lr;
        Wgh = Wgh - dWgh * lr;

        Wfx = Wfx - dWfx * lr;
        Wix = Wix - dWix * lr;
        Wox = Wox - dWox * lr;
        Wgx = Wgx - dWgx * lr;

        Bf = Bf - dBf * lr;
        Bi = Bi - dBi * lr;
        Bo = Bo - dBo * lr;
        Bg = Bg - dBg * lr;
    }
};