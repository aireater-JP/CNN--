#pragma once
#include "Sigmoid.hpp"

template <typename T>
class LSTM
{
    Array<T> F;
    Array<T> G;
    Array<T> I;
    Array<T> O;

    Array<T> X;
    Array<T> H_prev;
    Array<T> C_prev;
    Array<T> H_next;
    Array<T> C_next;

    Array<T> dC_prev;
    Array<T> dH_prev;

    Array<T> &Wfx, Wfh, Bf, dWfx, dWfh, dBf;
    Array<T> &Wgx, Wgh, Bg, dWgx, dWgh, dBg;
    Array<T> &Wix, Wih, Bi, dWix, dWih, dBi;
    Array<T> &Wox, Woh, Bo, dWox, dWoh, dBo;

public:
    LSTM(Array<T> &Wfx, Array<T> &Wfh, Array<T> &Bf, Array<T> &dWfx, Array<T> &dWfh, Array<T> &dBf,
         Array<T> &Wgx, Array<T> &Wgh, Array<T> &Bg, Array<T> &dWgx, Array<T> &dWgh, Array<T> &dBg,
         Array<T> &Wix, Array<T> &Wih, Array<T> &Bi, Array<T> &dWix, Array<T> &dWih, Array<T> &dBi,
         Array<T> &Wox, Array<T> &Woh, Array<T> &Bo, Array<T> &dWox, Array<T> &dWoh, Array<T> &dBo)
        : Wfx(Wfx), Wfh(Wfh), Bf(Bf), dWfx(dWfx), dWfh(dWfh), dBf(dBf),
          Wgx(Wgx), Wgh(Wgh), Bg(Bg), dWgx(dWgx), dWgh(dWgh), dBg(dBg),
          Wix(Wix), Wih(Wih), Bi(Bi), dWix(dWix), dWih(dWih), dBi(dBi),
          Wox(Wox), Woh(Woh), Bo(Bo), dWox(dWox), dWoh(dWoh), dBo(dBo)
    {
    }

    Index initialize(const Index &input_dimension)
    {
    }

    Array<T> forward(const Array<T> &x, const Array<T> &h_prev, const Array<T> &c_prev)
    {
        X = x;
        C_prev = c_prev;
        H_prev = h_prev;

        F = sigmoid(dot(x, Wfx) + dot(h_prev, Wfh) + Bf);
        G = tanh(dot(x, Wgx) + dot(h_prev, Wgh) + Bg);
        I = sigmoid(dot(x, Wix) + dot(h_prev, Wih) + Bi);
        O = sigmoid(dot(x, Wox) + dot(h_prev, Woh) + Bo);

        C_next = F * c_prev + G * I;
        H_next = O * tanh(C_next);

        return H_next;
    }

    Array<T> backward(const Array<T> &dH_next, const Array<T> &dC_next)
    {
        Array<T> tanh_C_next = tanh(C_next);

        Array<T> dS = dC_next + (dH_next * O) * (1 - tanh_C_next * tanh_C_next);

        dC_prev = dS * F;

        Array<T> dI = dS * G * I * (1 - I);
        Array<T> dF = dS * C_prev * F * (1 - F);
        Array<T> dO = dH_next * tanh_C_next * O * (1 - O);
        Array<T> dG = dS * I * (1 - G * G);

        Array<T> HT = H_prev.Transpose();
        Array<T> XT = X.Transpose();

        dWfh += dot(HT, dF);
        dWfx += dot(XT, dF);
        dBf += dF.sum(1);

        dWih += dot(HT, dI);
        dWix += dot(XT, dI);
        dBi += dI.sum(1);

        dWoh += dot(HT, dO);
        dWox += dot(XT, dO);
        dBo += dO.sum(1);

        dWgh += dot(HT, dG);
        dWgx += dot(XT, dG);
        dBg += dG.sum(1);

        Array<T> dX = dot(dF, Wfx.Transpose()) + dot(dI, Wix.Transpose()) + dot(dO, Wox.Transpose()) + dot(dG, Wgx.Transpose());
        dH_prev = dot(dF, Wfh.Transpose()) + dot(dI, Wih.Transpose()) + dot(dO, Woh.Transpose()) + dot(dG, Wgh.Transpose());

        return dX;
    }

    Array<T> get_c()
    {
        return C_next;
    }

    Array<T> get_dh()
    {
        return dH_prev;
    }

    Array<T> get_dc()
    {
        return dC_prev;
    }
};