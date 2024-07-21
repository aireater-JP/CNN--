#include "CppNN/Loss.hpp"

int main(){
    Identity_with_Loss<float> test;
    
    Array<float> x({10});
    x[0]=0.1; 
    x[1]=0.05; 
    x[2]=0.6; 
    x[3]=0.0; 
    x[4]=0.05; 
    x[5]=0.1; 
    x[6]=0.0; 
    x[7]=0.1; 
    x[8]=0.0; 
    x[9]=0.0; 

    Array<float> t({10});
    t[7]=1;

    test.forward(x,t);
    out(test.backward());
}