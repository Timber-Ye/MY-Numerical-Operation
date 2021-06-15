//
// Created by 翰樵 on 2021/4/30.
//

#include "Inc/Unconstrained_opt.h"
#define EPS 1e-9
#define FOR_DEBUG 0
using namespace std;

namespace numOpt{


int LineSearch::FibList[maxn] = {0};
vector<LineSearch::stepLengthRec>LineSearch::record;

LineSearch::LineSearch(var(*_fct)(VectorXvar &), VectorXd &initialX){
    Obj_fct = _fct;
    X = initialX;
    dimension = X.rows();
    H = MatrixXd::Identity(dimension, dimension);
    X_var = dToVar(X);
    y = (*Obj_fct)(X_var);
    G = gradient(y, X_var);
    it_counter = 0;lambda=0; gSquareNorm=0;
}
LineSearch::LineSearch(var(*_fct)(VectorXvar &), VectorXd &currentX, VectorXd &currentP, VectorXd &currentG, MatrixXd &currentH){
    Obj_fct = _fct;
    X = currentX;
    dimension = X.rows();
    it_counter = 0;
    P = currentP;
    G = currentG;
    H = currentH;
    lambda=0; gSquareNorm=0;
}

int LineSearch::FibGen(const double eps, double width){
    int i=0;
    do{
        if(i==0||i==1) FibList[i] = 1;
        else{
            FibList[i] = FibList[i-1]+FibList[i-2];
        }
    }while(FibList[i++]<width*(1/eps));
    return i-2>=3? i-2:1 ;
}

double LineSearch::stepLength(double(*fct)(double), double lowerBound, double upperBound, unsigned int method, double eps){
    double a = lowerBound, b = upperBound;
    double lambda1, lambda2, l1, l2;
    double upper = b, lower = a;
    double ans, temp;
    double width = b-a;

    //初始第一步
    if(method == Fibonacci) {
        int n = FibGen(eps, width); //Fibonacci法n必须要找准， 不然结果会存在较大误差。
        lambda1 = (double) FibList[n - 1] / FibList[n + 1] * (b - a) + a,
                lambda2 = a + (double) FibList[n] / FibList[n + 1] * (b - a);
    }else if(method == GoldenRatio)
        lambda1 = 0.382*(b-a)+a, lambda2 = a+0.618*(b-a);

    l1 = (*fct)(lambda1), l2 = (*fct)(lambda2);

    while(width>eps*(b-a)){

#if FOR_DEBUG
        record.emplace_back(stepLengthRec{lower, upper, lambda1, lambda2, l1, l2, width});
#endif

        if(l2-l1>EPS){
            upper = lambda2;
            if(method==Fibonacci) {
                temp = lambda2;
                lambda2 = lambda1;
                lambda1 = temp - lambda2+lower;
            }else if(method==GoldenRatio) {
                lambda2 = lambda1;
                lambda1 = lower + 0.382*(upper - lower);
            }
            l2 = l1;
            l1 = (*fct)(lambda1);
        }else{
            lower = lambda1;
            if(method==Fibonacci) {
                temp = lambda1;
                lambda1 = lambda2;
                lambda2 = upper-(lambda1-temp);
            }
            else if(method==GoldenRatio) {
                lambda1 = lambda2;
                lambda2 = lower+0.618*(upper-lower);
            }
            l1 = l2;
            l2 = (*fct)(lambda2);
        }
        width = upper-lower;
    }

    ans = (upper+lower)/2;
    return ans;
}

double LineSearch::stepLength(double lowerBound, double upperBound, unsigned int method, double eps){

    double a = lowerBound, b = upperBound;
    double lambda1, lambda2, l1, l2;
    double upper = b, lower = a;
    double ans, temp;
    double width = b-a;

    //初始第一步
    if(method == Fibonacci) {
        int n = FibGen(eps, width); //Fibonacci法n必须要找准， 不然结果会存在较大误差。
        lambda1 = (double) FibList[n - 1] / FibList[n + 1] * (b - a) + a,
                lambda2 = a + (double) FibList[n] / FibList[n + 1] * (b - a);
    }else if(method == GoldenRatio)
        lambda1 = 0.382*(b-a)+a, lambda2 = a+0.618*(b-a);

    l1 = stepLengthFct(lambda1), l2 = stepLengthFct(lambda2);

    while(width>eps*(b-a)){

#if FOR_DEBUG
        record.emplace_back(stepLengthRec{lower, upper, lambda1, lambda2, l1, l2, width});
#endif

        if(l2-l1>EPS){
            upper = lambda2;
            if(method==Fibonacci) {
                temp = lambda2;
                lambda2 = lambda1;
                lambda1 = temp - lambda2+lower;
            }else if(method==GoldenRatio) {
                lambda2 = lambda1;
                lambda1 = lower + 0.382*(upper - lower);
            }
            l2 = l1;
            l1 = stepLengthFct(lambda1);
        }else{
            lower = lambda1;
            if(method==Fibonacci) {
                temp = lambda1;
                lambda1 = lambda2;
                lambda2 = upper-(lambda1-temp);
            }
            else if(method==GoldenRatio) {
                lambda1 = lambda2;
                lambda2 = lower+0.618*(upper-lower);
            }
            l1 = l2;
            l2 = stepLengthFct(lambda2);
        }
        width = upper-lower;
    }

    ans = (upper+lower)/2;
    return ans;
}

bool LineSearch::findDescentDirection(unsigned int method){
//    assert(method==SteepestDescent||method==Newton);
    bool FLAG = false;
    gSquareNorm = G.squaredNorm();
    if(gSquareNorm<1e-5){
        std::cout<<std::endl<<"Local Minimum Arrived!"<<std::endl
                 << "# Local Minimum Pos:    [" << X.transpose() << "]'" << std::endl
                 << "# Local Minimum Value:  " << y.expr->val << std::endl
                 << "# gSquareNorm:          " << gSquareNorm << std::endl
                 << "# Iteration Times:      " << it_counter << std::endl;
        return false;
    }

    if(method==SteepestDescent) {
        P = -G;
        lambda = this->stepLength(0, 5, Fibonacci, 1e-4);
        genNextStep();
        FLAG = true;

    }
    else if(method==Newton){
        MatrixXd H_inv;
        H = hessian(y, X_var);
        H_inv = H.inverse();
        P = -H_inv*G;
        lambda = 1;
        genNextStep();
        FLAG = true;
    }
    else if(method==Quasi_Newton_DFP){
        if(H.isIdentity()) P = -G;
        else P = -H*G;

        lambda = this->stepLength(0, 3, GoldenRatio, 1e-4);
        genNextStep();

        FLAG = true;
        VectorXd Y, S;
        S = X_next-X; Y = G_next-G;
        H_next = H-H*Y*Y.transpose()*H/(Y.transpose()*H*Y)+S*S.transpose()/(S.transpose()*Y);

    }
    else if(method==Conjugate_Gradient){
        if(H.isIdentity()) P = -G;
        else{
            double tempH = H(0,0);
            VectorXd temp_P = -G+tempH*P;
            P = temp_P;
        }
        lambda = this->stepLength(0, 3, GoldenRatio, 1e-4);
        genNextStep();
        FLAG = true;

        H_next = G_next.transpose()*(G_next-G)/(-P.transpose()*G);
    }
    return FLAG;
}

VectorXvar LineSearch::dToVar(VectorXd &src){
    int len = src.rows();
    VectorXvar X(len);
    for(int i=0;i<len;i++){
        X(i)= src(i);
    }
    return X;
}

inline double LineSearch::stepLengthFct(double l){
    VectorXvar x_var= dToVar(X), P_var= dToVar(P);
    VectorXvar temp = x_var+l*P_var;
    var ans = (*Obj_fct)(temp);
    return ans.expr->val;
}

inline void LineSearch::genNextStep(){
    X_next = X+P*lambda;
    X_next_var = dToVar(X_next);
    y_next = (*Obj_fct)(X_next_var);
    G_next = gradient(y_next, X_next_var);
#if FOR_DEBUG
    std::cout<<std::endl<<"///////////////Iteration #"<<it_counter<<"///////////////"<<std::endl
                        <<"Current Position  X("<<it_counter<<"): ["<<X.transpose()<<"]'"<<std::endl
                        <<"Objective Value   f("<<it_counter<<"): "<<y<<std::endl
                        <<"Current Gradient  g("<<it_counter<<"): ["<<G.transpose()<<"]'   ||g||= "<<gSquareNorm<<std::endl
                        <<"descent direction P("<<it_counter<<"): ["<<P.transpose()<<"]"<<std::endl
                        <<"Step Length  lambda("<<it_counter<<"): "<<lambda<<std::endl;

#endif
    it_counter++;
}

VectorXd LineSearch::descentD(){return P;}
VectorXd LineSearch::currentX(){return X;}
double LineSearch::currentF(){return y.expr->val;}
VectorXd LineSearch::NextX(){return X_next;}
void LineSearch::Update(){X = X_next;G = G_next;H = H_next;y = y_next;}
}