//
// Created by 翰樵 on 2021/4/28.
//
#include "Inc/numOpt.h"
#include "Inc/heuOpt.h"
#define dimMENSION 4
using namespace numOpt;
using namespace autodiff;
using namespace Eigen;

VectorXd X0(dimMENSION);
MatrixXvar Q(dimMENSION, dimMENSION);
VectorXvar B(dimMENSION);
MatrixXd Q1(dimMENSION, dimMENSION); // 启发式算法不要使用autodiff库中的var， 否则计算会很慢
VectorXd B1(dimMENSION);
var func(VectorXvar &X){
    var a = B.transpose()*X;
    var b = 0.5*X.transpose()*Q*X;
    return a+b;
}
var f1(VectorXvar &X){
    var a = ((B-Q*X).transpose())*(B-Q*X);
    return a;
}
double f2(VectorXd &X){
    double a = ((B1-Q1*X).transpose())*(B1-Q1*X);
    return a;
}
int main()
{
    clock_t start, end;
//    double stepLength = LineSearch::stepLength(ywss, 0, 10, Fibonacci, 1e-4);
//    std::cout<<"stepLength: "<<stepLength<<"     "<<ywss(stepLength)<<std::endl;


//    VectorXd X0(3);
//    var d =1;
//    X0<<1, 1, -1;
//    Q << 2, 0, 0,
//        0, 8, 0,
//        0, 0, 18;
//    B << -2, 0, 18;
//
//    VectorXd P, g;
//    LineSearch UnconstrainedOPT(func, X0);
//    UnconstrainedOPT.findDescentDirection(Newton);
////    std::cout << "# Current position: [" << UnconstrainedOPT.currentX().transpose() << "]'" << std::endl
////              << "# descent direction: [" << UnconstrainedOPT.descentD().transpose() << "]'" << std::endl
////              << "# Next position: [" << UnconstrainedOPT.NextX().transpose() << "]'" << std::endl;
//
//    UnconstrainedOPT.Update();
//    bool flag = UnconstrainedOPT.findDescentDirection(Newton);
//    if(flag == true){
//        std::cout << "# Current position: [" << UnconstrainedOPT.currentX().transpose() << "]'" << std::endl
//                  << "# descent direction: [" << UnconstrainedOPT.descentD().transpose() << "]'" << std::endl
//                << "# Next position: [" << UnconstrainedOPT.NextX().transpose() << "]'" << std::endl;
//    }
//    X0<<-2, 4;
//    Q << 3, -1,
//        -1, 1;
//    B << -2, 0;
    X0 << 100, -10, 10, -1000;
    Q1 << 1, 2, 3, 4,
        2, 4, 7, 5,
        6, 7, 3, 2,
        1, 8, 4, 2;
    B1 << 6, 4, 3, 8;
    Q = Q1;
    B = B1;

    LineSearch UnconstrainedOPT(f1, X0);
    bool flag = true;
    int count = 0;
    start = clock();
    while(count<20&& flag){
        flag = UnconstrainedOPT.findDescentDirection(Quasi_Newton_DFP);
        UnconstrainedOPT.Update();
        count++;
    }
    end = clock();
    std::cout<<"Processing Time:"<<(double)(end-start)/CLOCKS_PER_SEC<<std::endl;
    if(flag){
        std::cout << std::endl << "Fail to arrive at a Local Minimum!"<<std::endl
                  << "# Current position:  [" << UnconstrainedOPT.currentX().transpose() << "]'" << std::endl
                  << "# descent direction: [" << UnconstrainedOPT.descentD().transpose() << "]'" << std::endl
                  << "# Next position:     [" << UnconstrainedOPT.NextX().transpose() << "]'" << std::endl
                  << "# Objective Value:    " << UnconstrainedOPT.currentF() << std::endl
                  << "# Iteration Times:    " << count << std::endl;
    }

    heuOpt::simulated_Annealing S(f2, X0);
    start = clock();
    S.solve();
    end = clock();
    std::cout<<"Processing Time:"<<(double)(end-start)/CLOCKS_PER_SEC<<std::endl;

}
