//
// Created by 翰樵 on 2021/5/7.
//

#ifndef OPERATIONRESEARCH_OPTCORE_H
#define OPERATIONRESEARCH_OPTCORE_H
#include<ctime>
#include <iostream>
#include "autodiff/reverse.hpp"
#include <autodiff/reverse/eigen.hpp>

class Optimizer{
protected:
    int dimension;
    long long it_counter;
    autodiff::var(*Obj_fct)(Eigen::VectorXvar &);
    double (*Obj_Fct)(Eigen::VectorXd &);
    Eigen::VectorXd X, X_next;
};

#endif //OPERATIONRESEARCH_OPTCORE_H
