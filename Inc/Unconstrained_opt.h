//
// Created by 翰樵 on 2021/4/30.
//

#ifndef OPERATIONRESEARCH_UNCONSTRAINED_OPT_H
#define OPERATIONRESEARCH_UNCONSTRAINED_OPT_H
#include "OptCore.h"
#include <vector>


#define maxn 100
using namespace autodiff;
using namespace Eigen;
namespace numOpt{

enum StepLengthMethod {
    /** 斐波那契法 */
    Fibonacci   =   1,
    /** 黄金分割比法 */
    GoldenRatio =   2,
    /** Wolf_Powell法 */
    Wolf_Powell = 3,
    /** Goldstein法 */
    Goldstein = 4
};
enum SearchDirectionMethod {
    /** 最速梯度下降法*/
    SteepestDescent     =   1,
    /** 牛顿法*/
    Newton              =   2,
    /** 拟牛顿法*/
    Quasi_Newton_DFP    =   3,
    Quasi_Newton_BFGS   =   4,
    /** 共轭梯度法*/
    Conjugate_Gradient  =   5,
};

    class LineSearch: public Optimizer{
        protected:

            struct stepLengthRec{
                double lowerR, upperR, lam1R, lam2R, f1R, f2R, widthR;
            };

            static std::vector<stepLengthRec>record;
            static int FibList[maxn];
            VectorXd P, G, G_next;
            MatrixXd H, H_next;
            Eigen::VectorXvar X_var, X_next_var;
            autodiff::var y, y_next;
            double lambda, gSquareNorm;
            /**@brief 根据精度需要， 生成Fibonacci数列. 要求F_{n+1}>[(b-a)/epsilon]
            *
            * @param eps 精度需要
            * @return 生成Fibonacci数列的长度
            */
            static int FibGen(double eps, double width);

            /**@brief 寻找最优步长时的目标函数
             *
             * @param lambda 步长
             * @return 取该步长时的目标函数值
             */
            inline double stepLengthFct(double lambda);

            /**@brief 确定方向和步长后更新下一步
             *
             */
            inline void genNextStep();

            /**@brief 数据类型转换， 将VectorXd类型转换为autodiff库中的VectorXvar类型， 以用于求导
             *
             * @param src：  Eigen库VectorXd数据类型， 表示用double存储的一个向量
             * @return Autodiff库VectorXvar数据类型
             */
            static VectorXvar dToVar(VectorXd &src);

            double stepLength(double lowerBound, double upperBound, unsigned int method, double eps);

        public:

            /**@brief 一维搜索： 在无约束规划问题中确定步长
             *
             * @param fct 目标函数
             * @param lowerBound 搜索上界
             * @param upperBound 搜索下界
             * @param method 搜索方式
             * @param eps 搜索精度
             * @return 最优精确步长
             * @enum numOpt::StepLengthMethod
             * @example
            @code
            #include "numOpt.h"
            using namespace numOpt;
            double func(double x) {return x*x-6*x+2;}

            int main{
                double stepLength = LineSearch::stepLength(func, 0, 10, Fibonacci, 1e-4);
            }
            @endcode
             >>> stepLength = 3.00347   minimum = -6.99999
             */
            static double stepLength(double(*fct)(double), double lowerBound, double upperBound, unsigned int method, double eps);

            /**@brief 最速下降法，牛顿法寻找搜索方向
             *
             * @param fct 目标函数
             * @param X0 当前位置
             * @param P 方向
             * @param g 当前位置目标函数的梯度
             * @param method 最速下降法， 牛顿法， 共轭梯度法
             * @return bool变量 方向搜索失败返回false
             * @example
            @code
#include "numOpt.h"
using namespace numOpt;
using namespace autodiff;
using namespace Eigen;

MatrixXvar Q(3, 3);
VectorXvar B(3);
var func(VectorXvar X){
    var a = B.transpose()*X;
    var b = 0.5*X.transpose()*Q*X;
    return a+b;
}
int main()
{

    VectorXd X(3);
    var d =1;
    X <<1, 1, -1;
    Q << 2, 0, 0,
        0, 8, 0,
        0, 0, 18;
    B << -2, 0, 18;


    VectorXd P, g;

    LineSearch::findDescentDirection(func, X, P, g, Newton);
    std::cout<<"descent direction: at["<<X.transpose()<<"]'    "
    <<"["<<P.transpose()<<"]"<<std::endl;
}
            @endcode
            >>>descent direction: at[ 1  1 -1]' :   [ 0 -1  0]'
             */
            bool findDescentDirection(unsigned int method);

            LineSearch(var(*_fct)(VectorXvar &), VectorXd &initialX);

            LineSearch(var(*_fct)(VectorXvar &), VectorXd &currentX, VectorXd &currentP, VectorXd &currentG
                       , MatrixXd &currentH);

            VectorXd descentD();
            VectorXd currentX();
            void Update();
            VectorXd NextX();
            double currentF();
        };
}
#endif //OPERATIONRESEARCH_UNCONSTRAINED_OPT_H
