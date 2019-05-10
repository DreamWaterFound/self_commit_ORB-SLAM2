/**
 * @file PnPsolver.cc
 * @author guoqing (1337841346@qq.com)
 * @brief EPnP 相机位姿求解器
 * @version 0.1
 * @date 2019-05-08
 * 
 * @copyright Copyright (c) 2019
 * 
 */

/**
* This file is part of ORB-SLAM2.
* This file is a modified version of EPnP <http://cvlab.epfl.ch/EPnP/index.php>, see FreeBSD license below.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

/**
* Copyright (c) 2009, V. Lepetit, EPFL
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
*    list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
* ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The views and conclusions contained in the software and documentation are those
* of the authors and should not be interpreted as representing official policies,
*   either expressed or implied, of the FreeBSD Project
*/

//这里的pnp求解用的是EPnP的算法。
// 参考论文：EPnP:An Accurate O(n) Solution to the PnP problem
// https://en.wikipedia.org/wiki/Perspective-n-Point
// http://docs.ros.org/fuerte/api/re_vision/html/classepnp.html
// 如果不理解，可以看看中文的："摄像机位姿的高精度快速求解" "摄像头位姿的加权线性算法"

// PnP求解：已知世界坐标系下的3D点与图像坐标系对应的2D点，求解相机的外参(R t)，即从世界坐标系到相机坐标系的变换。
// 而EPnP的思想是：
// 将世界坐标系所有的3D点用四个虚拟的控制点来表示，将图像上对应的特征点转化为相机坐标系下的四个控制点
// 根据世界坐标系下的四个控制点与相机坐标系下对应的四个控制点（与世界坐标系下四个控制点有相同尺度）即可恢复出(R t)


//                                   |x|
//   |u|   |fx r  u0||r11 r12 r13 t1||y|
// s |v| = |0  fy v0||r21 r22 r23 t2||z|
//   |1|   |0  0  1 ||r32 r32 r33 t3||1|

// step1:用四个控制点来表达所有的3D点
// p_w = sigma(alphas_j * pctrl_w_j), j从0到4
// p_c = sigma(alphas_j * pctrl_c_j), j从0到4
// sigma(alphas_j) = 1,  j从0到4

// step2:根据针孔投影模型
// s * u = K * sigma(alphas_j * pctrl_c_j), j从0到4

// step3:将step2的式子展开, 消去s
// sigma(alphas_j * fx * Xctrl_c_j) + alphas_j * (u0-u)*Zctrl_c_j = 0
// sigma(alphas_j * fy * Xctrl_c_j) + alphas_j * (v0-u)*Zctrl_c_j = 0

// step4:将step3中的12未知参数（4个控制点*3维参考点坐标）提成列向量
// Mx = 0,计算得到初始的解x后可以用Gauss-Newton来提纯得到四个相机坐标系的控制点

// step5:根据得到的p_w和对应的p_c，最小化重投影误差即可求解出R t



#include <iostream>

#include "PnPsolver.h"

#include <vector>
#include <cmath>
#include <opencv2/core/core.hpp>
#include "Thirdparty/DBoW2/DUtils/Random.h"
#include <algorithm>

using namespace std;

namespace ORB_SLAM2
{

// 在大体的pipeline上和Sim3Solver差不多,都是 构造->设置RANSAC参数->外部调用迭代函数,进行计算->得到计算的结果

// pcs表示3D点在camera坐标系下的坐标
// pws表示3D点在世界坐标系下的坐标
// us表示图像坐标系下的2D点坐标
// alphas为真实3D点用4个虚拟控制点表达时的系数
// 构造函数
PnPsolver::PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches):
    pws(0), us(0), alphas(0), pcs(0), //这里的四个变量都是指针啊,直接这样子写的原因可以参考函数 set_maximum_number_of_correspondences()
    maximum_number_of_correspondences(0), number_of_correspondences(0), mnInliersi(0),
    mnIterations(0), mnBestInliers(0), N(0)
{
    // 根据点数初始化容器的大小
    mvpMapPointMatches = vpMapPointMatches;
    mvP2D.reserve(F.mvpMapPoints.size());
    mvSigma2.reserve(F.mvpMapPoints.size());
    mvP3Dw.reserve(F.mvpMapPoints.size());
    mvKeyPointIndices.reserve(F.mvpMapPoints.size());
    mvAllIndices.reserve(F.mvpMapPoints.size());

    // 生成地图点和特征点在当前求解器的vector中的下标
    int idx=0;
    // 遍历给出的每一个地图点
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMapPointMatches[i];//依次获取一个MapPoint

        if(pMP)
        {
            if(!pMP->isBad())
            {
                const cv::KeyPoint &kp = F.mvKeysUn[i];//得到2维特征点, 将KeyPoint类型变为Point2f

                mvP2D.push_back(kp.pt);//存放到mvP2D容器
                mvSigma2.push_back(F.mvLevelSigma2[kp.octave]);//记录特征点是在哪一层提取出来的

                cv::Mat Pos = pMP->GetWorldPos();//世界坐标系下的3D点
                mvP3Dw.push_back(cv::Point3f(Pos.at<float>(0),Pos.at<float>(1), Pos.at<float>(2)));

                mvKeyPointIndices.push_back(i);//记录被使用特征点在原始特征点容器中的索引, mvKeyPointIndices是跳跃的
                mvAllIndices.push_back(idx);//记录被使用特征点的索引, mvAllIndices是连续的

                idx++;
            }
        }
    } // 遍历给出的每一个地图点

    // Set camera calibration parameters
    fu = F.fx;
    fv = F.fy;
    uc = F.cx;
    vc = F.cy;

    // 设置默认的RANSAC参数,这个和Sim3Solver中的操作是相同的
    SetRansacParameters();
}

// 析构函数
PnPsolver::~PnPsolver()
{
  // 释放堆内存
  delete [] pws;
  delete [] us;
  delete [] alphas;
  delete [] pcs;
}

// 设置RANSAC迭代的参数
void PnPsolver::SetRansacParameters(double probability, int minInliers, int maxIterations, int minSet, float epsilon, float th2)
{
    // 注意这次里在每一次采样的过程中,需要采样四个点,即最小集应该设置为4

    // step 1 获取给定的参数
    mRansacProb = probability;
    mRansacMinInliers = minInliers;
    mRansacMaxIts = maxIterations;
    mRansacEpsilon = epsilon;         
    mRansacMinSet = minSet;           


    // step 2 计算理论内点数,并且选 min(给定内点数,最小集,理论内点数) 作为最终在迭代过程中使用的最小内点数
    N = mvP2D.size(); // number of correspondences, 所有二维特征点个数

    mvbInliersi.resize(N);// inlier index, mvbInliersi记录每次迭代inlier的点

    // Adjust Parameters according to number of correspondences
    // 再根据 epsilon 来计算理论上的内点数;
    // NOTICE 实际在计算的过程中使用的 mRansacMinInliers = min(给定内点数,最小集,理论内点数)
    int nMinInliers = N*mRansacEpsilon; 
    if(nMinInliers<mRansacMinInliers)
        nMinInliers=mRansacMinInliers;
    if(nMinInliers<minSet)
        nMinInliers=minSet;
    mRansacMinInliers = nMinInliers;

    // step 3 根据敲定的"最小内点数"来调整 内点数/总体数 这个比例 epsilon

    // 这个变量却是希望取得高一点,也可以理解为想让和调整之后的内点数 mRansacMinInliers 保持一致吧
    if(mRansacEpsilon<(float)mRansacMinInliers/N)
        mRansacEpsilon=(float)mRansacMinInliers/N;

    // step 4  根据给出的各种参数计算RANSAC的理论迭代次数,并且敲定最终在迭代过程中使用的RANSAC最大迭代次数
    // Set RANSAC iterations according to probability, epsilon, and max iterations -- 这个部分和Sim3Solver中的操作是一样的
    int nIterations;

    if(mRansacMinInliers==N)//根据期望的残差大小来计算RANSAC需要迭代的次数
        nIterations=1;
    else
        nIterations = ceil(log(1-mRansacProb)/log(1-pow(mRansacEpsilon,3)));

    mRansacMaxIts = max(1,min(nIterations,mRansacMaxIts));

    // step 5 计算不同图层上的特征点在进行内点检验的时候,所使用的不同判断误差阈值

    mvMaxError.resize(mvSigma2.size());// 图像提取特征的时候尺度层数
    for(size_t i=0; i<mvSigma2.size(); i++)// 不同的尺度，设置不同的最大偏差
        mvMaxError[i] = mvSigma2[i]*th2;
}

cv::Mat PnPsolver::find(vector<bool> &vbInliers, int &nInliers)
{
    bool bFlag;
    return iterate(mRansacMaxIts,bFlag,vbInliers,nInliers);    
}

//进行迭代计算
cv::Mat PnPsolver::iterate(int nIterations, bool &bNoMore, vector<bool> &vbInliers, int &nInliers)
{
    bNoMore = false;        //已经达到最大迭代次数的标志
    vbInliers.clear();
    nInliers=0;             // 当前次迭代时的内点数

    // mRansacMinSet 为每次RANSAC需要的特征点数，默认为4组3D-2D对应点,这里的操作是告知原EPnP代码
    set_maximum_number_of_correspondences(mRansacMinSet);

    // N为所有2D点的个数, mRansacMinInliers 为正常退出RANSAC迭代过程中最少的inlier数
    if(N<mRansacMinInliers)
    {
        bNoMore = true;
        return cv::Mat();
    }

    // mvAllIndices为所有参与PnP的2D点的索引
    // vAvailableIndices为每次从mvAllIndices中随机挑选mRansacMinSet组3D-2D对应点进行一次RANSAC
    vector<size_t> vAvailableIndices;

    // 当前的迭代次数id
    int nCurrentIterations = 0;

    // 进行迭代的条件:
    // 条件1: 历史进行的迭代次数少于最大迭代值
    // 条件2: 当前进行的迭代次数少于当前函数给定的最大迭代值
    while(mnIterations<mRansacMaxIts || nCurrentIterations<nIterations)
    {
        // 迭代次数更新
        nCurrentIterations++;
        mnIterations++;
        // 清空已有的匹配点的计数,为新的一次迭代作准备
        reset_correspondences();

        vAvailableIndices = mvAllIndices;

        // Get min set of points
        for(short i = 0; i < mRansacMinSet; ++i)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size()-1);

            // 将生成的这个索引映射到给定帧的特征点id
            int idx = vAvailableIndices[randi];

            // 将对应的3D-2D压入到pws和us. 这个过程中需要知道将这些点的信息存储到数组中的哪个位置,这个就由变量 number_of_correspondences 来指示了
            add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);

            // 从"可用索引表"中删除这个已经被使用的点
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        } // 选取最小集

        // Compute camera pose
        // 计算相机的位姿
        compute_pose(mRi, mti);

        // Check inliers
        // 内点外点的检查
        CheckInliers();

        // 如果当前次迭代得到的内点数已经达到了合格的要求了
        if(mnInliersi>=mRansacMinInliers)
        {
            // If it is the best solution so far, save it
            // 更新最佳的计算结果
            if(mnInliersi>mnBestInliers)
            {
                mvbBestInliers = mvbInliersi;
                mnBestInliers = mnInliersi;

                cv::Mat Rcw(3,3,CV_64F,mRi);
                cv::Mat tcw(3,1,CV_64F,mti);
                Rcw.convertTo(Rcw,CV_32F);
                tcw.convertTo(tcw,CV_32F);
                mBestTcw = cv::Mat::eye(4,4,CV_32F);
                Rcw.copyTo(mBestTcw.rowRange(0,3).colRange(0,3));
                tcw.copyTo(mBestTcw.rowRange(0,3).col(3));
            } // 更新最佳的计算结果

            // 还要求精
            if(Refine())   // 如果求精成功
            {
                nInliers = mnRefinedInliers;
                // 转录,作为计算结果
                vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
                for(int i=0; i<N; i++)
                {
                    if(mvbRefinedInliers[i])
                        vbInliers[mvKeyPointIndices[i]] = true;
                }

                // 对直接返回了求精之后的相机位姿
                return mRefinedTcw.clone();
            } // 如果求精成功

        } // 如果当前次迭代得到的内点数已经达到了合格的要求了
    } // 迭代

    // 如果执行到这里,说明可能已经超过了上面的两种迭代次数中的一个了
    // 如果是超过了程序中给定的最大迭代次数
    if(mnIterations>=mRansacMaxIts)
    {
        // 没有更多的允许迭代次数了
        bNoMore=true;
        // 但是如果我们目前得到的最好结果看上去还不错的话
        if(mnBestInliers>=mRansacMinInliers)
        {
            // 返回计算结果
            nInliers=mnBestInliers;
            vbInliers = vector<bool>(mvpMapPointMatches.size(),false);
            for(int i=0; i<N; i++)
            {
                if(mvbBestInliers[i])
                    vbInliers[mvKeyPointIndices[i]] = true;
            }
            return mBestTcw.clone();
        }
    }

    // 如果也没有好的计算结果,只好说明迭代失败咯...
    return cv::Mat();
}

// FIXME: 求精?
bool PnPsolver::Refine()
{
    vector<int> vIndices;
    vIndices.reserve(mvbBestInliers.size());

    for(size_t i=0; i<mvbBestInliers.size(); i++)
    {
        if(mvbBestInliers[i])
        {
            vIndices.push_back(i);
        }
    }

    set_maximum_number_of_correspondences(vIndices.size());

    reset_correspondences();

    for(size_t i=0; i<vIndices.size(); i++)
    {
        int idx = vIndices[i];
        add_correspondence(mvP3Dw[idx].x,mvP3Dw[idx].y,mvP3Dw[idx].z,mvP2D[idx].x,mvP2D[idx].y);
    }

    // Compute camera pose
    compute_pose(mRi, mti);

    // Check inliers
    CheckInliers();

    // 通过CheckInliers函数得到那些inlier点用来提纯
    mnRefinedInliers =mnInliersi;
    mvbRefinedInliers = mvbInliersi;

    if(mnInliersi>mRansacMinInliers)
    {
        cv::Mat Rcw(3,3,CV_64F,mRi);
        cv::Mat tcw(3,1,CV_64F,mti);
        Rcw.convertTo(Rcw,CV_32F);
        tcw.convertTo(tcw,CV_32F);
        mRefinedTcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(mRefinedTcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(mRefinedTcw.rowRange(0,3).col(3));
        return true;
    }

    return false;
}

// 通过之前求解的(R t)检查哪些3D-2D点对属于inliers
// FIXME: 
void PnPsolver::CheckInliers()
{
    mnInliersi=0;

    for(int i=0; i<N; i++)
    {
        cv::Point3f P3Dw = mvP3Dw[i];
        cv::Point2f P2D = mvP2D[i];

        // 将3D点由世界坐标系旋转到相机坐标系
        float Xc = mRi[0][0]*P3Dw.x+mRi[0][1]*P3Dw.y+mRi[0][2]*P3Dw.z+mti[0];
        float Yc = mRi[1][0]*P3Dw.x+mRi[1][1]*P3Dw.y+mRi[1][2]*P3Dw.z+mti[1];
        float invZc = 1/(mRi[2][0]*P3Dw.x+mRi[2][1]*P3Dw.y+mRi[2][2]*P3Dw.z+mti[2]);

        // 将相机坐标系下的3D进行针孔投影
        double ue = uc + fu * Xc * invZc;
        double ve = vc + fv * Yc * invZc;

        // 计算残差大小
        float distX = P2D.x-ue;
        float distY = P2D.y-ve;

        float error2 = distX*distX+distY*distY;

        if(error2<mvMaxError[i])
        {
            mvbInliersi[i]=true;
            mnInliersi++;
        }
        else
        {
            mvbInliersi[i]=false;
        }
    }
}

// number_of_correspondences为RANSAC每次PnP求解时时3D点和2D点匹配对数
// RANSAC需要很多次，maximum_number_of_correspondences为匹配对数最大值
// 这个变量用于决定pws us alphas pcs容器的大小，因此只能逐渐变大不能减小
// 如果maximum_number_of_correspondences之前设置的过小，则重新设置，并重新初始化pws us alphas pcs的大小
// ↑ 泡泡机器人注释原文
// To sum up: 这部分相当于是 ORB的代码和EPnP代码的一个"接口",设置最小集以及有关的参数
void PnPsolver::set_maximum_number_of_correspondences(int n)
{
  // 如果当前的这个变量,小于最小集,
  if (maximum_number_of_correspondences < n) {
    // 那么就先释放之前创建的数组 (NOTE 看到这里终于明白为什么在构造函数中这四个指针要先赋值为0了)
    // ? 提问:ORB中使用到的这段代码是别人写的, 而且从这些处理手法上来看非常拙劣, 那开源协议是否允许我们重构这部分代码?
    if (pws != 0) delete [] pws;
    if (us != 0) delete [] us;
    if (alphas != 0) delete [] alphas;
    if (pcs != 0) delete [] pcs;

    // 更新
    maximum_number_of_correspondences = n;
    pws = new double[3 * maximum_number_of_correspondences];    // 每个3D点有(X Y Z)三个值
    us = new double[2 * maximum_number_of_correspondences];     // 每个图像2D点有(u v)两个值
    alphas = new double[4 * maximum_number_of_correspondences]; // 每个3D点由四个控制点拟合，有四个系数
    pcs = new double[3 * maximum_number_of_correspondences];    // 每个3D点有(X Y Z)三个值
  }
}

// 清空当前已有的匹配点计数,为进行新的一次迭代作准备
void PnPsolver::reset_correspondences(void)
{
  number_of_correspondences = 0;
}

// 将给定的3D,2D点的数据压入到数组中
void PnPsolver::add_correspondence(double X, double Y, double Z, double u, double v)
{
  pws[3 * number_of_correspondences    ] = X;
  pws[3 * number_of_correspondences + 1] = Y;
  pws[3 * number_of_correspondences + 2] = Z;

  us[2 * number_of_correspondences    ] = u;
  us[2 * number_of_correspondences + 1] = v;

  // 当前次迭代中,已经采样的匹配点的个数;也用来指导这个"压入到数组"的过程中操作
  number_of_correspondences++;
}

void PnPsolver::choose_control_points(void)
{
  // Take C0 as the reference points centroid:
  // step 1：第一个控制点：参与PnP计算的参考3D点的几何中心
  cws[0][0] = cws[0][1] = cws[0][2] = 0;
  // 遍历每个匹配点中的3D点，然后对每个坐标轴加和
  for(int i = 0; i < number_of_correspondences; i++)
    for(int j = 0; j < 3; j++)
      cws[0][j] += pws[3 * i + j];
  // 再对每个轴上取均值
  for(int j = 0; j < 3; j++)
    cws[0][j] /= number_of_correspondences;


  // Take C1, C2, and C3 from PCA on the reference points:
  // step 2：计算其它三个控制点，C1, C2, C3通过PCA分解得到
  // TODO 
  // ref: https://www.zhihu.com/question/38417101
  // ref: https://yjk94.wordpress.com/2016/11/11/pca-to-layman/

  // 将所有的3D参考点写成矩阵，(number_of_correspondences * ３)的矩阵
  CvMat * PW0 = cvCreateMat(number_of_correspondences, 3, CV_64F);

  double pw0tpw0[3 * 3], dc[3], uct[3 * 3];         // 下面变量的数据区
  CvMat PW0tPW0 = cvMat(3, 3, CV_64F, pw0tpw0);     // 如变量名所示.其实这里使用cv::Mat格式主要是为了进行SVD分解
  CvMat DC      = cvMat(3, 1, CV_64F, dc);          // 分解上面矩阵得到的奇异值组成的矩阵
  CvMat UCt     = cvMat(3, 3, CV_64F, uct);         // 分解上面矩阵得到的左奇异矩阵

  // step 2.1：将存在pws中的参考3D点减去第一个控制点的坐标（相当于把第一个控制点作为原点）, 并存入PW0
  for(int i = 0; i < number_of_correspondences; i++)
    for(int j = 0; j < 3; j++)
      PW0->data.db[3 * i + j] = pws[3 * i + j] - cws[0][j];

  // step 2.2：利用SVD分解P'P可以获得P的主分量
  // 类似于齐次线性最小二乘求解的过程，
  // PW0的转置乘以PW0
  // cvMulTransposed(A_src,Res_dst,order, delta=null,scale=1): Calculates Res=(A-delta)*(A-delta)^T (order=0) or (A-delta)^T*(A-delta) (order=1)
  cvMulTransposed(PW0, &PW0tPW0, 1);
  cvSVD(&PW0tPW0,                         // A
        &DC,                              // W
        &UCt,                             // U
        0,                                // V
        CV_SVD_MODIFY_A | CV_SVD_U_T);    // flags

  cvReleaseMat(&PW0);

  // step 2.3：得到C1, C2, C3三个3D控制点，最后加上之前减掉的第一个控制点这个偏移量
  // 讲道理这里的条件不应写成4,而应该是变量 number_of_correspondences 啊
  for(int i = 1; i < 4; i++) {
    double k = sqrt(dc[i - 1] / number_of_correspondences);
    for(int j = 0; j < 3; j++)
      cws[i][j] = cws[0][j] + k * uct[3 * (i - 1) + j];
  }
}

// 求解四个控制点的系数alphas
// 每一个3D控制点，都有一组alphas与之对应
// cws1 cws2 cws3 cws4为四个控制点的坐标
// pws为3D参考点的坐标
// (a2 a3 a4)' = inverse(cws2-cws1 cws3-cws1 cws4-cws1)*(pws-cws1)，a1 = 1-a2-a3-a4
// 四个控制点相当于确定了一个子坐标系,原点是c1,另外三个轴的非单位向量是cws2-cws1 cws3-cws1 cws4-cws1
// 上式中 pws-cws1 是当前遍历的空间点到控制点1的距离(三维向量)(也就是到这个子坐标系原点的距离),然后投影到这个子坐标系的三个轴上,并且使用这三个"轴向量"的长度来表示
// 最后a1要将这个坐标形成 4x1 的齐次坐标表达形式 -- 齐次坐标的表示也是这么来的
void PnPsolver::compute_barycentric_coordinates(void)
{
  double cc[3 * 3], cc_inv[3 * 3];
  CvMat CC     = cvMat(3, 3, CV_64F, cc);       // 另外三个控制点在控制点坐标系下的坐标
  CvMat CC_inv = cvMat(3, 3, CV_64F, cc_inv);   // 上面这个矩阵的逆矩阵

  // 第一个控制点在质心的位置，后面三个控制点减去第一个控制点的坐标（以第一个控制点为原点）
  // step 1：减去质心后得到x y z轴
  // 
  // cws的排列 |cws1_x cws1_y cws1_z|  ---> |cws1|
  //          |cws2_x cws2_y cws2_z|       |cws2|
  //          |cws3_x cws3_y cws3_z|       |cws3|
  //          |cws4_x cws4_y cws4_z|       |cws4|
  //          
  // cc的排列  |cc2_x cc3_x cc4_x|  --->|cc2 cc3 cc4|
  //          |cc2_y cc3_y cc4_y|
  //          |cc2_z cc3_z cc4_z|
  for(int i = 0; i < 3; i++)                      // x y z 轴
    for(int j = 1; j < 4; j++)                    // 哪个控制点
      cc[3 * i + j - 1] = cws[j][i] - cws[0][i];  // 坐标索引中的-1是考虑到跳过了最初的控制点0

  cvInvert(&CC, &CC_inv, CV_SVD);
  double * ci = cc_inv;
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pi = pws + 3 * i;                    // pi指向第i个3D点的首地址
    double * a = alphas + 4 * i;                  // a指向第i个控制点系数alphas的首地址

    // pi[]-cws[0][]表示将pi和步骤1进行相同的平移
    // 生成a2,a3,a4
    for(int j = 0; j < 3; j++)
      // +1 是因为跳过了a0
      a[1 + j] = ci[3 * j    ] * (pi[0] - cws[0][0]) +
                 ci[3 * j + 1] * (pi[1] - cws[0][1]) +
                 ci[3 * j + 2] * (pi[2] - cws[0][2]);
    // 最后计算用于进行归一化的a0
    a[0] = 1.0f - a[1] - a[2] - a[3];
    // HERE
  } // 遍历每一个匹配点
}

// 填充最小二乘的M矩阵
// 对每一个3D参考点：
// |ai1 0    -ai1*ui, ai2  0    -ai2*ui, ai3 0   -ai3*ui, ai4 0   -ai4*ui|
// |0   ai1  -ai1*vi, 0    ai2  -ai2*vi, 0   ai3 -ai3*vi, 0   ai4 -ai4*vi|
// 其中i从0到4
void PnPsolver::fill_M(CvMat * M,
		  const int row, const double * as, const double u, const double v)
{
  double * M1 = M->data.db + row * 12;
  double * M2 = M1 + 12;

  for(int i = 0; i < 4; i++) {
    M1[3 * i    ] = as[i] * fu;
    M1[3 * i + 1] = 0.0;
    M1[3 * i + 2] = as[i] * (uc - u);

    M2[3 * i    ] = 0.0;
    M2[3 * i + 1] = as[i] * fv;
    M2[3 * i + 2] = as[i] * (vc - v);
  }
}

// 每一个控制点在相机坐标系下都表示为特征向量乘以beta的形式，EPnP论文的公式16
void PnPsolver::compute_ccs(const double * betas, const double * ut)
{
  for(int i = 0; i < 4; i++)
    ccs[i][0] = ccs[i][1] = ccs[i][2] = 0.0f;

  for(int i = 0; i < 4; i++) {
    const double * v = ut + 12 * (11 - i);
    for(int j = 0; j < 4; j++)
      for(int k = 0; k < 3; k++)
    ccs[j][k] += betas[i] * v[3 * j + k];
  }
}

// 用四个控制点作为单位向量表示下的世界坐标系下3D点的坐标
void PnPsolver::compute_pcs(void)
{
  for(int i = 0; i < number_of_correspondences; i++) {
    double * a = alphas + 4 * i;
    double * pc = pcs + 3 * i;

    for(int j = 0; j < 3; j++)
      pc[j] = a[0] * ccs[0][j] + a[1] * ccs[1][j] + a[2] * ccs[2][j] + a[3] * ccs[3][j];
  }
}

// FIXME: 根据类成员变量中给出的匹配点,计算相机位姿
double PnPsolver::compute_pose(double R[3][3], double t[3])
{
  // step 1：获得EPnP算法中的四个控制点
  choose_control_points();
  // step 2：计算世界坐标系下每个3D点用4个控制点线性表达时的系数alphas，公式1
  // HERE
  compute_barycentric_coordinates();

  // 步骤3：构造M矩阵，公式(3)(4)-->(5)(6)(7)
  CvMat * M = cvCreateMat(2 * number_of_correspondences, 12, CV_64F);

  for(int i = 0; i < number_of_correspondences; i++)
    fill_M(M, 2 * i, alphas + 4 * i, us[2 * i], us[2 * i + 1]);

  double mtm[12 * 12], d[12], ut[12 * 12];
  CvMat MtM = cvMat(12, 12, CV_64F, mtm);
  CvMat D   = cvMat(12,  1, CV_64F, d);
  CvMat Ut  = cvMat(12, 12, CV_64F, ut);

  // 步骤3：求解Mx = 0
  // SVD分解M'M
  cvMulTransposed(M, &MtM, 1);
  cvSVD(&MtM, &D, &Ut, 0, CV_SVD_MODIFY_A | CV_SVD_U_T);//得到向量ut
  cvReleaseMat(&M);

  double l_6x10[6 * 10], rho[6];
  CvMat L_6x10 = cvMat(6, 10, CV_64F, l_6x10);
  CvMat Rho    = cvMat(6,  1, CV_64F, rho);

  compute_L_6x10(ut, l_6x10);
  compute_rho(rho);

  double Betas[4][4], rep_errors[4];
  double Rs[4][3][3], ts[4][3];

  // 不管什么情况，都假设论文中N=4，并求解部分betas（如果全求解出来会有冲突）
  // 通过优化得到剩下的betas
  // 最后计算R t

  // EPnP论文公式10 15
  find_betas_approx_1(&L_6x10, &Rho, Betas[1]);
  gauss_newton(&L_6x10, &Rho, Betas[1]);
  rep_errors[1] = compute_R_and_t(ut, Betas[1], Rs[1], ts[1]);

  // EPnP论文公式11 15
  find_betas_approx_2(&L_6x10, &Rho, Betas[2]);
  gauss_newton(&L_6x10, &Rho, Betas[2]);
  rep_errors[2] = compute_R_and_t(ut, Betas[2], Rs[2], ts[2]);

  find_betas_approx_3(&L_6x10, &Rho, Betas[3]);
  gauss_newton(&L_6x10, &Rho, Betas[3]);
  rep_errors[3] = compute_R_and_t(ut, Betas[3], Rs[3], ts[3]);

  int N = 1;
  if (rep_errors[2] < rep_errors[1]) N = 2;
  if (rep_errors[3] < rep_errors[N]) N = 3;

  copy_R_and_t(Rs[N], ts[N], R, t);

  return rep_errors[N];
}

void PnPsolver::copy_R_and_t(const double R_src[3][3], const double t_src[3],
			double R_dst[3][3], double t_dst[3])
{
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++)
      R_dst[i][j] = R_src[i][j];
    t_dst[i] = t_src[i];
  }
}

double PnPsolver::dist2(const double * p1, const double * p2)
{
  return
    (p1[0] - p2[0]) * (p1[0] - p2[0]) +
    (p1[1] - p2[1]) * (p1[1] - p2[1]) +
    (p1[2] - p2[2]) * (p1[2] - p2[2]);
}

double PnPsolver::dot(const double * v1, const double * v2)
{
  return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

double PnPsolver::reprojection_error(const double R[3][3], const double t[3])
{
  double sum2 = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    double * pw = pws + 3 * i;
    double Xc = dot(R[0], pw) + t[0];
    double Yc = dot(R[1], pw) + t[1];
    double inv_Zc = 1.0 / (dot(R[2], pw) + t[2]);
    double ue = uc + fu * Xc * inv_Zc;
    double ve = vc + fv * Yc * inv_Zc;
    double u = us[2 * i], v = us[2 * i + 1];

    sum2 += sqrt( (u - ue) * (u - ue) + (v - ve) * (v - ve) );
  }

  return sum2 / number_of_correspondences;
}

// 根据世界坐标系下的四个控制点与机体坐标下对应的四个控制点（和世界坐标系下四个控制点相同尺度），求取R t
void PnPsolver::estimate_R_and_t(double R[3][3], double t[3])
{
  double pc0[3], pw0[3];

  pc0[0] = pc0[1] = pc0[2] = 0.0;
  pw0[0] = pw0[1] = pw0[2] = 0.0;

  for(int i = 0; i < number_of_correspondences; i++) {
    const double * pc = pcs + 3 * i;
    const double * pw = pws + 3 * i;

    for(int j = 0; j < 3; j++) {
      pc0[j] += pc[j];
      pw0[j] += pw[j];
    }
  }
  for(int j = 0; j < 3; j++) {
    pc0[j] /= number_of_correspondences;
    pw0[j] /= number_of_correspondences;
  }

  double abt[3 * 3], abt_d[3], abt_u[3 * 3], abt_v[3 * 3];
  CvMat ABt   = cvMat(3, 3, CV_64F, abt);
  CvMat ABt_D = cvMat(3, 1, CV_64F, abt_d);
  CvMat ABt_U = cvMat(3, 3, CV_64F, abt_u);
  CvMat ABt_V = cvMat(3, 3, CV_64F, abt_v);

  cvSetZero(&ABt);
  for(int i = 0; i < number_of_correspondences; i++) {
    double * pc = pcs + 3 * i;
    double * pw = pws + 3 * i;

    for(int j = 0; j < 3; j++) {
      abt[3 * j    ] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
      abt[3 * j + 1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
      abt[3 * j + 2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
    }
  }

  cvSVD(&ABt, &ABt_D, &ABt_U, &ABt_V, CV_SVD_MODIFY_A);

  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      R[i][j] = dot(abt_u + 3 * i, abt_v + 3 * j);

  const double det =
    R[0][0] * R[1][1] * R[2][2] + R[0][1] * R[1][2] * R[2][0] + R[0][2] * R[1][0] * R[2][1] -
    R[0][2] * R[1][1] * R[2][0] - R[0][1] * R[1][0] * R[2][2] - R[0][0] * R[1][2] * R[2][1];

  if (det < 0) {
    R[2][0] = -R[2][0];
    R[2][1] = -R[2][1];
    R[2][2] = -R[2][2];
  }

  t[0] = pc0[0] - dot(R[0], pw0);
  t[1] = pc0[1] - dot(R[1], pw0);
  t[2] = pc0[2] - dot(R[2], pw0);
}

void PnPsolver::print_pose(const double R[3][3], const double t[3])
{
  cout << R[0][0] << " " << R[0][1] << " " << R[0][2] << " " << t[0] << endl;
  cout << R[1][0] << " " << R[1][1] << " " << R[1][2] << " " << t[1] << endl;
  cout << R[2][0] << " " << R[2][1] << " " << R[2][2] << " " << t[2] << endl;
}

void PnPsolver::solve_for_sign(void)
{
  if (pcs[2] < 0.0) {
    for(int i = 0; i < 4; i++)
      for(int j = 0; j < 3; j++)
	ccs[i][j] = -ccs[i][j];

    for(int i = 0; i < number_of_correspondences; i++) {
      pcs[3 * i    ] = -pcs[3 * i];
      pcs[3 * i + 1] = -pcs[3 * i + 1];
      pcs[3 * i + 2] = -pcs[3 * i + 2];
    }
  }
}

double PnPsolver::compute_R_and_t(const double * ut, const double * betas,
			     double R[3][3], double t[3])
{
  compute_ccs(betas, ut);
  compute_pcs();

  solve_for_sign();

  estimate_R_and_t(R, t);

  return reprojection_error(R, t);
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_1 = [B11 B12     B13         B14]

void PnPsolver::find_betas_approx_1(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x4[6 * 4], b4[4];
  CvMat L_6x4 = cvMat(6, 4, CV_64F, l_6x4);
  CvMat B4    = cvMat(4, 1, CV_64F, b4);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x4, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x4, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x4, i, 2, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x4, i, 3, cvmGet(L_6x10, i, 6));
  }

  cvSolve(&L_6x4, Rho, &B4, CV_SVD);

  if (b4[0] < 0) {
    betas[0] = sqrt(-b4[0]);
    betas[1] = -b4[1] / betas[0];
    betas[2] = -b4[2] / betas[0];
    betas[3] = -b4[3] / betas[0];
  } else {
    betas[0] = sqrt(b4[0]);
    betas[1] = b4[1] / betas[0];
    betas[2] = b4[2] / betas[0];
    betas[3] = b4[3] / betas[0];
  }
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_2 = [B11 B12 B22                            ]

void PnPsolver::find_betas_approx_2(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x3[6 * 3], b3[3];
  CvMat L_6x3  = cvMat(6, 3, CV_64F, l_6x3);
  CvMat B3     = cvMat(3, 1, CV_64F, b3);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x3, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x3, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x3, i, 2, cvmGet(L_6x10, i, 2));
  }

  cvSolve(&L_6x3, Rho, &B3, CV_SVD);

  if (b3[0] < 0) {
    betas[0] = sqrt(-b3[0]);
    betas[1] = (b3[2] < 0) ? sqrt(-b3[2]) : 0.0;
  } else {
    betas[0] = sqrt(b3[0]);
    betas[1] = (b3[2] > 0) ? sqrt(b3[2]) : 0.0;
  }

  if (b3[1] < 0) betas[0] = -betas[0];

  betas[2] = 0.0;
  betas[3] = 0.0;
}

// betas10        = [B11 B12 B22 B13 B23 B33 B14 B24 B34 B44]
// betas_approx_3 = [B11 B12 B22 B13 B23                    ]

void PnPsolver::find_betas_approx_3(const CvMat * L_6x10, const CvMat * Rho,
			       double * betas)
{
  double l_6x5[6 * 5], b5[5];
  CvMat L_6x5 = cvMat(6, 5, CV_64F, l_6x5);
  CvMat B5    = cvMat(5, 1, CV_64F, b5);

  for(int i = 0; i < 6; i++) {
    cvmSet(&L_6x5, i, 0, cvmGet(L_6x10, i, 0));
    cvmSet(&L_6x5, i, 1, cvmGet(L_6x10, i, 1));
    cvmSet(&L_6x5, i, 2, cvmGet(L_6x10, i, 2));
    cvmSet(&L_6x5, i, 3, cvmGet(L_6x10, i, 3));
    cvmSet(&L_6x5, i, 4, cvmGet(L_6x10, i, 4));
  }

  cvSolve(&L_6x5, Rho, &B5, CV_SVD);

  if (b5[0] < 0) {
    betas[0] = sqrt(-b5[0]);
    betas[1] = (b5[2] < 0) ? sqrt(-b5[2]) : 0.0;
  } else {
    betas[0] = sqrt(b5[0]);
    betas[1] = (b5[2] > 0) ? sqrt(b5[2]) : 0.0;
  }
  if (b5[1] < 0) betas[0] = -betas[0];
  betas[2] = b5[3] / betas[0];
  betas[3] = 0.0;
}

// 计算并填充矩阵L
void PnPsolver::compute_L_6x10(const double * ut, double * l_6x10)
{
  const double * v[4];

  v[0] = ut + 12 * 11;
  v[1] = ut + 12 * 10;
  v[2] = ut + 12 *  9;
  v[3] = ut + 12 *  8;

  double dv[4][6][3];

  for(int i = 0; i < 4; i++) {
    int a = 0, b = 1;
    for(int j = 0; j < 6; j++) {
      dv[i][j][0] = v[i][3 * a    ] - v[i][3 * b];
      dv[i][j][1] = v[i][3 * a + 1] - v[i][3 * b + 1];
      dv[i][j][2] = v[i][3 * a + 2] - v[i][3 * b + 2];

      b++;
      if (b > 3) {
        a++;
        b = a + 1;
      }
    }
  }

  for(int i = 0; i < 6; i++) {
    double * row = l_6x10 + 10 * i;

    row[0] =        dot(dv[0][i], dv[0][i]);
    row[1] = 2.0f * dot(dv[0][i], dv[1][i]);
    row[2] =        dot(dv[1][i], dv[1][i]);
    row[3] = 2.0f * dot(dv[0][i], dv[2][i]);
    row[4] = 2.0f * dot(dv[1][i], dv[2][i]);
    row[5] =        dot(dv[2][i], dv[2][i]);
    row[6] = 2.0f * dot(dv[0][i], dv[3][i]);
    row[7] = 2.0f * dot(dv[1][i], dv[3][i]);
    row[8] = 2.0f * dot(dv[2][i], dv[3][i]);
    row[9] =        dot(dv[3][i], dv[3][i]);
  }
}

// 计算四个控制点任意两点间的距离，总共6个距离
void PnPsolver::compute_rho(double * rho)
{
  rho[0] = dist2(cws[0], cws[1]);
  rho[1] = dist2(cws[0], cws[2]);
  rho[2] = dist2(cws[0], cws[3]);
  rho[3] = dist2(cws[1], cws[2]);
  rho[4] = dist2(cws[1], cws[3]);
  rho[5] = dist2(cws[2], cws[3]);
}

void PnPsolver::compute_A_and_b_gauss_newton(const double * l_6x10, const double * rho,
					double betas[4], CvMat * A, CvMat * b)
{
  for(int i = 0; i < 6; i++) {
    const double * rowL = l_6x10 + i * 10;
    double * rowA = A->data.db + i * 4;

    rowA[0] = 2 * rowL[0] * betas[0] +     rowL[1] * betas[1] +     rowL[3] * betas[2] +     rowL[6] * betas[3];
    rowA[1] =     rowL[1] * betas[0] + 2 * rowL[2] * betas[1] +     rowL[4] * betas[2] +     rowL[7] * betas[3];
    rowA[2] =     rowL[3] * betas[0] +     rowL[4] * betas[1] + 2 * rowL[5] * betas[2] +     rowL[8] * betas[3];
    rowA[3] =     rowL[6] * betas[0] +     rowL[7] * betas[1] +     rowL[8] * betas[2] + 2 * rowL[9] * betas[3];

    cvmSet(b, i, 0, rho[i] -
	   (
	    rowL[0] * betas[0] * betas[0] +
	    rowL[1] * betas[0] * betas[1] +
	    rowL[2] * betas[1] * betas[1] +
	    rowL[3] * betas[0] * betas[2] +
	    rowL[4] * betas[1] * betas[2] +
	    rowL[5] * betas[2] * betas[2] +
	    rowL[6] * betas[0] * betas[3] +
	    rowL[7] * betas[1] * betas[3] +
	    rowL[8] * betas[2] * betas[3] +
	    rowL[9] * betas[3] * betas[3]
	    ));
  }
}

void PnPsolver::gauss_newton(const CvMat * L_6x10, const CvMat * Rho,
			double betas[4])
{
  const int iterations_number = 5;

  double a[6*4], b[6], x[4];
  CvMat A = cvMat(6, 4, CV_64F, a);
  CvMat B = cvMat(6, 1, CV_64F, b);
  CvMat X = cvMat(4, 1, CV_64F, x);

  for(int k = 0; k < iterations_number; k++) {
    compute_A_and_b_gauss_newton(L_6x10->data.db, Rho->data.db,
				 betas, &A, &B);
    qr_solve(&A, &B, &X);

    for(int i = 0; i < 4; i++)
      betas[i] += x[i];
  }
}

void PnPsolver::qr_solve(CvMat * A, CvMat * b, CvMat * X)
{
  static int max_nr = 0;
  static double * A1, * A2;

  const int nr = A->rows;
  const int nc = A->cols;

  if (max_nr != 0 && max_nr < nr) {
    delete [] A1;
    delete [] A2;
  }
  if (max_nr < nr) {
    max_nr = nr;
    A1 = new double[nr];
    A2 = new double[nr];
  }

  double * pA = A->data.db, * ppAkk = pA;
  for(int k = 0; k < nc; k++) {
    double * ppAik = ppAkk, eta = fabs(*ppAik);
    for(int i = k + 1; i < nr; i++) {
      double elt = fabs(*ppAik);
      if (eta < elt) eta = elt;
      ppAik += nc;
    }

    if (eta == 0) {
      A1[k] = A2[k] = 0.0;
      cerr << "God damnit, A is singular, this shouldn't happen." << endl;
      return;
    } else {
      double * ppAik = ppAkk, sum = 0.0, inv_eta = 1. / eta;
      for(int i = k; i < nr; i++) {
	*ppAik *= inv_eta;
	sum += *ppAik * *ppAik;
	ppAik += nc;
      }
      double sigma = sqrt(sum);
      if (*ppAkk < 0)
	sigma = -sigma;
      *ppAkk += sigma;
      A1[k] = sigma * *ppAkk;
      A2[k] = -eta * sigma;
      for(int j = k + 1; j < nc; j++) {
	double * ppAik = ppAkk, sum = 0;
	for(int i = k; i < nr; i++) {
	  sum += *ppAik * ppAik[j - k];
	  ppAik += nc;
	}
	double tau = sum / A1[k];
	ppAik = ppAkk;
	for(int i = k; i < nr; i++) {
	  ppAik[j - k] -= tau * *ppAik;
	  ppAik += nc;
	}
      }
    }
    ppAkk += nc + 1;
  }

  // b <- Qt b
  double * ppAjj = pA, * pb = b->data.db;
  for(int j = 0; j < nc; j++) {
    double * ppAij = ppAjj, tau = 0;
    for(int i = j; i < nr; i++)	{
      tau += *ppAij * pb[i];
      ppAij += nc;
    }
    tau /= A1[j];
    ppAij = ppAjj;
    for(int i = j; i < nr; i++) {
      pb[i] -= tau * *ppAij;
      ppAij += nc;
    }
    ppAjj += nc + 1;
  }

  // X = R-1 b
  double * pX = X->data.db;
  pX[nc - 1] = pb[nc - 1] / A2[nc - 1];
  for(int i = nc - 2; i >= 0; i--) {
    double * ppAij = pA + i * nc + (i + 1), sum = 0;

    for(int j = i + 1; j < nc; j++) {
      sum += *ppAij * pX[j];
      ppAij++;
    }
    pX[i] = (pb[i] - sum) / A2[i];
  }
}



void PnPsolver::relative_error(double & rot_err, double & transl_err,
			  const double Rtrue[3][3], const double ttrue[3],
			  const double Rest[3][3],  const double test[3])
{
  double qtrue[4], qest[4];

  mat_to_quat(Rtrue, qtrue);
  mat_to_quat(Rest, qest);

  double rot_err1 = sqrt((qtrue[0] - qest[0]) * (qtrue[0] - qest[0]) +
			 (qtrue[1] - qest[1]) * (qtrue[1] - qest[1]) +
			 (qtrue[2] - qest[2]) * (qtrue[2] - qest[2]) +
			 (qtrue[3] - qest[3]) * (qtrue[3] - qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  double rot_err2 = sqrt((qtrue[0] + qest[0]) * (qtrue[0] + qest[0]) +
			 (qtrue[1] + qest[1]) * (qtrue[1] + qest[1]) +
			 (qtrue[2] + qest[2]) * (qtrue[2] + qest[2]) +
			 (qtrue[3] + qest[3]) * (qtrue[3] + qest[3]) ) /
    sqrt(qtrue[0] * qtrue[0] + qtrue[1] * qtrue[1] + qtrue[2] * qtrue[2] + qtrue[3] * qtrue[3]);

  rot_err = min(rot_err1, rot_err2);

  transl_err =
    sqrt((ttrue[0] - test[0]) * (ttrue[0] - test[0]) +
	 (ttrue[1] - test[1]) * (ttrue[1] - test[1]) +
	 (ttrue[2] - test[2]) * (ttrue[2] - test[2])) /
    sqrt(ttrue[0] * ttrue[0] + ttrue[1] * ttrue[1] + ttrue[2] * ttrue[2]);
}

void PnPsolver::mat_to_quat(const double R[3][3], double q[4])
{
  double tr = R[0][0] + R[1][1] + R[2][2];
  double n4;

  if (tr > 0.0f) {
    q[0] = R[1][2] - R[2][1];
    q[1] = R[2][0] - R[0][2];
    q[2] = R[0][1] - R[1][0];
    q[3] = tr + 1.0f;
    n4 = q[3];
  } else if ( (R[0][0] > R[1][1]) && (R[0][0] > R[2][2]) ) {
    q[0] = 1.0f + R[0][0] - R[1][1] - R[2][2];
    q[1] = R[1][0] + R[0][1];
    q[2] = R[2][0] + R[0][2];
    q[3] = R[1][2] - R[2][1];
    n4 = q[0];
  } else if (R[1][1] > R[2][2]) {
    q[0] = R[1][0] + R[0][1];
    q[1] = 1.0f + R[1][1] - R[0][0] - R[2][2];
    q[2] = R[2][1] + R[1][2];
    q[3] = R[2][0] - R[0][2];
    n4 = q[1];
  } else {
    q[0] = R[2][0] + R[0][2];
    q[1] = R[2][1] + R[1][2];
    q[2] = 1.0f + R[2][2] - R[0][0] - R[1][1];
    q[3] = R[0][1] - R[1][0];
    n4 = q[2];
  }
  double scale = 0.5f / double(sqrt(n4));

  q[0] *= scale;
  q[1] *= scale;
  q[2] *= scale;
  q[3] *= scale;
}

} //namespace ORB_SLAM
