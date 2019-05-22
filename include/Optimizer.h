/**
 * @file Optimizer.h
 * @author guoqing (1337841346@qq.com)
 * @brief 优化器，所有用到的优化函数的声明都在这个文件中
 * @version 0.1
 * @date 2019-05-22
 * 
 * @copyright Copyright (c) 2019
 * 
 */

/**
* This file is part of ORB-SLAM2.
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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "Map.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "LoopClosing.h"
#include "Frame.h"

#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class LoopClosing;

/** @brief 优化器,所有的优化相关的函数都在这个类中; 并且这个类只有成员函数没有成员变量,相对要好分析一点 */
class Optimizer
{
public:

    /**
     * @brief bundle adjustment Optimization
     * 
     * 3D-2D 最小化重投影误差 e = (u,v) - project(Tcw*Pw) \n
     * 
     * 1. Vertex: g2o::VertexSE3Expmap()，即当前帧的Tcw
     *            g2o::VertexSBAPointXYZ()，MapPoint的mWorldPos
     * 2. Edge:
     *     - g2o::EdgeSE3ProjectXYZ()，BaseBinaryEdge
     *         + Vertex：待优化当前帧的Tcw
     *         + Vertex：待优化MapPoint的mWorldPos
     *         + measurement：MapPoint在当前帧中的二维位置(u,v)
     *         + InfoMatrix: invSigma2(与特征点所在的尺度有关)
     *         
     * @param   vpKFs    关键帧 
     *          vpMP     MapPoints
     *          nIterations 迭代次数（20次）
     *          pbStopFlag  是否强制暂停
     *          nLoopKF  关键帧的个数 -- 但是我觉得形成了闭环关系的当前关键帧的id
     *          bRobust  是否使用核函数
     */
    void static BundleAdjustment(const std::vector<KeyFrame*> &vpKF, const std::vector<MapPoint*> &vpMP,
                                 int nIterations = 5, bool *pbStopFlag=NULL, const unsigned long nLoopKF=0,
                                 const bool bRobust = true);

    /**
     * @brief 进行全局BA优化，但主要功能还是调用 BundleAdjustment,这个函数相当于加了一个壳.
     * @param[in] pMap          地图对象的指针
     * @param[in] nIterations   迭代次数
     * @param[in] pbStopFlag    外界给的控制GBA停止的标志位
     * @param[in] nLoopKF       回环关键帧的个数? //?
     * @param[in] bRobust       是否使用鲁棒核函数
     */
    void static GlobalBundleAdjustemnt(Map* pMap, int nIterations=5, bool *pbStopFlag=NULL,
                                       const unsigned long nLoopKF=0, const bool bRobust = true);
    void static LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);
    int static PoseOptimization(Frame* pFrame);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    void static OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
                                       const LoopClosing::KeyFrameAndPose &NonCorrectedSim3,
                                       const LoopClosing::KeyFrameAndPose &CorrectedSim3,
                                       const map<KeyFrame *, set<KeyFrame *> > &LoopConnections,
                                       const bool &bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    // 闭环刚刚形成的时候,对当前关键帧和闭环关键帧之间的sim3变换的优化
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches1,
                            g2o::Sim3 &g2oS12, const float th2, const bool bFixScale);
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
