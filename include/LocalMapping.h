/**
 * @file LocalMapping.h
 * @author guoqing (1337841346@qq.com)
 * @brief 局部建图线程
 * @version 0.1
 * @date 2019-04-29
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



#ifndef LOCALMAPPING_H
#define LOCALMAPPING_H

#include "KeyFrame.h"
#include "Map.h"
#include "LoopClosing.h"
#include "Tracking.h"
#include "KeyFrameDatabase.h"

#include <mutex>


namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;

/** @brief 局部建图线程类 */
class LocalMapping
{
public:

    /**
     * @brief 构造函数
     * @param[in] pMap          局部地图的句柄？ //?
     * @param[in] bMonocular    当前系统是否是单目输入
     */
    LocalMapping(Map* pMap, const float bMonocular);

    /**
     * @brief 设置回环检测线程句柄
     * @param[in] pLoopCloser 回环检测线程句柄
     */
    void SetLoopCloser(LoopClosing* pLoopCloser);

    /**
     * @brief 设置追踪线程句柄
     * @param[in] pTracker 追踪线程句柄
     */
    void SetTracker(Tracking* pTracker);

    // Main function
    /** @brief 线程主函数 */
    void Run();

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();
    void RequestReset();
    bool Stop();
    void Release();
    bool isStopped();
    bool stopRequested();
    bool AcceptKeyFrames();
    /**
     * @brief 设置"允许接受关键帧"的状态标志
     * @param[in] flag 是或者否
     */
    void SetAcceptKeyFrames(bool flag);
    bool SetNotStop(bool flag);

    void InterruptBA();

    void RequestFinish();
    bool isFinished();
    //查看队列中等待插入的关键帧数目
    int KeyframesInQueue(){
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    /**
     * @brief 查看列表中是否有等待被插入的关键帧
     * @return 如果存在，返回true
     */
    bool CheckNewKeyFrames();
    /**
     * @brief 处理列表中的关键帧
     * 
     * - 计算Bow，加速三角化新的MapPoints
     * - 关联当前关键帧至MapPoints，并更新MapPoints的平均观测方向和观测距离范围
     * - 插入关键帧，更新Covisibility图和Essential图
     * @see VI-A keyframe insertion
     */
    void ProcessNewKeyFrame();
    /** @brief 相机运动过程中和共视程度比较高的关键帧通过三角化恢复出一些MapPoints */
    void CreateNewMapPoints();

    /**
     * @brief 剔除ProcessNewKeyFrame和CreateNewMapPoints函数中引入的质量不好的MapPoints
     * @see VI-B recent map points culling
     */
    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    /**
     * 根据两关键帧的姿态计算两个关键帧之间的基本矩阵
     * @param  pKF1 关键帧1
     * @param  pKF2 关键帧2
     * @return      基本矩阵
     */
    cv::Mat ComputeF12(KeyFrame* &pKF1, KeyFrame* &pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    /// 当前系统输入数单目还是双目RGB-D的标志
    bool mbMonocular;

    void ResetIfRequested();
    /// 当前系统是否收到了请求复位的信号
    bool mbResetRequested;
    std::mutex mMutexReset;

    bool CheckFinish();
    void SetFinish();
    /// 当前线程是否收到了请求终止的信号
    bool mbFinishRequested;
    /// 当前线程的主函数是否已经停止了工作
    bool mbFinished;
    std::mutex mMutexFinish;

    // 指向局部地图的句柄
    Map* mpMap;

    LoopClosing* mpLoopCloser;
    Tracking* mpTracker;

    // Tracking线程向LocalMapping中插入关键帧是先插入到该队列中
    std::list<KeyFrame*> mlNewKeyFrames; ///< 等待处理的关键帧列表
    /// 当前正在处理的关键帧
    KeyFrame* mpCurrentKeyFrame;

    /// 存储当前关键帧生成的地图点,也是等待检查的地图点列表
    std::list<MapPoint*> mlpRecentAddedMapPoints;

    /// 操作关键帧列表时使用的互斥量 
    std::mutex mMutexNewKFs;

    bool mbAbortBA;

    bool mbStopped;
    bool mbStopRequested;
    bool mbNotStop;
    std::mutex mMutexStop;

    /// 当前局部建图线程是否允许关键帧输入
    bool mbAcceptKeyFrames;
    /// 和操作上面这个变量有关的互斥量
    std::mutex mMutexAccept;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
