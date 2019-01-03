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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include<vector>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"MapPoint.h"
#include"KeyFrame.h"
#include"Frame.h"


namespace ORB_SLAM2
{

class ORBmatcher
{    
public:

    ORBmatcher(float nnratio=0.6, bool checkOri=true);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    /**
     * @brief 通过投影，对Local MapPoint进行跟踪
     *
     * 将Local MapPoint投影到当前帧中, 由此增加当前帧的MapPoints \n
     * 在SearchLocalPoints()中已经将Local MapPoints重投影（isInFrustum()）到当前帧 \n
     * 并标记了这些点是否在当前帧的视野中，即mbTrackInView \n
     * 对这些MapPoints，在其投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
     * @param  F           当前帧
     * @param  vpMapPoints Local MapPoints
     * @param  th          阈值
     * @return             成功匹配的数量
     * @see SearchLocalPoints() isInFrustum()
     */
    int SearchByProjection(Frame &F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    /**
     * @brief 通过投影，对上一帧的特征点进行跟踪
     *
     * 上一帧中包含了MapPoints，对这些MapPoints进行tracking，由此增加当前帧的MapPoints \n
     * 1. 将上一帧的MapPoints投影到当前帧(根据速度模型可以估计当前帧的Tcw)
     * 2. 在投影点附近根据描述子距离选取匹配，以及最终的方向投票机制进行剔除
     * @param  CurrentFrame 当前帧
     * @param  LastFrame    上一帧
     * @param  th           阈值
     * @param  bMono        是否为单目
     * @return              成功匹配的数量
     * @see SearchByBoW()
     */
    int SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame &CurrentFrame, KeyFrame* pKF, const std::set<MapPoint*> &sAlreadyFound, const float th, const int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
    int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    /**
     * @brief 通过词包，对关键帧的特征点进行跟踪
     * 
     * KeyFrame中包含了MapPoints，对这些MapPoints进行tracking \n
     * 由于每一个MapPoint对应有描述子，因此可以通过描述子距离进行跟踪 \n
     * 为了加速匹配过程，将关键帧和当前帧的描述子划分到特定层的nodes中 \n
     * 对属于同一node的描述子计算距离进行匹配 \n
     * 通过距离阈值、比例阈值和角度投票进行剔除误匹配
     * @param  pKF               KeyFrame
     * @param  F                 Current Frame
     * @param  vpMapPointMatches F中MapPoints对应的匹配，NULL表示未匹配
     * @return                   成功匹配的数量
     */
    int SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame &F1, Frame &F2, std::vector<cv::Point2f> &vbPrevMatched, std::vector<int> &vnMatches12, int windowSize=10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    int SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat F12,
                               std::vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12, const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, const vector<MapPoint *> &vpMapPoints, const float th=3.0);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint);

public:

    static const int TH_LOW;
    static const int TH_HIGH;
    static const int HISTO_LENGTH;


protected:

    bool CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &F12, const KeyFrame *pKF);

    float RadiusByViewingCos(const float &viewCos);

    void ComputeThreeMaxima(std::vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3);

    float mfNNratio;
    bool mbCheckOrientation;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
