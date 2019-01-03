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

#ifndef FRAME_H
#define FRAME_H

#include<vector>

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <opencv2/opencv.hpp>

namespace ORB_SLAM2
{
	
//这里是定义图像网格的多少
#define FRAME_GRID_ROWS 48
#define FRAME_GRID_COLS 64

class MapPoint;
class KeyFrame;

class Frame
{
public:
	//无参构造函数
    Frame();

    // Copy constructor. 拷贝构造函数
    Frame(const Frame &frame);

    // Constructor for stereo cameras.  为双目相机准备的构造函数
    Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for RGB-D cameras.	为RGBD相机准备的构造函数
    Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Constructor for Monocular cameras.	为单目相机准备的构造函数
    Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    // 提取的关键点存放在mvKeys和mDescriptors中
    // ORB是直接调orbExtractor提取的
	//由于左侧图像和右侧图像进行特征点提取后的结果所保存在的变量不同，所以这里需要加左右侧标志
	//TODO 此外左右侧图像使用的orbExtractor也不相同，这个有什么说法吗？
    void ExtractORB(int flag, const cv::Mat &im);

    // Compute Bag of Words representation.
    // 存放在mBowVec中
    void ComputeBoW();

    // Set the camera pose.
    // 用Tcw更新mTcw
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    inline cv::Mat GetCameraCenter()
	{
        return mOw.clone();
    }

    // Returns inverse of rotation
    //NOTICE 默认的mRwc存储的是当前帧时，相机从当前的坐标系变换到世界坐标系所进行的旋转，而我们常谈的旋转则说的是从世界坐标系到当前相机坐标系的旋转
    inline cv::Mat GetRotationInverse()
	{
		//所以直接返回其实就是我们常谈的旋转的逆了
        return mRwc.clone();
    }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    // 判断路标点是否在视野中
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit);

    // Compute the cell of a keypoint (return false if outside the grid)
	//根据特征点的坐标计算该点所处的图像网格
    bool PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY);

	//获取指定区域（x,y,r）内的特征点
    vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel=-1, const int maxLevel=-1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
	//博客[https://www.cnblogs.com/panda1/p/7001052.html]中说，这个是将地图点和其深度对应起来
	//是不是可以这样理解，为了能够将特征点反投影到三维空间中得到其在相机坐标系以及在世界坐标系下的坐标，我们需要获得它
	//在当前相机下的深度。对于双目相机，我们是通过计算左侧图像中特征点在右图中的坐标，然后计算其深度；对于RGBD图像我们可以直接
	//从深度图像上获得特征点的深度，不过为了处理上的一致这里使用这个深度计算了彩色图像（左图）中的特征点在假想的“右图”中的
	//坐标。这就是这个函数的工作
    void ComputeStereoFromRGBD(const cv::Mat &imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(const int &i);

public:
    // Vocabulary used for relocalization.
    ORBVocabulary* mpORBvocabulary;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;

    // Frame timestamp.
    double mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat mK;
	//NOTICE 注意这里的相机内参数其实都是类的静态成员变量；此外相机的内参数矩阵和矫正参数矩阵却是普通的成员变量，
	//NOTE 这样是否有些浪费内存空间？
    static float fx;
    static float fy;
    static float cx;
    static float cy;
    static float invfx;
    static float invfy;
	//TODO 目测是opencv提供的图像去畸变参数矩阵的，但是其具体组成未知
    cv::Mat mDistCoef;

    // Stereo baseline multiplied by fx.
    float mbf;

    // Stereo baseline in meters.
    float mb;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
	//TODO 这里它所说的话还不是很理解。尤其是后面的一句。
    float mThDepth;

    // Number of KeyPoints.
    int N; ///< KeyPoints数量

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    // mvKeys:原始左图像提取出的特征点（未校正）
    // mvKeysRight:原始右图像提取出的特征点（未校正）
    // mvKeysUn:校正mvKeys后的特征点，对于双目摄像头，一般得到的图像都是校正好的，再校正一次有点多余
    //校正操作是在帧的构造函数中进行的。
    std::vector<cv::KeyPoint> mvKeys, mvKeysRight;
	//原来这个向量存储的是已经矫正过后的特征点啊
    std::vector<cv::KeyPoint> mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    // 对于双目，mvuRight存储了左目像素点在右目中的对应点的横坐标 （因为纵坐标是一样的）
    // mvDepth对应的深度
    // 单目摄像头，这两个容器中存的都是-1
    std::vector<float> mvuRight;	//m-member v-vector u-指代横坐标,因为最后这个坐标是通过各种拟合方法逼近出来的，所以使用float存储
    std::vector<float> mvDepth;

    // Bag of Words Vector structures.
    DBoW2::BowVector mBowVec;
    DBoW2::FeatureVector mFeatVec;

    // ORB descriptor, each row associated to a keypoint.
    // 左目摄像头和右目摄像头特征点对应的描述子
    cv::Mat mDescriptors, mDescriptorsRight;

    // MapPoints associated to keypoints, NULL pointer if no association.
    // 每个特征点对应的MapPoint
    std::vector<MapPoint*> mvpMapPoints;

    // Flag to identify outlier associations.
    // 观测不到Map中的3D点
	//TODO 可是这里说的不应该是当前帧中标记那些属于外点的特征点吗，目前frame.c中没有真正意义上使用到，无法做定论
    std::vector<bool> mvbOutlier;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
	//原来通过对图像分区域还能够降低重投影地图点时候的匹配复杂度啊。。。。。
    // 坐标乘以mfGridElementWidthInv和mfGridElementHeightInv就可以确定在哪个格子
	//注意到这个也是类的静态成员变量， 有一个专用的标志mbInitialComputations用来在帧的构造函数中标记这些静态成员变量是否需要被赋值
    static float mfGridElementWidthInv;
    static float mfGridElementHeightInv;
    // 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
    // FRAME_GRID_ROWS 48
    // FRAME_GRID_COLS 64
	//这个向量中存储的是每个图像网格内特征点的id（左图）
    std::vector<std::size_t> mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat mTcw; ///< 相机姿态 世界坐标系到相机坐标坐标系的变换矩阵,是我们常规理解中的相机位姿

    // Current and Next Frame id.
    // 类的静态成员变量，这些变量则是在整个系统开始执行的时候被初始化的——它在全局区被初始化
    static long unsigned int nNextId; ///< Next Frame id.
    long unsigned int mnId; ///< Current Frame id.

    // Reference Keyframe.
    KeyFrame* mpReferenceKF;//指针，指向参考关键帧

    // Scale pyramid info.
    int mnScaleLevels;//图像提金字塔的层数
    float mfScaleFactor;//图像提金字塔的尺度因子
    float mfLogScaleFactor;//图像提金字塔的尺度因子的对数值？  TODO 为什么要计算存储这个，有什么实际意义吗
    vector<float> mvScaleFactors;		//图像金字塔每一层的缩放因子
    vector<float> mvInvScaleFactors;	//以及上面的这个变量的倒数
    vector<float> mvLevelSigma2;		//？？？ TODO 目前在frame.c中没有用到，无法下定论
    vector<float> mvInvLevelSigma2;		//上面变量的倒数

    // Undistorted Image Bounds (computed once).
    // 用于确定画格子时的边界 （未校正图像的边界，只需要计算一次，因为是类的静态成员变量）
    static float mnMinX;
    static float mnMaxX;
    static float mnMinY;
    static float mnMaxY;

	//一个标志，标记是否已经进行了这些初始化计算
    static bool mbInitialComputations;

private:

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
	//根据给出的校正参数对图像的特征点进行去校正操作，由构造函数调用
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(const cv::Mat &imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Rotation, translation and camera center
    cv::Mat mRcw; ///< Rotation from world to camera
    cv::Mat mtcw; ///< Translation from world to camera
    cv::Mat mRwc; ///< Rotation from camera to world
    cv::Mat mOw;  ///< mtwc,Translation from camera to world
};

}// namespace ORB_SLAM

#endif // FRAME_H
