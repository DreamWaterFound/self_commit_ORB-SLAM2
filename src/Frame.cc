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

/**
 * @file Frame.cc
 * @author guoqing (1337841346@qq.com)
 * @brief 帧的实现文件
 * @version 0.1
 * @date 2019-01-03
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{


//下一个生成的帧的ID,这里是初始化类的静态成员变量
long unsigned int Frame::nNextId=0;

//是否要进行初始化操作的标志
//这里给这个标志置位的操作是在最初系统开始加载到内存的时候进行的，下一帧就是整个系统的第一帧，所以这个标志要置位
bool Frame::mbInitialComputations=true;

//TODO 下面这些都没有进行赋值操作，但是也写在这里，是为什么？可能仅仅是前视声明?
//目测好像仅仅是对这些类的静态成员变量做个前视声明，没有发现这个操作有特殊的含义
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;


//无参的构造函数默认为空
Frame::Frame()
{}

/** @details 另外注意，调用这个函数的时候，这个函数中隐藏的this指针其实是指向目标帧的
 */
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), 
     mpORBextractorLeft(frame.mpORBextractorLeft), 
     mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), 
     mK(frame.mK.clone()),									//深拷贝
     mDistCoef(frame.mDistCoef.clone()),					//深拷贝
     mbf(frame.mbf), 
     mb(frame.mb), 
     mThDepth(frame.mThDepth), 
     N(frame.N), 
     mvKeys(frame.mvKeys),									//经过实验，确定这种通过同类型对象初始化的操作是具有深拷贝的效果的
     mvKeysRight(frame.mvKeysRight), 						//深拷贝
     mvKeysUn(frame.mvKeysUn),  							//深拷贝
     mvuRight(frame.mvuRight),								//深拷贝
     mvDepth(frame.mvDepth), 								//深拷贝
     mBowVec(frame.mBowVec), 								//深拷贝
     mFeatVec(frame.mFeatVec),								//深拷贝
     mDescriptors(frame.mDescriptors.clone()), 				//cv::Mat深拷贝
     mDescriptorsRight(frame.mDescriptorsRight.clone()),	//cv::Mat深拷贝
     mvpMapPoints(frame.mvpMapPoints), 						//深拷贝
     mvbOutlier(frame.mvbOutlier), 							//深拷贝
     mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), 
     mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), 
     mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), 					//深拷贝
     mvInvScaleFactors(frame.mvInvScaleFactors),			//深拷贝
     mvLevelSigma2(frame.mvLevelSigma2), 					//深拷贝
     mvInvLevelSigma2(frame.mvInvLevelSigma2)				//深拷贝
{
	//逐个复制，其实这里也是深拷贝
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
			//这里没有使用前面的深拷贝方式的原因可能是mGrid是由若干vector类型对象组成的vector，
			//但是自己不知道vector内部的源码不清楚其赋值方式，在第一维度上直接使用上面的方法可能会导致
			//错误使用不合适的复制函数，导致第一维度的vector不能够被正确地“拷贝”
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
		//这里说的是给新的帧设置Pose
        SetPose(frame.mTcw);
}


// 双目的初始化
Frame::Frame(const cv::Mat &imLeft, 			//左目图像
			 const cv::Mat &imRight, 			//右目图像
			 const double &timeStamp, 			//时间戳
			 ORBextractor* extractorLeft, 		//左侧的特征点提取器句柄
			 ORBextractor* extractorRight, 		//右侧图像的特征点提取器句柄
			 ORBVocabulary* voc, 				//ORB字典句柄
			 cv::Mat &K, 						//相机的内参数矩阵
			 cv::Mat &distCoef, 				//相机的去畸变参数
			 const float &bf, 					//baseline*f
			 const float &thDepth) 				//远点、近点的深度区分阈值
    :mpORBvocabulary(voc),						//下面是对类的成员变量进行初始化
     mpORBextractorLeft(extractorLeft),
     mpORBextractorRight(extractorRight), 
     mTimeStamp(timeStamp), 
     mK(K.clone()),								//注意这里是深拷贝
     mDistCoef(distCoef.clone()), 				//注意这里是深拷贝
     mbf(bf), 
     mb(0), 									//这里将双目相机的基线设置为0其实没有什么道理，因为在其构造函数中mb还是会被正确计算的
     mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))//NOTICE 暂时先不设置参考关键帧
{
    /** 主要步骤: */

    // Frame ID
	/** 1. 分配这个帧的id */
    mnId=nNextId++;

    
    /** 2. 处理图像金字塔的相关参数 */

    // Scale Level Info
	//目测下面的内容是获取图像金字塔的每层的缩放信息，都是左目图像的
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//这个是获得层与层之前的缩放比
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	//计算上面缩放比的对数, NOTICE log=自然对数，log10=才是以10为基底的对数 
    mfLogScaleFactor = log(mfScaleFactor);
	//获取每层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//同样获取每层图像缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//高斯模糊的时候，使用的方差
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    /** 3. 对左目右目图像提取ORB特征.此过程中还开辟了两个线程,调用 Frame::ExtractORB() 函数来执行. */

    // ORB extraction
    // 同时对左右目提特征，还同时开了两个线程
    thread threadLeft(&Frame::ExtractORB,		//该线程的主函数
					  this,						//当前帧对象的对象指针
					  0,						//表示是左图图像
					  imLeft);					//图像数据
	//对右目图像提取ORB特征，参数含义同上
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
	//等待两张图像特征点提取过程完成
    threadLeft.join();
    threadRight.join();

	//mvKeys中保存的是左图像中的特征点，这里是获取左侧图像中特征点的个数
    N = mvKeys.size();

    /** 4. 判断是否成功提取到特征点,如果没有就返回 */

	//如果左图像中没有成功提取到特征点那么就返回，也意味这这一帧的图像无法使用
    if(mvKeys.empty())
        return;
	
    /** 5. 特征点去畸变,使用 Frame::UndistortKeyPoints() 函数.\n实际上由于双目输入的图像已经预先经过矫正,所以实际上并没有对特征点进行任何处理操作 */

    // Undistort特征点，这里没有对双目进行校正，因为要求输入的图像已经进行极线校正
    UndistortKeyPoints();

    /** 6. 计算双目间特征点的匹配\n只有匹配成功的特征点会计算其深度,深度存放在 mvDepth 中. 使用 Frame::ComputeStereoMatches()函数. */
	//应当说，mvuRight中存储的应该是左图像中的点所匹配的在右图像中的点的横坐标（纵坐标相同）；
	//mvDepth才是估计的深度
    ComputeStereoMatches();

    /** 7. 地图点集初始化 \n生成等数量的地图点句柄,由一个成员变量vector mvpMapPoints处理 \n然后默认所有的特征点均为inlier */

    // 对应的mappoints
	//这里其实是生成了一个空的地图点句柄vector，这部分Frame.cpp中没有进行相关操作
    //N个点,每个点的句柄的初始值为<MapPoint*>(NULL)
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));   
	//对于每个地图点，让其为外点的标记清空，先认为都是inlier
    mvbOutlier = vector<bool>(N,false);

    /** 8. 如果当前帧是第一帧,或者刚刚进行了重定位等矫正,那么需要进行特殊的初始化 \n其实就是对一些类的静态成员变量进行赋值\n内容主要包括下面几种: \n
     *      - 计算未矫正图像的边界.  Frame::ComputeImageBounds() \n
     *      - 计算一个像素列相当于几个（<1）图像网格列 \n
     *      - 对相机内参进行赋值
     *      - 完成上述操作之后对标志进行复位
     */


    // This is done only for the first Frame (or after a change in the calibration)
	//检查是否需要对当前的这一帧进行特殊的初始化，这里“特殊的初始化”的意思就是对一些类的静态成员变量进行赋值
    if(mbInitialComputations)
    {
		//计算未校正图像的边界
        ComputeImageBounds(imLeft);

		//计算一个像素列相当于几个（<1）图像网格列
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

		//对这些类的静态成员变量进行赋值，其实前面的那个也是，都是相机的基本内参
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//这里将这个标记恢复false，就说明这个标记的确是标记进行“特殊初始化”用的
        mbInitialComputations=false;
    }//查看“特殊初始化”标记，如果有的话就要进行特殊初始化

    /** 9. 计算相机基线长度 */

    //双目相机的基线长度是在这里被计算的, TODO  为什么要在这里进行计算啊？这个不是一个常量吗对于一个特定的双目相机？
    mb = mbf/fx;

    /** 10. 将提取出来的特征点分配到图像网格中 \n Frame::AssignFeaturesToGrid() */
    AssignFeaturesToGrid();    
}

// RGBD初始化
Frame::Frame(const cv::Mat &imGray, 	//灰度化之后的彩色图像
			 const cv::Mat &imDepth, 	//深度图像
			 const double &timeStamp, 	//时间戳
			 ORBextractor* extractor,	//ORB特征提取器句柄
			 ORBVocabulary* voc, 		//ORB字典句柄
			 cv::Mat &K, 				//相机的内参数矩阵
			 cv::Mat &distCoef, 		//相机的去畸变参数
			 const float &bf, 			//baseline*f
			 const float &thDepth)		//区分远近点的深度阈值
    :mpORBvocabulary(voc),
     mpORBextractorLeft(extractor),
     mpORBextractorRight(static_cast<ORBextractor*>(NULL)),	//实际上这里没有使用ORB特征点提取器
     mTimeStamp(timeStamp), 
     mK(K.clone()),
     mDistCoef(distCoef.clone()), 
     mbf(bf), 
     mThDepth(thDepth)
{
    /** 主要步骤: */
    // Frame ID
	/** 1. 分配当前帧ID */
    mnId=nNextId++;

    /** 2. 计算图像金字塔的相关参数 */
    // Scale Level Info
	//图像层的尺度缩放信息，和双目相机的帧的初始化相同
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
	//计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
	//获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//TODO 也是获取这个不知道有什么实际含义的sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//计算上面获取的sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    /** 3. 提取彩色图像(其实现在已经灰度化成为灰度图像了)的特征点 \n Frame::ExtractORB() */

    // ORB extraction
	//对左侧图像提取ORB特征点
    ExtractORB(0,imGray);

	//获取特征点的个数
    N = mvKeys.size();

    /** 4. 判断是否正确提取出特征点,如果没有则放弃 */

	//如果这一帧没有能够提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    /** 5. 对提取到的特征点进行去畸变操作,根据特征点的深度反推假想中的右侧特征点 \n
     *  去畸变: Frame::UndistortKeyPoints()
     *  恢复假想右图特征点: Frame::ComputeStereoFromRGBD()
    */
	//运行到这里说明以及获得到了特征点，这里对这些特征点进行去畸变操作
    UndistortKeyPoints();

	//获取灰度化后的彩色图像的深度，并且根据这个深度计算其假象的右图中匹配的特征点的视差
    ComputeStereoFromRGBD(imDepth);

    /** 6. 初始化本帧地图点 */

	//初始化存储地图点句柄的vector
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	//然后默认所有的地图点都是inlier
    mvbOutlier = vector<bool>(N,false);

    /** 7. 判断是否需要进行进行特殊初始化,这个过程一般是在第一帧或者是重定位之后进行.主要操作有:\n
     *      - 计算未校正图像的边界 Frame::ComputeImageBounds()
     *      - 计算一个像素列相当于几个（<1）图像网格列
     *      - 给相机的内参数赋值
     *      - 标志复位
     */ 
    // This is done only for the first Frame (or after a change in the calibration)
	//判断是否是需要首次进行的“特殊初始化”
    if(mbInitialComputations)
    {
		//计算未校正图像的边界
        ComputeImageBounds(imGray);

		//计算一个像素列相当于几个（<1）图像网格列
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//对类的静态成员变量进行赋值
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//现在这个“特殊”的初始化过程进行完成了，将这个标志复位
        mbInitialComputations=false;
    }

    //由于后面要对从RGBD相机输入的特征点,结合相机基线长度,焦距,以及点的深度等信息来计算其在假想的"右侧图像"上的匹配点,所以
    //需要计算这个东西
    //TODO 但是好奇的是,mbf是怎么得到的?
    /** 8. 计算假想的基线长度 baseline= mbf/fx */
    mb = mbf/fx;

	/** 9.将特征点分配到图像网格中 \n Frame::AssignFeaturesToGrid() */
    AssignFeaturesToGrid();
}

// 单目初始化
Frame::Frame(const cv::Mat &imGray, 			//灰度化后的彩色图像
			 const double &timeStamp, 			//时间戳
			 ORBextractor* extractor,			//ORB特征点提取器的句柄
			 ORBVocabulary* voc, 				//ORB字典的句柄
			 cv::Mat &K, 						//相机的内参数矩阵
			 cv::Mat &distCoef, 				//相机的去畸变参数
			 const float &bf, 					//baseline*f
			 const float &thDepth)				//区分远近点的深度阈值
    :mpORBvocabulary(voc),
     mpORBextractorLeft(extractor),
     mpORBextractorRight(static_cast<ORBextractor*>(NULL)),	//因为单目图像没有这个右侧图像的定义，所以这里的右图像特征点提取器的句柄为空
     mTimeStamp(timeStamp), 
     mK(K.clone()),
     mDistCoef(distCoef.clone()), 
     mbf(bf), 
     mThDepth(thDepth)
{
    /** 主要步骤: */

    // Frame ID
	/** 1. 获取帧的ID */
    mnId=nNextId++;

    /** 2. 计算图像金字塔的参数 */
    // Scale Level Info
	//和前面相同，这里也是获得图像金字塔的一些属性
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	//计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
	//获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//TODO 获取sigma^2这个不知道有什么用的变量
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//计算sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
	/** 3. 对这个单目图像进行提取特征点操作 \n Frame::ExtractORB() */
    ExtractORB(0,imGray);

	//求出特征点的个数
    N = mvKeys.size();

    /** 4. 查看是否成功提取出特征点,如果没有提取到有效的特征点那么就放弃本帧 */

	//如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

    /** 5. 对提取到的特征点进行矫正 \n Frame::UndistortKeyPoints() */
    // 调用OpenCV的矫正函数矫正orb提取的特征点
    UndistortKeyPoints();

    // Set no stereo information
	/** 6. 由于单目相机无法直接获得立体信息，所以这里要给这个右图像对应点的横坐标和深度赋值-表示没有相关信息 */
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);


    /** 7. 初始化本帧的地图点 */
	//初始化存储地图点句柄的vector
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	//开始认为默认的地图点均为inlier
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
	//和前面一样的，看看是否需要进行特殊的初始化
     /** 8. 判断是否需要进行进行特殊初始化,这个过程一般是在第一帧或者是重定位之后进行.主要操作有:\n
     *      - 计算未校正图像的边界 Frame::ComputeImageBounds() 
     *      - 计算一个像素列相当于几个（<1）图像网格列
     *      - 给相机的内参数赋值
     *      - 标志复位
     */ 
    if(mbInitialComputations)
    {
		//计算未校正图像的边界
        ComputeImageBounds(imGray);

		//这个变量表示一个图像像素列相当于多少个图像网格列
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		//这个也是一样，不多代表是图像网格行
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		//我猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }//是否需要进行特殊的初始化

    /** 9. 计算 basline */
    //目测也是假想的
    mb = mbf/fx;

	/** 10. 将特征点分配到图像网格中 \n Frame::AssignFeaturesToGrid() */
    AssignFeaturesToGrid();
}

//将提取的ORB特征点分配到图像网格中
void Frame::AssignFeaturesToGrid()
{
    /** 步骤: */
    /** 1. 提前给 Frame::mGrid 中存储特征点的vector预分配空间 */
	//这里是提前给那些网格中的vector预分配的空间
	//TODO 可是为什么是平均值的一半呢？仅凭Frame.cpp部分还不足以知道原因
    //猜想也有可能是作者预想的,这样的空间分配上最小,以后运行的时候因为空间不够进行额外分配的时候时间占用也最少
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
	//开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // 在mGrid中记录了各特征点，严格来说应该是各特征点在vector mvKeysUn中的索引
	//对于每个特征点
    /** 2. 开始遍历每个特征点 */
    for(int i=0;i<N;i++)
    {
		//从类的成员变量中获取已经去畸变后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];

		//用于存储某个特征点所在网格的网格坐标
        int nGridPosX, nGridPosY;
		//计算某个特征点所在网格的网格坐标，如果失败的话返回false
        /** 3. 对遍历到的特征点使用 Frame::PosInGrid() 函数确定其所在的网格 */
        if(PosInGrid(kp,nGridPosX,nGridPosY))
			//将这个特征点的索引追加到对应网格的vector中
            /** 4. 如果的确在某个网格中,就将当前遍历到的特征点追加到对应的网格中的vector中 */
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }/** 5. 特征点遍历结束 */
}

//提取图像的ORB特征
void Frame::ExtractORB(int flag, 			//0-左图  1-右图
					   const cv::Mat &im)	//等待提取特征点的图像
{
    /** 步骤:<ul>*/
    /** <li> 判断是左图还是右图 </li> */
    if(flag==0)
        /** <li> 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中 \n
		 * 其实这里的提取器句柄就是一个函数指针...或者说,是运算符更加合适 \n
         * ORBextractor::operator()  </li> */
        (*mpORBextractorLeft)(im,				//待提取特征点的图像
							  cv::Mat(),		//TODO ？？？？ 这个参数的含义要参考这部分的源文件才能知道
                                                //问题是参考了源文件也不清楚,说是掩摸图像,但实际在对应的程序中根本就没有用到
							  mvKeys,			//输出变量，用于保存提取后的特征点
							  mDescriptors);	//输出变量，用于保存特征点的描述子
    else
        /** <li> 右图的话就需要使用右图指定的特征点提取器，并将提取结果保存到对应的变量中  </li> \n ORBextractor::operator()  */
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
	/** </ul> */
	//所以，上面区分左右图的原因就是因为保存结果的变量不同。不过新的疑问是：
	// TODO 左图的特征点提取器和右图的特征点提取器有什么不同之处吗？
    //这个好像的确是有的, 右图的特征点提取貌似是要在左图的基础上进行,来加速特征点的提取过程
}

// 设置相机姿态，随后会调用 UpdatePoseMatrices() 来改变mRcw,mRwc等变量的值
void Frame::SetPose(cv::Mat Tcw)
{
	/** 1. 更改类的成员变量,深拷贝 */
    mTcw = Tcw.clone();
	/** 2. 调用 Frame::UpdatePoseMatrices() 来更新、计算类的成员变量中所有的位姿矩阵 */
    UpdatePoseMatrices();
}

//根据Tcw计算mRcw、mtcw和mRwc、mOw
void Frame::UpdatePoseMatrices()
{
    /** 主要计算四个量. 定义程序中的符号和公式表达: \n
     * Frame::mTcw = \f$ \mathbf{T}_{cw} \f$ \n
     * Frame::mRcw = \f$ \mathbf{R}_{cw} \f$ \n
     * Frame::mRwc = \f$ \mathbf{R}_{wc} \f$ \n
     * Frame::mtcw = \f$ \mathbf{t}_{cw} \f$ 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系\n
     * Frame::mOw  = \f$ \mathbf{O}_{w} \f$  即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系\n  
     * 步骤: */
    /** 1. 计算mRcw,即相机从世界坐标系到当前帧的相机位置的旋转. \n
     * 这里是直接从 Frame::mTcw 中提取出旋转矩阵. \n*/
    // [x_camera 1] = [R|t]*[x_world 1]，坐标为齐次形式
    // x_camera = R*x_world + t
	//注意，rowRange这个只取到范围的左边界，而不取右边界
	//所以下面这个其实就是从变换矩阵中提取出旋转矩阵
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    /** 2. 相反的旋转就是取个逆，对于正交阵也就是取个转置: \n
     * \f$ \mathbf{R}_{wc}=\mathbf{R}_{cw}^{-1}=\mathbf{R}_{cw}^{\text{T}} \f$ \n
     * 得到 mRwc .
     */
    mRwc = mRcw.t();
	/** 3. 同样地，从变换矩阵 \f$ \mathbf{T}_{cw} \f$中提取出平移向量 \f$ \mathbf{t}_{cw} \f$ \n 
     * 进而得到 mtcw. 
     */
    mtcw = mTcw.rowRange(0,3).col(3);
    // mtcw, 即相机坐标系下相机坐标系到世界坐标系间的向量, 向量方向由相机坐标系指向世界坐标系
    // mOw, 即世界坐标系下世界坐标系到相机坐标系间的向量, 向量方向由世界坐标系指向相机坐标系

    //可能又错误,不要看接下来的这一段!!!
	//其实上面这两个平移向量应当描述的是两个坐标系原点之间的相互位置，mOw也就是相机的中心位置吧（在世界坐标系下）
	//上面的两个量互为相反的关系,但是由于mtcw这个向量是在相机坐标系下来说的，所以要反旋转变换到世界坐标系下，才能够表示mOw

    /** 4. 最终求得相机光心在世界坐标系下的坐标: \n
     * \f$ \mathbf{O}_w=-\mathbf{R}_{cw}^{\text{T}}\mathbf{t}_{cw} \f$ \n
     * 使用这个公式的原因可以按照下面的思路思考. 假设我们有了一个点 \f$ \mathbf{P} \f$ ,有它在世界坐标系下的坐标 \f$ \mathbf{P}_w \f$ 
     * 和在当前帧相机坐标系下的坐标 \f$ \mathbf{P}_c \f$ ,那么比较容易有: \n
     * \f$ \mathbf{P}_c=\mathbf{R}_{cw}\mathbf{P}_w+\mathbf{t}_{cw}  \f$ \n
     * 移项: \n
     * \f$ \mathbf{P}_c-\mathbf{t}_{cw}=\mathbf{R}_{cw}\mathbf{P}_w \f$ \n
     * 移项,考虑到旋转矩阵为正交矩阵,其逆等于其转置: \n
     * \f$ \mathbf{R}_{cw}^{-1}\left(\mathbf{P}_c-\mathbf{t}_{cw}\right)=
     * \mathbf{R}_{cw}^{\text{T}}\left(\mathbf{P}_c-\mathbf{t}_{cw}\right) = 
     * \mathbf{P}_w \f$ \n
     * 此时,如果点\f$ \mathbf{P} \f$就是表示相机光心的话,那么这里的\f$ \mathbf{P}_w \f$也就是相当于 \f$ \mathbf{O}_w \f$ 了,
     * 并且形象地可以知道 \f$ \mathbf{P}_c=0 \f$. 所以上面的式子就变成了: \n
     * \f$ \mathbf{O}_w=\mathbf{P}_w=\left(-\mathbf{t}_{cw}\right) \f$ \n
     * 于是就有了程序中的计算的公式. \n
     * 也许你会想说为什么不是 \f$ \mathbf{O}_w=-\mathbf{t}_{cw} \f$,是因为如果这样做的话没有考虑到坐标系之间的旋转.
     */ 
	
    mOw = -mRcw.t()*mtcw;

	/* 下面都是之前的推导,可能有错误,不要看!!!!!
	其实上面的算式可以写成下面的形式：
	mOw=(Rcw')*(-tcw)*[0,0,0]'  (MATLAB写法)（但是这里的计算步骤貌似是不对的）
	这里的下标有些意思，如果倒着读，就是“坐标系下点的坐标变化”，如果正着读，就是“坐标系本身的变化”，正好两者互逆
	所以这里就有：
	tcw:相机坐标系到世界坐标系的平移向量
	-tcw:世界坐标系到相机坐标系的平移向量
	Rcw：相机坐标系到世界坐标系所发生的旋转
	Rcw^t=Rcw^-1:世界坐标系到相机坐标系所发生的旋转
	最后一个因子则是世界坐标系的原点坐标
	不过这样一来，如果是先旋转后平移的话，上面的计算公式就是这个了：
	mOw=(Rcw')*[0,0,0]'+(-tcw)
	唉也确实是，对于一个原点无论发生什么样的旋转，其坐标还是不变啊。有点懵逼。
	会不会是这样：在讨论坐标系变换问题的时候，我们处理的是先平移，再旋转？如果是这样的话：
	mOw=(Rcw')*([0,0,0]'+(-tcw))=(Rcw')*(-tcw)
	就讲得通了
	新问题：这里的mOw和-tcw又是什么关系呢?目前来看好似是前者考虑了旋转而后者没有（后来的想法证明确实是这样的）
	另外还有一种理解方法：
	Pc=Rcw*Pw+tcw
	Pc-tcw=Rcw*Pw
	Rcw'(Pc-tcw)=Pw		然后前面是对点在不同坐标系下的坐标变换，当计算坐标系本身的变化的时候反过来，
	(这一行的理解不正确)将Pc认为是世界坐标系原点坐标，Pw认为是相机的光心坐标
    应该这样说,当P点就是相机光心的时候,Pw其实就是这里要求的mOw,Pc明显=0
	Rcw'(o-tcw)=c
	c=Rcw'*(-tcw)			就有了那个式子
	**/
}

// 判断路标点是否在视野中
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    /** 步骤: \n <ul>*/
    /**  <li> 1.默认设置标志 MapPoint::mbTrackInView 为否,即设置该地图点不进行重投影. </li>\n
     * mbTrackInView是决定一个地图点是否进行重投影的标志，这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个 
     * 验证关卡。这里默认设置为否. \n
     */
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    /** <li> 2.获得这个地图点的世界坐标, 使用 MapPoint::GetWorldPos() 来获得。</li>\n*/
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    /** <li> 3.然后根据 Frame::mRcw 和 Frame::mtcw 计算这个点\f$\mathbf{P}\f$在当前相机坐标系下的坐标: </li>\n
     * \f$ \mathbf{R}_{cw}\mathbf{P}+\mathbf{t}_{cw} \f$ \n
     * 并提取出三个坐标的坐标值.
     */ 
    
    const cv::Mat Pc = mRcw*P+mtcw; // 这里的Rt是经过初步的优化后的
    //然后提取出三个坐标的坐标值
    const float &PcX = Pc.at<float>(0);
    const float &PcY = Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    /** <li> 4. <b>关卡一</b>：检查这个地图点在当前帧的相机坐标系下，是否有正的深度.如果是负的，就说明它在当前帧下不在相机视野中，也无法在当前帧下进行重投影. </li>*/
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    /** <li> 5. <b>关卡二</b>：将MapPoint投影到当前帧, 并判断是否在图像内（即是否在图像边界中）。</li>\n
     * 投影方程： \n
     * \f$ \begin{cases}
     * z^{-1} &= \frac{1}{\mathbf{P}_c.Z} \\
     * u &= z^{-1} f_x \mathbf{P}_c.X + c_x \\
     * v &= z^{-1} f_y \mathbf{P}_c.Y + c_y 
     * \end{cases} \f$
     */ 
    // V-D 1) 将MapPoint投影到当前帧, 并判断是否在图像内
    const float invz = 1.0f/PcZ;			//1/Z，其实这个Z在当前的相机坐标系下的话，就是这个点到相机光心的距离，也就是深度
    const float u=fx*PcX*invz+cx;			//计算投影在当前帧图像上的像素横坐标
    const float v=fy*PcY*invz+cy;			//计算投影在当前帧图像上的像素纵坐标

    //判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    // V-D 3) 计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内
    /** <li> 6. <b>关卡三</b>：计算MapPoint到相机中心的距离, 并判断是否在尺度变化的距离内 </li>  \n
     * 这里所说的尺度变化是指地图点到相机中心距离的一段范围，如果计算出的地图点到相机中心距离不在这个范围的话就认为这个点在 
     * 当前帧相机位姿下不能够得到正确、有效、可靠的观测，就要跳过. \n
     * 为了完成这个任务有两个子任务：\n <ul>
     */ 
     
     /** <li> 6.1 得到认为的可靠距离范围。</li> \n
     *  这个距离的上下限分别通过 MapPoint::GetMaxDistanceInvariance() 和 MapPoint::GetMinDistanceInvariance() 来得到。 */
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();

    /** <li> 6.2 得到当前3D地图点距离当前帧相机光心的距离。</li> \n
     * 具体实现上是通过构造3D点P到相机光心的向量 \f$\mathbf{P}_0 \f$ ，通过对向量取模即可得到距离\f$dist\f$。
     * </ul>
     */
    const cv::Mat PO = P-mOw;
	//取模就得到了距离
    const float dist = cv::norm(PO);

	//如果不在允许的尺度变化范围内，认为重投影不可靠
    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    // V-D 2) 计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回
    /** <li> 7. <b>关卡四</b>：计算当前视角和平均视角夹角的余弦值, 若小于cos(60), 即夹角大于60度则返回 </li>     
     * <ul>
     */ 
    /** <li> 7.1 使用 MapPoint::GetNormal() 来获得平均视角(其实是一个单位向量\f$ \mathbf{P}_n \f$ ) </li> */
	//获取平均视角，目测这个平均视角只是一个方向向量，模长为1，它表示了当前帧下观测到的点的分布情况
	//TODO 但是这个平均视角估计是在map.cpp或者mapoint.cpp中计算的，还不是很清楚这个具体含义  其实现在我觉得就是普通的视角的理解吧
    cv::Mat Pn = pMP->GetNormal();

	/** <li> 7.2 计算当前视角和平均视角夹角的余弦值，注意平均视角为单位向量 </li>  \n
     * 其实就是初中学的计算公式： \n
     * \f$  viewCos= {\mathbf{P}_0 \cdot \mathbf{P}_n }/{dist} \f$
    */
    const float viewCos = PO.dot(Pn)/dist;

    /** <li> 7.3 然后判断视角是否超过阈值即可。 </li></ul> */
	//如果大于规定的阈值，认为这个点太偏了，重投影不可靠，返回
    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    // V-D 4) 根据深度预测尺度（对应特征点在一层）
    /** <li> 8. 经过了上面的重重考验，说明这个地图点可以被重投影了。接下来需要记录关于这个地图点的一些信息：</li> <ul>*/
     
    //注意在特征点提取的过程中,图像金字塔的不同的层代表着特征点的不同的尺度
    /** <li> 8.1 使用 MapPoint::PredictScale() 来预测该地图点在现有距离 \f$ dist \f$ 下时，在当前帧
     * 的图像金字塔中，所可能对应的尺度。\n
     * 其实也就是预测可能会在哪一层。这个信息将会被保存在这个地图点对象的 MapPoint::mnTrackScaleLevel 中。
    */
    const int nPredictedLevel = pMP->PredictScale(dist,		//这个点到光心的距离
												  this);	//给出这个帧

    // Data used by the tracking	
    /** <li> 8.2 通过置位标记 MapPoint::mbTrackInView 来表示这个地图点要被投影 </li> */
    pMP->mbTrackInView = true;	
    /** <li> 8.3 然后计算这个点在左侧图像和右侧图像中的横纵坐标。 </li> 
     * 地图点在当前帧中被追踪到的横纵坐标其实就是其投影在当前帧上的像素坐标 u,v => MapPoint::mTrackProjX,MapPoint::mTrackProjY \n
     * 其中在右侧图像上的横坐标 MapPoint::mTrackProjXR 按照公式计算：\n
     * \f$ X_R=u-z^{-1} \cdot mbf \f$ \n
     * 其来源参考SLAM十四讲的P51式5.16。
    */
    pMP->mTrackProjX = u;				//该地图点投影在左侧图像的像素横坐标
    pMP->mTrackProjXR = u - mbf*invz; 	//bf/z其实是视差，为了求右图像中对应点的横坐标就得这样减了～
										//这里确实直接使用mbf计算会非常方便
    pMP->mTrackProjY = v;				//该地图点投影在左侧图像的像素纵坐标

    pMP->mnTrackScaleLevel = nPredictedLevel;		//TODO 根据上面的计算来看是存储了根据深度预测的尺度，但是有什么用不知道
    /** <li> 8.4 保存当前视角和平均视角夹角的余弦值 </li></ul>*/
    pMP->mTrackViewCos = viewCos;					

    //执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
    return true;

    /** </ul> */
}

//找到在 以x,y为中心,半径为r的圆形内且在[minLevel, maxLevel]的特征点
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    /** 步骤： <ul> */
	//生成用于存储搜索结果的vector
    vector<size_t> vIndices;
	//预分配空间
    vIndices.reserve(N);

	
    /** <li> 1.<b>检查一</b>：检查圆形区域是否在图像中，具体做法是分别求圆形搜索区域的上下左右四个边界是否能够满足图像的边界条件。 </li> \n
     * 这里的边界条件以圆的左边界为例，就是首先求出左边界所在的图像网格列，然后判断这个网格列位置是否超过了图像网格的上限。类似这样：\n
     * <img src="../imgs/1.png" alt="图例">
    */

   //下面的这段计算的代码其实可以这样理解：
	//首先(mnMaxX-mnMinX)/FRAME_GRID_COLS表示每列网格可以平均分得几个像素坐标的列
	//那么它的倒数，就可以表示每个像素列相当于多少（<1）个网格的列
	//而前面的(x-mnMinX-r)，可以看做是从图像的左边界到半径r的圆的左边界区域占的像素列数
	//两者相乘，就是求出那个半径为r的圆的左侧边界在那个网格列中。这个变量的名其实也是这个意思
    const int nMinCellX = max(0,												//这个用来确保最后的值>0
							  //mnMinX是图像的边界
							  (int)floor(			//floor，小于等于X的最大整数
								  //mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX)
								  (x-mnMinX-r)*mfGridElementWidthInv)
								);
	//如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

	//NOTICE 注意这里的网格列也是从0开始编码的
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,		//最右侧的网格列id
							  (int)ceil(					//ceil，大于X的最小整数
									//这里的算式其实是和上面非常相近的，把-r换成了+r
								  (x-mnMinX+r)*mfGridElementWidthInv));
	//如果计算出的圆右边界所在的网格不合法，也说明找不到要求的特征点，直接返回空vector
    if(nMaxCellX<0)
        return vIndices;

	//后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id，不再注释
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    /** <li> 2. <b>检查二</b>:检查需要搜索的图像金字塔层数范围是否符合要求 </li> */
	//可是如果bCheckLevels==0就说明minLevel<=0且maxLevel<0,或者是只要其中有一个层大于0就可以
	//TODO 这又意味着什么嘞？层为负的有什么意义？这个需要阅读ORB特征提取那边儿才能够理解
	//注意这里的minLevel、maxLevel都是函数的入口参数
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    /** <li> 3. 遍历圆形区域内的所有网格  </li> <ul>*/

	//开始遍历指定区域内的所有网格（X方向）
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
		//开始遍历指定区域内的所有网格（Y方向）
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            /** <li> 3.1 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引,其实也就是得到校正后的特征点。</li>*/
            const vector<size_t> vCell = mGrid[ix][iy];
			/** <li> 3.2 如果这个图像网格中没有特征点，那么就直接跳过这个网格. </li> */
            if(vCell.empty())
                continue;

            /** <li> 3.3 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点 </li> <ul>*/
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
				/** <li> 3.3.1 根据索引先读取这个特征点 </li> */
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
				/** <li> 3.3.2 如果给定的搜索图层范围合法，则检查这个特征点是否是在给定搜索图层范围内生成的</li> */
                if(bCheckLevels)
                {
					//那么就检查层
					//cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
					//@[https://www.cnblogs.com/cj695/p/4041399.html]
					//查看提取数据的那一层特征点是否在minLevel和maxLevel之间
                    if(kpUn.octave<minLevel)
						//如果不是的话，跳过这个特征点
                        continue;
                    if(maxLevel>=0)		//TODO 为什么要强调这一点？为什么要强调给定的搜索层范围必须大于0?
                        if(kpUn.octave>maxLevel)
							//如果不是的话，跳过这个特征点
                            continue;
                }//检查这个特征点是否在指定的图像金字塔层范围之间
                

                //通过检查，说明当前遍历到的这个特征点在指定的图像金字塔层范围之间
                /** <li> 3.3.3 计算这个特征点到指定的搜索中心的距离（x方向和y方向），查看是否是在这个圆形区域之内。
                 * 在的话就追加到结果的vector中。</li>*/
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;
				//NOTICE 在这里，kpUn中存储的就已经是变换后的坐标、相对于整张图片来讲的坐标

				//如果x方向和y方向的距离都在指定的半径之内，
                if(fabs(distx)<r && fabs(disty)<r)
					//那么说明这个特征点就是我们想要的！！！将它追加到结果vector中
                    vIndices.push_back(vCell[j]);
            }//遍历这个图像网格中所有的特征点
            /** </ul> */
        }//开始遍历指定区域内的所有网格（Y方向）
    }//开始遍历指定区域内的所有网格（X方向） 
    /** </ul> */

    //返回搜索结果
    return vIndices;
    /** </ul> */
}


//计算指定特征点属于哪个图像网格
bool Frame::PosInGrid(			//返回值：true-说明找到了指定特征点所在的图像网格  false-说明没有找到
	const cv::KeyPoint &kp,		//输入，指定的特征点
	int &posX,int &posY)		//输出，指定的图像特征点所在的图像网格的横纵id（其实就是图像网格的坐标）
{
    /** 这段函数的想法很简单，就是利用划分网格时的图像边界坐标 Frame::mnMinX Frame::mnMinY 和特征点本身的坐标来计算到图像边界需要多少个图像网格列，
     * 从而得到这个特征点的归属。\n
     * 当然具体计算的时候需要检查这个特征点是否能够正确地落到划分的图像网格中。 \n
     * @todo 但是我不明白的是，这里不应该使用四舍五入啊。。。这里应该使用完全去除小数部分的取整方法啊，可能与图像网格的设计有关 
    */

	//std::round(x)返回x的四舍五入值
	//根据前面的分析，mfGridElementWidthInv就是表示一个像素列相当于多少个（<1）图像网格列
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
		//如果最后计算出来的所归属的图像网格的坐标不合法，那么说明这个特征点的坐标很有可能是没有经过校正，
		//因此落在了图像的外面，返回false表示确定失败
        return false;

	//返回true表示计算成功
    return true;
}

//计算词包 mBowVec 和 mFeatVec
void Frame::ComputeBoW()
{
	
    /** 这个函数只有在当前帧的词袋是空的时候才回进行操作。步骤如下:<ul> */
    if(mBowVec.empty())
    {
		/** <li> 1.要写入词袋信息,将以OpenCV格式存储的描述子 Frame::mDescriptors 转换成为vector<cv::Mat>存储</li> */
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
		/** <li> 2. 将特征点的描述子转换成为当前帧的词袋 </li> */
        mpORBvocabulary->transform(vCurrentDesc,	//当前的描述子vector
								   mBowVec,			//输出，词袋向量
								   mFeatVec,		//输出，保存有特征点索引的特征 vector
								   4);				//获取某一层的节点索引
		//@todo 为什么这里要指定“获取某一层的节点索引”？
    }//判断当前帧的词袋是否是空的
    /** </ul> */
}

//对特征点去畸变
void Frame::UndistortKeyPoints()
{
    /** 步骤如下：<ul>*/
    
    /** <li> 1. 如果图像校正过，那么就直接赋值 Frame::mvKeysUn = Frame::mvKeys 不再进行其他操作 </li> */
	//变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3,k4,k5,k6[,s1,s2,s3,s4[,\taux,\tauy...)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
		//跳过后面的操作
        return;
    }//判断图像是否已经矫正过


    /** <li> 2. 如果图像没有矫正过，那么就要准备进行矫正。 </li> */
    /** 但是为了避免在矫正的过程中丢失不必要的信息，这里新建了一个 Frame::N x 2 的矩阵用来暂时存储待去畸变的特征点的坐标 */
    // Fill matrix with points，其实就是将每个特征点的坐标保存到一个矩阵中
    // N为提取的特征点数量(Frame类的成员变量)，将N个特征点保存在N*2的mat中
    cv::Mat mat(N,2,CV_32F);
	//遍历每个特征点
    for(int i=0; i<N; i++)
    {
		//然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }//遍历每个特征点，并将它们的坐标保存到矩阵中

    // Undistort points
    // 调整mat的通道为2，矩阵的行列形状不变
    //这个函数的原型是：cv::Mat::reshape(int cn,int rows=0) const
    //其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
    //不过根据手册发现这里的修改通道只是在逻辑上修改，并没有真正地操作数据
    //这里调整通道的目的应该是这样的，下面的undistortPoints()函数接收的mat认为是2通道的，两个通道的数据正好组成了一个点的两个坐标
    /** <li> 3. 为了能够直接调用opencv的函数来去畸变，需要先将刚才的临时矩阵调整为2通道 </li> */
    mat=mat.reshape(2);
    /** <li> 4. 调用 cv::undistortPoints() 函数，结合去畸变参数 Frame::mDistCoef 来对这些点进行去畸变矫正 </li> */
    cv::undistortPoints(	// 用cv的函数进行失真校正
		mat,				//输入的特征点坐标
		mat,				//输出的特征点坐标，也就是校正后的特征点坐标， NOTICE 并且看起来会自动写入到通道二里面啊
		mK,					//相机的内参数矩阵
		mDistCoef,			//保存相机畸变参数的变量
		cv::Mat(),			//一个空的cv::Mat()类型，对应为函数原型中的R。Opencv的文档中说如果是单目相机，这里可以是恐惧真
		mK); 				//相机的内参数矩阵，对应为函数原型中的P
	
	/** <li> 5. 然后调整回只有一个通道，回归我们正常的处理方式 </li> */
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    // 存储校正后的特征点
    /** <li> 6.然后将得到的去畸变的点的坐标和原来的信息合并，存储在 Frame::mvKeysUn 中 </li> */
	//预分配空间
    mvKeysUn.resize(N);
	//遍历每一个特征点
    for(int i=0; i<N; i++)
    {
		//根据索引获取这个特征点
		//注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
		//读取校正后的坐标并覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
		//然后送回保存
        mvKeysUn[i]=kp;
    }//遍历每一个特征点
    /** </ul> */
}

//计算去畸变图像的边界
void Frame::ComputeImageBounds(const cv::Mat &imLeft)	//参数是需要计算边界的图像
{
    

    /** 步骤如下： <ul> */
    /** <li> 1. 首先判断是是否已经经过了校正操作了 </li>*/
    if(mDistCoef.at<float>(0)!=0.0)
	{
        /** <li> 2. 如果图像没有进行矫正操作，那么就要先保存矫正前的四个边界点坐标：  </li>
         * \n (0,0) (cols,0) (0,rows) (cols,rows)
         * \n 然后结合去畸变参数 Frame::mDistCoef 调用 cv::undistortPoints() 进行去畸变操作。
         * \n 注意校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界。
        */
		
		//保存四个边界点的变量
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0;         //左上
		mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; //右上
		mat.at<float>(1,1)=0.0;
		mat.at<float>(2,0)=0.0;         //左下
		mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; //右下
		mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
		//然后是和前面校正特征点一样的操作，将这几个边界点当做特征点进行校正，使用opencv的函数对这几个初步的边界点进行校正
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

		//校正后的四个边界点已经不能够围成一个严格的矩形，因此在这个四边形的外侧加边框作为坐标的边界
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));//左上和左下横坐标最小的
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));//右上和右下横坐标最大的
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));//左上和右上纵坐标最小的
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));//左下和右下纵坐标最小的
    }
    else
    {
        /** <li> 3. 如果图像已经校正过，那么就直接获得图像边界  </li>*/
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }//判断图像是否已经矫正过，从而采取不同的获得图像边界的策略

    /** </ul> */
}

/*
 * 双目匹配函数
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n ‘
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n 
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一个单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就越准确
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
    /** 这个算法比较复杂。步骤如下： <ul> */

    //初始化用作输出的类成员变量，初始值均为-1
	mvuRight = vector<float>(N,-1.0f);		//存储在右图像中匹配到的特征点坐标
    mvDepth = vector<float>(N,-1.0f);		//以及这对特征点的深度估计

    /** <li> 1. 按下式计算ORB距离的阈值。如果匹配的ORB特征点描述子的距离大于这个阈值，我们就认为是错误的匹配 </li> 
     * \n \f$ distance = (1/2)\cdot (TH_{HIGH}+TH_{LOW}) \f$
     * \n 其中两个阈值分别为 ORBmatcher::TH_HIGH 和 ORBmatcher::TH_LOW 
     * \n 这个阈值在下面进行双目特征点匹配的时候会用到。只有得出的初步匹配的特征点的描述子距离小于这个阈值的时候，我们才能够认为这对匹配是
     * 正确的匹配，我们才能够进行下一步的SAD算法的操作。
    */
	//TODO 但是我还是不明白为什么要按下式计算
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    /** <li> 2. 通过获取 ORBextractor::mvImagePyramid 也就是图像金字塔第0层的行数得到整个图像的行数 </li> */
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    /** <li> 3.<b>关键步骤一</b> 建立特征点搜索范围对应表，一个特征点在一个带状区域内搜索匹配特征点 </li> 
     * \n 匹配搜索的时候，不仅仅是在一条横线上搜索，而是在一条横向搜索带上搜索,简而言之，原本每个特征点的纵坐标为1，
     * 这里把特征点体积放大，纵坐标占好几行, 例如左目图像某个特征点的纵坐标为20，那么在右侧图像上搜索时是在纵坐标为
     * 18到22这条带上搜索，搜索带宽度为正负2，搜索带的宽度和特征点所在金字塔层数有关.
     * \n 
     * 简单来说，如果纵坐标是20，特征点在图像第20行，那么认为18 19 20 21 22行都有这个特征点:
     * \n 
     * 变量 vRowIndices[18]、vRowIndices[19]、vRowIndices[20]、vRowIndices[21]、vRowIndices[22]都有这个特征点编号。
     * \n
     * \n 在具体操作上: <ul>
     */    

	//存储每行所能够匹配到的特征点的索引，说白了就是在右图就是每行一个vector，存储可能存在的特征点id
    vector<vector<size_t> > vRowIndices(nRows,				//元素个数
										vector<size_t>());	//初始值

	//给每一行存储匹配特征点索引的vector预分配200个特征点的存储位置
	//TODO 这里的数字200是随机给定的吗？还是说每个图片采集的特征点有数目要求，这里根据这个数目要求取了一个约数？
    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

	//获取右图图像中提取出的特征点的个数
    const int Nr = mvKeysRight.size();

	/** <li > 首先遍历右图中所有提取到的特征点，对每个特征点进行如下操作： </li> <ul>*/
    for(int iR=0; iR<Nr; iR++)
    {
        // !!在这个函数中没有对双目进行校正，双目校正是在外层程序中实现的 --  不是，要求输入的图像就已经进行了双目矫正
		//获取右图中特征点
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        // 计算匹配搜索的纵向宽度，尺度越大（层数越高，距离越近），搜索范围越大 （这里的意思是，层数越高，说明尺度越大）
        // 如果特征点在金字塔第一层，则搜索范围为:正负2
        // 尺度越大其位置不确定性越高，所以其搜索半径越大
		//这里想想也能够明白，在图像金字塔的高层，一个像素其实对应到底层是好几个像素
		//到这里应该是默认右侧图像已经进行了特征点的提取
		//计算这个搜索范围
        //这里的.octave属性就是特征点所在的金字塔层
        /** <li> 获得该特征点所在的金字塔层数，并且从 Frame::mvScaleFactors 中得到这层金字塔的缩放系数  </li> */
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        /** <li> 然后这个特征点的纵坐标±这个系数，取整，得到搜索界的上下界 </li> */
        const int maxr = ceil(kpY+r);		//保存大于这个数的最小整数，搜索带下界
        const int minr = floor(kpY-r);		//保存小于等于这个数的最小整数，搜索带上界

		//使搜索带范围内的vector记录这个特征点的索引，注意这个特征点是在右图中的
		//最后生成的这个东西叫做“搜索范围对应表”
		//在这里将当前这个特征点的id标记在搜索带对应的所有行上
        /** <li> 在搜索界中的每一行标记这个特征点的索引，表示在某一行这个特征点会出现 */
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }//遍历右图中所有提取到的特征点
    /** </ul> */
    /** </ul> */

    /** <li> <b>关键步骤二</b> 对左目相机每个特征点，通过描述子在右目带状搜索区域找到匹配点, 再通过SAD做亚像素匹配 </li>
     * \n 详细步骤如下： <ul>
    */

    // Set limits for search
    /** <li> 设置一些变量： </li> <ul>*/
    /** <li> 设置最小深度等于基线长度 </li> */
    const float minZ = mb;			//NOTE bug mb没有初始化，mb的赋值在构造函数中放在 ComputeStereoMatches 函数的后面
									//TODO 回读代码的时候记得检查上面的这个问题
									//这个变量对应着允许的空间点在当前相机坐标系下的最小深度
									//TODO 新问题：这里设置最小深度等于基线的原因是什么?自己的猜测，可能和相机的视角有一定关系
                                    //60度范围内观测的比较准?超过60度之后得到的点可信度比较差
    /** <li> 最小视差, 设置为0即可  对应着最大深度 </li> */
    const float minD = 0;			 
    /** <li> 最大视差, 对应最小深度，可以计算得到 </li> 
     * \f$ \frac{mbf}{minZ} \f$
     * \n 相关公式@视觉SLAM十四讲P91 式5.16
    */
    const float maxD = mbf/minZ;  	// 最大视差, 对应最小深度 mbf/minZ = mbf/mb = mbf/(mbf/fx) = fx (wubo???)									
    /** </ul> */

    // For each left keypoint search a match in the right image
    //用于记录匹配点对的vector
    //NOTE 不过这里的两个 int 在最后分别存储的内容是，最佳距离 bestDist (SAD匹配最小匹配偏差)，以及对应的左图特征点的id iL
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    
    // NOTICE 注意：这里是校正前的mvKeys，而不是校正后的mvKeysUn
    // NOTICE KeyFrame::UnprojectStereo和Frame::UnprojectStereo函数中不一致
    // 这里是不是应该对校正后特征点求深度呢？(wubo???)
	//NOTE 自己的理解：这里是使用的双目图像，而默认地，双目图像的校正在驱动程序那一关就已经进行了吧？ 
    /** <li> 遍历左图中的特征点，计算当前遍历的这个左图中的特征点在右图中的可能的匹配范围  </li> <ul>*/
    for(int iL=0; iL<N; iL++)
    {
		/** <li> 获取这个特征点，并且判断在右图中这个特征点所在的行中可能的匹配到的特征点 </li> */
		const cv::KeyPoint &kpL = mvKeys[iL];
		//获取这个特征点所在的金字塔图像层
        const int &levelL = kpL.octave;
		//同时获取这个特征点在左图中的像素坐标，应当注意这里是该特征点在图像金字塔最底层的坐标
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 获取这个特征点所在行，在右图中可能的匹配点，注意这里 vRowIndices 存储的是右图的特征点索引
        const vector<size_t> &vCandidates = vRowIndices[vL];

        /** <li> 如果当前的左图中的这个特征点所在的行，右图中同样的行却没有可能的匹配点，那么就跳过当前的这个点 </li> */
        if(vCandidates.empty())
            continue;

		//执行到这说明当前这个左图中的特征点在右图中是有可能匹配的特征点的,接下来计算匹配范围，其实就是在横向上的、合法的匹配范围
        /** <li> 如果匹配存在，那么根据刚刚计算过的最大最小视差我们可以得到这个点在右图中，在u轴方向（也就是横向）
         * 上的可能的坐标范围 minU ~ maxU </li> */
        const float minU = uL-maxD; // 最小匹配范围，结果小于uL
        const float maxU = uL-minD; // 最大匹配范围，这个结果其实还是uL，因为前面设置了minD=0;

        //其实就算是maxD>uL导致minU<0也是可以的，大不了我就不用管坐标为负的部分；但是如果maxU也小于0，此时的搜索范围就已经不再
        //正常的图片里面了
        /** <li> 然后判断这个特征点在右图中可能的匹配范围，是否在图像边界之内。</li> 
         * \n 不过实际计算的时候只要 maxU < 0 我们就可以认为它不在这个图像内。
        */
        if(maxU<0)
			//放弃对这个点的右图中的匹配点的搜索
            continue;
        /** </ul> */

		//最佳的ORB特征点描述子距离
        /** <li> 设置最佳的ORB特征点描述子距离为 ORBmatcher::TH_HIGH ,并且取得当前左图特征点对应的特征点描述子 </li> */
        int bestDist = ORBmatcher::TH_HIGH;
		//对应的右图中最佳的匹配点ID
        size_t bestIdxR = 0;
        // 每个特征点描述子占一行，建立一个指针指向iL特征点对应的描述子
		//这里的dL是只有一行的
        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        /** <li> 遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小） </li> <ul>*/
        // 步骤2.1：遍历右目所有可能的匹配点，找出最佳匹配点（描述子距离最小）
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            /** <li> 取得和当前左目特征点对应着的可能匹配的每个右目中的特征点，并且检查： </li> <ul> */
			//取可能的匹配的右目特征点ID
            const size_t iR = vCandidates[iC];
			//根据这个ID得到可能的右目匹配特征点
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            /** <li> 检查两个点是否在相邻的尺度中被提取的，也就是检查图像金字塔层数。 </li>
             * 如果两个特征点匹配但是它们所在的图像金字塔的层数相差过多，也认为是不可靠的
             */
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

			//当前这个可能的匹配的右目特征点的横坐标
            const float &uR = kpR.pt.x;

            /** <li> 判断这个右目特征点的横坐标是否满足前面计算出来的搜索范围 </li> 
             * 如果在的话，就计算并更新当前所有匹配点的最短描述子距离信息，并且记录具有最短描述子距离的那个特征点的索引，为下面
             * 选取正确的特征点做准备。
            */
            if(uR>=minU && uR<=maxU)
            {
				//如果在的话，获取这个右目特征点的描述子
                const cv::Mat &dR = mDescriptorsRight.row(iR);
				//并计算这两个特征点的描述子距离
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

				//更新最佳的描述子距离信息
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }//更细最佳的描述子距离信息
            }//判断这个右目特征点的横坐标是否满足前面计算出来的搜索范围
            /** </ul> */
            //如果那个右目特征点的横坐标不满足前面计算出来的搜索范围的话，就直接跳过对当前的这个点的操作了
        }//遍历右目所有可能的匹配点
        // 最好的匹配的匹配误差存在bestDist，匹配点位置存在bestIdxR中
        /** </ul> */

        // Subpixel match by correlation
        // 步骤2.2：通过SAD匹配提高像素匹配修正量bestincR
        
        //如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        if(bestDist<thOrbDist)
        {
            /** <li> 如果刚才匹配过程中得到的最佳描述子距离小于上面给定的阈值，那么接下来就要通过SAD匹配提高像素匹配修正量 bestincR </li> <ul> */
            /** <li> 首先得到这个左图特征点和这个“具有最佳描述子距离”的右图特征点，在原始图像下的坐标 </li> */
            // coordinates in image pyramid at keypoint scale
			//下面的这些参数是特征点的尺度所在的图像金字塔的层的坐标
            // kpL.pt.x对应金字塔最底层坐标，将最佳匹配的特征点对尺度变换到尺度对应层 (scaleduL, scaledvL) (scaleduR0, )
            const float uR0 = mvKeysRight[bestIdxR].pt.x;				//原始的特征点横坐标（右图）
            const float scaleFactor = mvInvScaleFactors[kpL.octave];	//获取该特征点所在层的缩放因子倒数 
            //下面是将特征点的坐标按照图像金字塔的缩放因子缩放之后得到的新坐标，即在特征点所对应的图像金字塔中的缩放的图像下的坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);			
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);				//上一步结束后，右图中这个点的基准值（缩放变换后）

            // sliding window search
			//W在这里可以理解为滑动窗口的大小
            const int w = 5; // 滑动窗口的大小11*11 注意该窗口取自resize后的图像
							//这个11*11的滑动窗口需要结合下面的语句才能够看出来
							//这里的resize指的是上面通过图像金字塔的缩放因子进行缩放变换。变换后的图像存储于特征点
							//提取器中的mvImagePyramid向量中。
            /** <li> 在左图中取以特征点为中心，在该左特征点被提取的金字塔图层中，w=5 为半径的小窗口图像(11x11)，并且在这个小图像中，计算所有像素和中心像素灰度值
             * 之差，得到一个所谓的“灰度中心归一化”图像（我自己起的名称） </li> */                            
			//首先在【左图】中取小窗口中的图像
            //scaledvL是点在金字塔的那个被缩小了的图像中的坐标
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
			//由于下面要进行简单的光度归一化，会有浮点数出现，所以在这里要将图像小窗中整型的数据转换成为浮点型的数据
            IL.convertTo(IL,CV_32F);
            //窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响
			IL = IL - IL.at<float>(w,w) * cv::Mat::ones(IL.rows,IL.cols,CV_32F);
			//这里是提取出窗口的中心像素灰度，然后于整个窗口图像作差,后面乘的单位矩阵是为了满足矩阵的计算法则
			//自己起个名字叫做“灰度中心归一化”

			//初始化“最佳距离”为整型数据所能够表示的最大值
            int bestDist = INT_MAX;
			//inc是increase的缩写，这个表示经过SAD算法后得出的“最佳距离”所对应的修正量
            int bestincR = 0;
			//滑动窗口的滑动范围为（-L, L）
            const int L = 5;
			//这里存储的是SAD算法意义上的距离
            vector<float> vDists;
			//调整大小，+1中的1指的是中心点
            vDists.resize(2*L+1); // 11

            /** <li> 滑动窗口的滑动范围为（-L, L）(L=5),提前判断滑动窗口滑动过程中是否会越界 </li> */
            //REVIEW如果将下面两个变量理解为,相对于滑动窗口左边界或者上边界,窗口中点偏移量的范围,这么说就是正确的了
            const float iniu = scaleduR0+L-w; 	//这个地方是否应该是scaleduR0-L-w (wubo???) 
            const float endu = scaleduR0+L+w+1;	//这里+1是因为下面的*.cols，这个列数是从1开始计算的吧
			//这里的确是在判断是否越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

			//
            /** <li> 遍历滑动窗口在【右图】中所有可能在的位置，下面的操作就都是在右图上进行的了 </li>
             * 其实是在右图中,横向滑动这个窗口 <ul>*/
            for(int incR=-L; incR<=+L; incR++)
            {
                /** <li> 获取在右图上滑动这个窗口所得到的图像 </li> */
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
				//将整型的图像数据转换为浮点型
                IR.convertTo(IR,CV_32F);
				/** <li> 然后进行灰度中心归一化(现在感觉叫做作差会更加合适) </li>
                 * 窗口中的每个元素减去正中心的那个元素，简单归一化，减小光照强度影响
                 */
                IR = IR - IR.at<float>(w,w) * cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                /** <li> 然后计算两个滑动窗口图像的差,得到它的一范数,称之为两张图像的图像距离 </li> */
				//对于这里的cv::norm函数，opencv给出的头文件注释的是：
				//computes norm of selected part of the difference between two arrays
                float dist = cv::norm(IL,IR,cv::NORM_L1); // 一范数，计算差的绝对值
                /** <li> 保存图像距离,并且更新最佳的图像距离,同时保存具有这个最佳图像距离的滑动窗口修正量 </li> 
                 * 其实这里的修正量就是当前的这个滑动窗口中心点相对于原始滑动窗口中心点的横向位移
                */
                if(dist<bestDist)
                {
                    bestDist = dist;// SAD匹配目前最小匹配偏差
                    bestincR = incR; // SAD匹配目前最佳的修正量
                }
                //L+incR体现了修正量的思想，这里是保存在不同修正量的时候的SAD距离
                vDists[L+incR] = dist; 	// 正常情况下，这里面的数据应该以抛物线形式变化
										//NOTE 凭感觉，在一个非常小的区域（10x10pixel）内，这个距离的变化的确是类似于抛物线的。
										//但是不一定是严格的抛物线吧
                /** </ul> */
            }

            //
            //
            /** <li> 判断最佳修正量是否出现在滑动窗口的两侧.如果出现了说明有问题,直接放弃求该特征点的深度 </li> 
             * 这里因为,通过刚刚的SAD过程所得到的,在右图中的特征点的像素位置修正量,随着滑动窗口在u轴(-L,L)上滑动时,
             * 是以抛物线的形式变化，所以在这个滑动区间的两端是不应该出现最佳的修正量的，换句话也就是说最小的匹配误差不应
             * 该出现在滑动窗口滑动区间的两端.
            */
            if(bestincR==-L || bestincR==L)
                continue;
            /** </ul> */

            // Sub-pixel match (Parabola fitting)
            // 步骤2.3：做抛物线拟合找谷底得到亚像素匹配deltaR
            /** <li> 做抛物线拟合找谷底得到亚像素匹配deltaR </li> 
             * (bestincR,dist) (bestincR-1,dist) (bestincR+1,dist)三个点拟合出抛物线
             * bestincR+deltaR就是抛物线谷底的位置，相对SAD匹配出的最小值bestincR的修正量为deltaR
             * deltaR是做抛物线拟合的时候，在bestincR的基础上做出的修正量
             * <ul>
             */ 
            
			//获取这三个点所对应的SAD距离
            const float dist1 = vDists[L+bestincR-1];	
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            /** <li> 这里可以设抛物线是y=ax^2+bx+c，将三个点代入，求deltaR=-b/2a+x2即可得出亚像素修正量 deltaR </li> 
             * 程序中的计算公式为:
             * \n \f$ \Delta R=\frac{d_1-d_3}{2(d_1+d_3-2d_2)}  \f$
             * \n 公式的相关推导可以看图:
             * <img src="../imgs/2.png"/>
             * <img src="../imgs/3.png"/>
             * <img src="../imgs/4.png"/>
            */
            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            /** <li> 然后判断:抛物线拟合得到的修正量不能超过一个像素，否则放弃求该特征点的深度 </li> 
             * 的确是，如果修正量超过了一个像素的话，其实也就是说明当前找到的bestincR根本就不对
             */  
            if(deltaR<-1 || deltaR>1)
                continue;
            /** </ul> */

            
            
            // Re-scaled coordinate
            /** <li> 接下来要准备计算真正的最佳匹配点的位置 bestuR </li> 
             * 通过描述子匹配得到匹配点位置为scaleduR0,通过SAD匹配找到修正量bestincR,通过抛物线拟合找到亚像素修正量deltaR加和.
            */
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);
            
            // 这里是disparity（视差），根据它算出depth，这也就是为什么非要这样精确地得到bestuR的原因
            /** <li> 计算视差 disparity = (uL-bestuR) ,并且判断视差是否在规定的范围内. 如果不在说明算错了,放弃这个特征点的深度求取 </li> */
            float disparity = (uL-bestuR);
			// 最后判断视差是否在范围内
            if(disparity>=minD && disparity<maxD) 
            {
                /** <li> 如果在的话，判断视差是否为负 </li> 
                 * 若视差为负,则说明视差本身非常低小以至于计算机在进行数值计算时出现了误差
                 * \n 那么我们就直接设置视差为一个非常小的正数,这里取0.01
                */
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // depth 是在这里计算的
                /** <li> 计算深度,并保存该特征点经过SAD匹配所得到的最小匹配偏差,匹配点的右坐标和深度等数据 </li> 
                 * \n depth=baseline*fx/disparity
                 */
                mvDepth[iL]=mbf/disparity;   // 深度
                mvuRight[iL] = bestuR;       // 匹配对在右图的横坐标
                vDistIdx.push_back(pair<int,int>(bestDist,iL)); // 该特征点SAD匹配最小匹配偏差
            }// 最后判断视差是否在范围内
        }//如果刚才匹配过程中的最佳描述子距离小于给定的阈值
        
    }//遍历左图中的特征点 

    /** </ul> */

    // 步骤3：剔除SAD匹配偏差较大的匹配特征点
    /** <li> <b>关键步骤三</b> 剔除SAD匹配偏差较大的匹配特征点  </li> 
     * \n 前面SAD匹配只判断滑动窗口中是否有局部最小值，这里通过对比剔除SAD匹配偏差比较大的特征点的深度
     * <ul> */
    /** <li> 根据所有匹配对的SAD偏差进行排序, 描述子距离由小到大 ,并且找到距离的中位数 median 并且计算距离阈值:</li>
     * \n thDist = 1.5f*1.4f*median 
     * \n 但是为什么选择 1.5 1.4 这两个数,不太明白
     */
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
	//TODO 这里的数为什么是1.5 和 1.4？自己定的吗
    const float thDist = 1.5f*1.4f*median; // 计算自适应距离, 大于此距离的匹配对将剔除

    //然后就是遍历成功进行SAD过程的左图特征点的“SAD匹配最小匹配偏差”了
    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        /** <li> 遍历所有的成功进行SAD匹配的特征点,判断这对特征点的描述子距离之差是否符合要求. </li> 
         * \n 只保留符合这个阈值的点.
         * \n 其实这个过程也可以看成是保留描述子距离较小的匹配点.那些比较大的就扔掉,这样的一个过程
        */
        if(vDistIdx[i].first<thDist)
			//符合，过
            break;
        else
        {
			//不符合，设置-1表示这个点被淘汰、不可用
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }//判断是否符合要求
    }//遍历成功进行SAD过程的左图特征点的“SAD匹配最小匹配偏差”了
    /** </ul> */
    /** <li> 最终得到了比较靠谱的左右目的特征点匹配关系. </li> */
    /** </ul> */
}

//计算RGBD图像的立体深度信息
void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)	//参数是深度图像
{
    /** 主要步骤如下:.对于彩色图像中的每一个特征点:<ul>  */
    // mvDepth直接由depth图像读取`
	//这里是初始化这两个存储“右图”匹配特征点横坐标和存储特征点深度值的vector
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

	//开始遍历彩色图像中的所有特征点
    for(int i=0; i<N; i++)
    {
        /** <li> 从<b>未矫正的特征点</b>提供的坐标来读取深度图像拿到这个点的深度数据 </li> */
		//获取校正前和校正后的特征点
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

		//获取其横纵坐标，注意 NOTICE 是校正前的特征点的
        const float &v = kp.pt.y;
        const float &u = kp.pt.x;
		//从深度图像中获取这个特征点对应的深度点
        //NOTE 从这里看对深度图像进行去畸变处理是没有必要的,我们依旧可以直接通过未矫正的特征点的坐标来直接拿到深度数据
        const float d = imDepth.at<float>(v,u);

		//
        /** <li> 如果获取到的深度点合法(d>0), 那么就保存这个特征点的深度,并且计算出等效的\在假想的右图中该特征点所匹配的特征点的横坐标 </li>
         * \n 这个横坐标的计算是 x-mbf/d
         * \n 其中的x使用的是<b>矫正后的</b>特征点的图像坐标
         */
        if(d>0)
        {
			//那么就保存这个点的深度
            mvDepth[i] = d;
			//根据这个点的深度计算出等效的、在假想的右图中的该特征点的横坐标
			//TODO 话说为什么要计算这个嘞，计算出来之后有什么用?可能是为了保持计算一致
            mvuRight[i] = kpU.pt.x-mbf/d;
        }//如果获取到的深度点合法
    }//开始遍历彩色图像中的所有特征点
    /** </ul> */
}

//当某个特征点的深度信息或者双目信息有效时,将它反投影到三维世界坐标系中
cv::Mat Frame::UnprojectStereo(const int &i)
{
    // KeyFrame::UnprojectStereo 
	// 貌似这里普通帧的反投影函数操作过程和关键帧的反投影函数操作过程有一些不同：
    // mvDepth是在ComputeStereoMatches函数中求取的
	// TODO 验证下面的这些内容. 虽然现在我感觉是理解错了,但是不确定;如果确定是真的理解错了,那么就删除下面的内容
    // mvDepth对应的校正前的特征点，可这里却是对校正后特征点反投影
    // KeyFrame::UnprojectStereo中是对校正前的特征点mvKeys反投影
    // 在ComputeStereoMatches函数中应该对校正后的特征点求深度？？ (wubo???)
	// NOTE 不过我记得好像上面的ComputeStereoMatches函数就是对于双目相机设计的，而双目相机的图像默认都是经过了校正的啊

    /** 步骤如下: <ul> */

    /** <li> 获取这个特征点的深度（这里的深度可能是通过双目视差得出的，也可能是直接通过深度图像的出来的） </li> */
	const float z = mvDepth[i];
    /** <li> 判断这个深度是否合法 </li> <ul> */
	//（其实这里也可以不再进行判断，因为在计算或者生成这个深度的时候都是经过检查了的_不行,RGBD的不是）
    if(z>0)
    {
        /** <li> 如果合法,就利用<b></b>矫正后的特征点的坐标 Frame::mvKeysUn 和相机的内参数,通过反投影和位姿变换得到空间点的坐标 </li> */
		//获取像素坐标，注意这里是矫正后的特征点的坐标
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
		//计算在当前相机坐标系下的坐标
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
		//生成三维点（在当前相机坐标系下）
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
		//然后计算这个点在世界坐标系下的坐标，这里是对的，但是公式还是要斟酌一下。首先变换成在没有旋转的相机坐标系下，最后考虑相机坐标系相对于世界坐标系的平移
        return mRwc*x3Dc+mOw;
    }
    else
        /** <li> 如果深度值不合法，那么就返回一个空矩阵,表示计算失败 </li> */
        return cv::Mat();
    /** </ul> */
    /** </ul> */
}

} //namespace ORB_SLAM
