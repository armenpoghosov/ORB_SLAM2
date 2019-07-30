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

#ifndef KEYFRAME_H
#define KEYFRAME_H

#include "MapPoint.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"
#include "ORBVocabulary.h"
#include "ORBextractor.h"
#include "Frame.h"
#include "KeyFrameDatabase.h"

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace ORB_SLAM2
{

class Map;
class MapPoint;
class Frame;
class KeyFrameDatabase;

class KeyFrame
{
public:

    KeyFrame(Frame& F, Map* pMap, KeyFrameDatabase* pKFDB);

    // Pose functions
    void SetPose(cv::Mat const& Tcw);

    cv::Mat GetPose()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.clone();
    }

    cv::Mat GetPoseInverse()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Twc.clone();
    }

    cv::Mat GetCameraCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Ow.clone();
    }

    cv::Mat GetStereoCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Cw.clone();
    }

    cv::Mat GetRotation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).colRange(0, 3).clone();
    }

    cv::Mat GetTranslation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0, 3).col(3).clone();
    }

    // Bag of Words Representation
    void ComputeBoW();

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, int weight);

    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();
    std::set<KeyFrame*> GetConnectedKeyFrames();
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(int N);
    std::vector<KeyFrame*> GetCovisiblesByWeight(int w);

    std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    int GetWeight(KeyFrame* pKF)
    {
        unique_lock<mutex> lock(mMutexConnections);
        auto it = mConnectedKeyFrameWeights.find(pKF);
        return it != mConnectedKeyFrameWeights.end() ? it->second : 0;
    }

    // Spanning tree functions
    void AddChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    void EraseChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.erase(pKF);
    }

    void ChangeParent(KeyFrame* pKF);

    std::unordered_set<KeyFrame*> GetChilds()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens;
    }

    KeyFrame* GetParent()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    bool hasChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens.count(pKF);
    }

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);

    std::unordered_set<KeyFrame*> GetLoopEdges()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    void EraseMapPointMatch(size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }

    void EraseMapPointMatch(MapPoint* pMP);

    void ReplaceMapPointMatch(size_t idx, MapPoint* pMP)
        { mvpMapPoints[idx] = pMP; }

    std::set<MapPoint*> GetMapPoints();

    std::vector<MapPoint*> GetMapPointMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    MapPoint* GetMapPoint(std::size_t idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    int TrackedMapPoints(int minObs);

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(const float &x, const float  &y, const float  &r) const;
    cv::Mat UnprojectStereo(std::size_t index);

    // Image
    bool IsInImage(float x, float y) const
    {
        return x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY;
    }

    // Enable/Disable bad flag changes
    void SetErase();

    void SetNotErase()
    {
        unique_lock<mutex> lock(mMutexConnections);
        mbNotErase = true;
    }

    // Set/check bad flag
    void SetBadFlag();

    bool isBad()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mbBad;
    }

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(int q);

    static bool weightComp(int a, int b)
        { return a > b; }

    static bool lId(KeyFrame* pKF1, KeyFrame* pKF2)
        { return pKF1->mnId < pKF2->mnId; }

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    static uint64_t         s_next_id;

    uint64_t                mnId;
    uint64_t const          mnFrameId;

    double const            mTimeStamp;

    // Grid (to speed up feature matching)
    int const               mnGridCols;
    int const               mnGridRows;
    float const             mfGridElementWidthInv;
    float const             mfGridElementHeightInv;

    // Variables used by the tracking
    uint64_t                mnTrackReferenceForFrame;
    uint64_t                mnFuseTargetForKF;

    // Variables used by the local mapping
    long unsigned int       mnBALocalForKF;
    long unsigned int       mnBAFixedForKF;

    // Variables used by the keyframe database
    long unsigned int       mnLoopQuery;
    int                     mnLoopWords;
    float                   mLoopScore;
    long unsigned int       mnRelocQuery;
    int                     mnRelocWords;
    float                   mRelocScore;

    // Variables used by loop closing
    cv::Mat                 mTcwGBA;
    cv::Mat                 mTcwBefGBA;
    long unsigned int       mnBAGlobalForKF;

    // Calibration parameters
    float const                     fx;
    float const                     fy;
    float const                     cx;
    float const                     cy;
    float const                     invfx;
    float const                     invfy;
    float const                     mbf;
    float const                     mb;
    float const                     mThDepth;

    // Number of KeyPoints
    int const                       N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    std::vector<cv::KeyPoint> const mvKeys;
    std::vector<cv::KeyPoint> const mvKeysUn;
    
    std::vector<float> const        mvuRight;   // negative value for monocular points
    std::vector<float> const        mvDepth;    // negative value for monocular points

    cv::Mat const                   mDescriptors;

    // BoW
    DBoW2::BowVector                mBowVec;
    DBoW2::FeatureVector            mFeatVec;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat mTcp;

    // Scale
    int const                       mnScaleLevels;
    float const                     mfScaleFactor;
    float const                     mfLogScaleFactor;
    std::vector<float> const        mvScaleFactors;
    std::vector<float> const        mvLevelSigma2;
    std::vector<float> const        mvInvLevelSigma2;

    // Image bounds and calibration
    int const                       mnMinX;
    int const                       mnMinY;
    int const                       mnMaxX;
    int const                       mnMaxY;
    cv::Mat const                   mK;

    // The following variables need to be accessed trough a mutex to be thread safe.
protected:

    // SE3 Pose and camera center
    cv::Mat                     Tcw;
    cv::Mat                     Twc;
    cv::Mat                     Ow;

    cv::Mat                     Cw; // Stereo middel point. Only for visualization

    // MapPoints associated to keypoints
    std::vector<MapPoint*>      mvpMapPoints;

    // BoW
    KeyFrameDatabase*           mpKeyFrameDB;
    ORBVocabulary*              mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector< std::vector <std::vector<size_t> > >   mGrid;

    std::unordered_map<KeyFrame*, int>    mConnectedKeyFrameWeights;

    std::vector<KeyFrame*>          mvpOrderedConnectedKeyFrames;
    std::vector<int>                mvOrderedWeights;

    // Spanning Tree and Loop Edges
    bool                            mbFirstConnection;
    KeyFrame*                       mpParent;
    std::unordered_set<KeyFrame*>   mspChildrens;
    std::unordered_set<KeyFrame*>   mspLoopEdges;

    // Bad flags
    bool                            mbNotErase;
    bool                            mbToBeErased;
    bool                            mbBad;

    float                           mHalfBaseline; // Only for visualization

    Map*                            mpMap;

    std::mutex mMutexPose;
    std::mutex mMutexConnections;
    std::mutex mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
