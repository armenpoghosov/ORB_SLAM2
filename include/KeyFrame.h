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

    cv::Mat GetPose() const
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Tcw.clone();
    }

    cv::Mat GetPoseInverse() const
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Twc.clone();
    }

    cv::Mat GetCameraCenter() const
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Twc.rowRange(0, 3).col(3).clone();
    }

    cv::Mat GetStereoCenter() const
    {
        cv::Mat center = (cv::Mat_<float>(4, 1) << (mb / 2.f), 0, 0, 1);
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Twc * center;
    }

    cv::Mat GetRotation() const
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Tcw.rowRange(0, 3).colRange(0, 3).clone();
    }

    cv::Mat GetTranslation() const
    {
        std::unique_lock<std::mutex> lock(mMutexPose);
        return m_Tcw.rowRange(0, 3).col(3).clone();
    }

    // Covisibility graph functions
    void AddConnection(KeyFrame* pKF, std::size_t points);

    void EraseConnection(KeyFrame* pKF);
    void UpdateConnections();
    void UpdateBestCovisibles();

    std::unordered_set<KeyFrame*> GetConnectedKeyFrames() const;
    std::vector<KeyFrame*> GetBestCovisibilityKeyFrames(std::size_t N) const;
    std::vector<KeyFrame*> GetCovisiblesByWeight(int w) const;

    std::vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames() const
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    std::size_t GetWeight(KeyFrame* pKF) const
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        auto it = mConnectedKeyFrameWeights.find(pKF);
        return it != mConnectedKeyFrameWeights.end() ? it->second : 0;
    }

    // Spanning tree functions
    void AddChild(KeyFrame* pKF)
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        m_children.insert(pKF);
    }

    void EraseChild(KeyFrame* pKF)
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        m_children.erase(pKF);
    }

    void ChangeParent(KeyFrame* pKF);

    std::unordered_set<KeyFrame*> GetChilds() const
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        return m_children;
    }

    KeyFrame* GetParent() const
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    bool hasChild(KeyFrame* pKF) const
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        return m_children.count(pKF) != 0;
    }

    // Loop Edges
    void AddLoopEdge(KeyFrame* pKF);

    std::unordered_set<KeyFrame*> GetLoopEdges() const
    {
        std::unique_lock<std::mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    // MapPoint observation functions
    void AddMapPoint(MapPoint* pMP, size_t idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = pMP;
    }

    void EraseMapPointMatch(size_t idx)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mvpMapPoints[idx] = nullptr;
    }

    void EraseMapPointMatch(MapPoint* pMP);

    void ReplaceMapPointMatch(size_t idx, MapPoint* pMP)
        { mvpMapPoints[idx] = pMP; }

    std::unordered_set<MapPoint*> GetMapPoints() const;

    std::vector<MapPoint*> GetMapPointMatches() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    MapPoint* GetMapPoint(std::size_t idx) const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    std::size_t TrackedMapPoints(std::size_t minObs) const;

    // KeyPoint functions
    std::vector<size_t> GetFeaturesInArea(float x, float y, float r) const;
    cv::Mat UnprojectStereo(std::size_t index);

    // Image
    bool IsInImage(float x, float y) const
        { return x >= mnMinX && x < mnMaxX && y >= mnMinY && y < mnMaxY; }

    // Enable/Disable bad flag changes
    void SetErase();

    void SetNotErase()
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        mbNotErase = true;
    }

    // Set/check bad flag
    void SetBadFlag();

    bool isBad() const
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);
        return mbBad;
    }

    // Compute Scene Depth (q=2 median). Used in monocular.
    float ComputeSceneMedianDepth(int q) const;

    static bool lId(KeyFrame const* pKF1, KeyFrame const* pKF2)
        { return pKF1->m_id < pKF2->m_id; }

    static void reset_id_counter()
        { s_next_id = 0; }

    uint64_t get_id() const
        { return m_id; }

    uint64_t get_frame_id() const
        { return mnFrameId; }

    DBoW2::BowVector const& get_BoW() const
    {
        const_cast<KeyFrame*>(this)->ensure_BoW_computed();
        return mBowVec;
    }

    DBoW2::FeatureVector const& get_BoW_features() const
    {
        const_cast<KeyFrame*>(this)->ensure_BoW_computed();
        return mFeatVec;
    }

    cv::Mat const& get_Tcp() const
        { return mTcp; }

    KeyFrame(KeyFrame const&) = delete;
    KeyFrame& operator = (KeyFrame const&) = delete;

    // The following variables are accesed from only 1 thread or never change (no mutex needed).
public:

    double const                    mTimeStamp;

    // Grid (to speed up feature matching)
    int const                       mnGridCols;
    int const                       mnGridRows;
    float const                     mfGridElementWidthInv;
    float const                     mfGridElementHeightInv;

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
    std::size_t const               m_kf_N;

    // KeyPoints, stereo coordinate and descriptors (all associated by an index)
    std::vector<cv::KeyPoint> const mvKeys;
    std::vector<cv::KeyPoint> const mvKeysUn;
    
    std::vector<float> const        mvuRight;   // negative value for monocular points
    std::vector<float> const        mvDepth;    // negative value for monocular points

    cv::Mat const                   mDescriptors;

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

protected:

    // Compute Bag of Words representation.
    void ensure_BoW_computed()
    {
        if (mBowVec.empty())
            compute_BoW();
    }

    void compute_BoW();

    static uint64_t                 s_next_id;
    uint64_t                        m_id;
    uint64_t const                  mnFrameId;

    // Pose relative to parent (this is computed when bad flag is activated)
    cv::Mat                         mTcp;

    // The following variables need to be accessed trough a mutex to be thread safe.

    // SE3 pose and inverse
    cv::Mat                         m_Tcw;
    cv::Mat                         m_Twc;

    // BoW
    DBoW2::BowVector                mBowVec;
    DBoW2::FeatureVector            mFeatVec;

    // MapPoints associated to keypoints
    std::vector<MapPoint*>          mvpMapPoints;

    // BoW
    KeyFrameDatabase*               mpKeyFrameDB;
    ORBVocabulary*                  mpORBvocabulary;

    // Grid over the image to speed up feature matching
    std::vector<std::vector<std::vector<size_t> > >
                                    mGrid;

    std::unordered_map<KeyFrame*, std::size_t> 
                                    mConnectedKeyFrameWeights;

    std::vector<KeyFrame*>          mvpOrderedConnectedKeyFrames;
    std::vector<std::size_t>        mvOrderedWeights;

    // Spanning Tree and Loop Edges
    KeyFrame*                       mpParent;
    std::unordered_set<KeyFrame*>   m_children;
    std::unordered_set<KeyFrame*>   mspLoopEdges;
    bool                            mbFirstConnection;

    // Bad flags
    bool                            mbNotErase;
    bool                            mbToBeErased;
    bool                            mbBad;

    Map*                            mpMap;

    std::mutex mutable              mMutexPose;
    std::mutex mutable              mMutexConnections;
    std::mutex mutable              mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // KEYFRAME_H
