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

#ifndef MAPPOINT_H
#define MAPPOINT_H

#include <opencv2/core/core.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace ORB_SLAM2
{

class KeyFrame;
class Map;
class Frame;

class MapPoint
{
public:

    MapPoint(cv::Mat const& Pos, KeyFrame* pRefKF, Map* pMap);
    MapPoint(cv::Mat const& Pos, Map* pMap, Frame* pFrame, int idxF);

    void SetWorldPos(const cv::Mat &Pos);

    cv::Mat GetWorldPos() const
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mWorldPos.clone();
    }

    cv::Mat GetNormal() const
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return mNormalVector.clone();
    }

    KeyFrame* GetReferenceKeyFrame() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mpRefKF;
    }

    std::unordered_map<KeyFrame*, size_t> GetObservations() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations;
    }

    std::size_t Observations() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return m_observe_count;
    }

    void AddObservation(KeyFrame* pKF,size_t idx);
    void EraseObservation(KeyFrame* pKF);

    std::size_t GetIndexInKeyFrame(KeyFrame *pKF) const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        auto it = mObservations.find(pKF);
        return it != mObservations.end() ? it->second : (std::size_t) - 1;
    }

    bool IsInKeyFrame(KeyFrame *pKF) const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mObservations.count(pKF) != 0;
    }

    void SetBadFlag();

    bool isBad() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mbBad;
    }

    void Replace(MapPoint* pMP);

    MapPoint* GetReplaced() const
    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        return mpReplaced;
    }

    void IncreaseVisible(int n = 1)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnVisible += n;
    }

    void IncreaseFound(int n = 1)
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mnFound += n;
    }

    float GetFoundRatio() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return (float)mnFound / mnVisible;
    }

    int GetFound() const
        { return mnFound; }

    void ComputeDistinctiveDescriptors();

    cv::Mat GetDescriptor() const
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        return mDescriptor.clone();
    }

    void UpdateNormalAndDepth();

    float GetMinDistanceInvariance() const
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 0.8f * mfMinDistance;
    }

    float GetMaxDistanceInvariance() const
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        return 1.2f * mfMaxDistance;
    }

    int PredictScale(float currentDist, KeyFrame* pKF) const;
    int PredictScale(float currentDist, Frame const* pF) const;

    uint64_t get_id() const
        { return mnId; }

public:

    static std::atomic<uint64_t>    s_next_id;

    uint64_t                        mnFirstKFid;
    uint64_t                        mnFirstFrame;

    std::size_t                     m_observe_count;

    // Variables used by the tracking
    float                           mTrackProjX;
    float                           mTrackProjY;
    float                           mTrackProjXR;
    bool                            mbTrackInView;
    int                             mnTrackScaleLevel;
    float                           mTrackViewCos;
    uint64_t                        mnTrackReferenceForFrame;
    uint64_t                        mnLastFrameSeen;

    // Variables used by local mapping
    uint64_t                        mnBALocalForKF;
    uint64_t                        mnFuseCandidateForKF;

    // Variables used by loop closing
    uint64_t                        mnLoopPointForKF;
    uint64_t                        mnCorrectedByKF;
    uint64_t                        mnCorrectedReference;
    cv::Mat                         mPosGBA;

    uint64_t                        mnBAGlobalForKF;

    static std::mutex               mGlobalMutex;

protected:

    uint64_t                        mnId;

     // Position in absolute coordinates
     cv::Mat                        mWorldPos;

     // Keyframes observing the point and associated index in keyframe
     std::unordered_map<KeyFrame*, size_t>
                                    mObservations;

     // Mean viewing direction
     cv::Mat                        mNormalVector;

     // Best descriptor to fast matching
     cv::Mat                        mDescriptor;

     // Reference KeyFrame
     KeyFrame*                      mpRefKF;

     // Tracking counters
     int                            mnVisible;
     
     
     int                            mnFound;

     // Bad flag (we do not currently erase MapPoint from memory)
     bool                           mbBad;
     MapPoint*                      mpReplaced;

     // Scale invariance distances
     float                          mfMinDistance;
     float                          mfMaxDistance;

     Map*                           mpMap;

     std::mutex mutable             mMutexPos;
     std::mutex mutable             mMutexFeatures;
};

} //namespace ORB_SLAM

#endif // MAPPOINT_H
