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

#ifndef LOOPCLOSING_H
#define LOOPCLOSING_H

#include "ORBVocabulary.h"

#include "Thirdparty/g2o/g2o/types/sim3.h"

#include <mutex>
#include <thread>
#include <unordered_set>
#include <unordered_map>

namespace ORB_SLAM2
{

class Frame;
class KeyFrame;
class KeyFrameDatabase;
class Map;
class MapPoint;
class Tracking;

class LoopClosing
{
public:

    typedef std::pair<std::unordered_set<KeyFrame*>, int> ConsistentGroup;

    typedef std::unordered_map<KeyFrame*, g2o::Sim3, std::hash<KeyFrame*>, std::equal_to<KeyFrame*>,
        Eigen::aligned_allocator<std::pair<KeyFrame const*, g2o::Sim3> > > KeyFrameAndPose;

public:

    LoopClosing(Map* pMap, KeyFrameDatabase* pDB, ORBVocabulary* pVoc, bool bFixScale);
    ~LoopClosing();

    void SetTracker(Tracking *pTracker)
        { mpTracker = pTracker; }

    void InsertKeyFrame(KeyFrame *pKF);

    // This function will run in a separate thread
    void RunGlobalBundleAdjustment();

    void RequestReset();

    bool isRunningGBA()
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbRunningGBA;
    }

    bool isFinishedGBA()
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);
        return mbFinishedGBA;
    }   

    void RequestFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool isFinished()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinished;
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

    // Main function
    void Run();

    bool CheckNewKeyFrames() const
    {
        std::unique_lock<std::mutex> lock(mMutexLoopQueue);
        return !mlpLoopKeyFrameQueue.empty();
    }

    bool DetectLoop();
    bool ComputeSim3();
    void SearchAndFuse(KeyFrameAndPose const& CorrectedPosesMap);
    void CorrectLoop();

    void ResetIfRequested();

    bool CheckFinish() const
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void SetFinish()
    {
        std::unique_lock<std::mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool                            mbFinishRequested;
    bool                            mbFinished;
    bool                            mbResetRequested;

    std::mutex mutable              mMutexReset;
    std::mutex mutable              mMutexFinish;

    Map*                            mpMap;
    Tracking*                       mpTracker;

    KeyFrameDatabase*               mpKeyFrameDB;
    ORBVocabulary*                  mpORBVocabulary;

    std::list<KeyFrame*>            mlpLoopKeyFrameQueue;

    std::mutex mutable              mMutexLoopQueue;

    // Loop detector parameters
    float                           mnCovisibilityConsistencyTh;

    // Loop detector variables
    KeyFrame*                       mpCurrentKF;
    KeyFrame*                       mpMatchedKF;
    std::vector<ConsistentGroup>    mvConsistentGroups;
    std::vector<KeyFrame*>          mvpEnoughConsistentCandidates;
    std::vector<KeyFrame*>          mvpCurrentConnectedKFs;
    std::vector<MapPoint*>          mvpCurrentMatchedPoints;
    std::unordered_set<MapPoint*>   mvpLoopMapPoints;
    cv::Mat                         mScw;
    g2o::Sim3                       mg2oScw;

    uint64_t                        mLastLoopKFid;

    // Variables related to Global Bundle Adjustment
    bool                            mbRunningGBA;
    bool                            mbFinishedGBA;
    bool                            mbStopGBA;
    std::mutex                      mMutexGBA;
    std::thread*                    mpThreadGBA;

    // Fix scale in the stereo/RGB-D case
    bool                            mbFixScale;

    std::thread                     m_thread;

    uint64_t                        mnFullBAIdx;
};

} //namespace ORB_SLAM

#endif // LOOPCLOSING_H
