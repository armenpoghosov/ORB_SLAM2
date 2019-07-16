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

class LocalMapping
{
public:

    LocalMapping(Map* pMap, bool bMonocular);
    ~LocalMapping();

    void SetLoopCloser(LoopClosing* pLoopCloser)
        { mpLoopCloser = pLoopCloser; }

    void SetTracker(Tracking *pTracker)
        { mpTracker = pTracker; }

    void InsertKeyFrame(KeyFrame* pKF);

    // Thread Synch
    void RequestStop();

    void RequestReset();

    bool Stop();

    void Release();

    bool isStopped();

    bool stopRequested();

    bool AcceptKeyFrames()
    {
        unique_lock<mutex> lock(mMutexAccept);
        return mbAcceptKeyFrames;
    }

    bool SetNotStop(bool flag)
    {
        unique_lock<mutex> lock(mMutexStop);

        if (flag && mbStopped)
            return false;

        mbNotStop = flag;

        return true;
    }

    void InterruptBA()
    {
        mbAbortBA = true;
    }

    void RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    std::size_t KeyframesInQueue()
    {
        unique_lock<std::mutex> lock(mMutexNewKFs);
        return mlNewKeyFrames.size();
    }

protected:

    // Main function
    void Run();

    void SetAcceptKeyFrames(bool flag)
    {
        unique_lock<mutex> lock(mMutexAccept);
        mbAcceptKeyFrames = flag;
    }

    bool CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexNewKFs);
        return !mlNewKeyFrames.empty();
    }

    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    cv::Mat ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2);

    cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    void ResetIfRequested();
    bool CheckFinish();
    void SetFinish();

    bool                    mbResetRequested;
    bool                    mbMonocular;
    bool                    mbFinishRequested;
    bool                    mbFinished;
    std::mutex              mMutexReset;
    std::mutex              mMutexFinish;
    Map*                    mpMap;
    LoopClosing*            mpLoopCloser;
    Tracking*               mpTracker;
    std::list<KeyFrame*>    mlNewKeyFrames;
    KeyFrame*               mpCurrentKeyFrame;
    std::list<MapPoint*>    mlpRecentAddedMapPoints;
    std::mutex              mMutexNewKFs;
    bool                    mbAbortBA;
    bool                    mbStopped;
    bool                    mbStopRequested;
    bool                    mbNotStop;
    std::mutex              mMutexStop;
    bool                    mbAcceptKeyFrames;
    std::mutex              mMutexAccept;
    std::thread             m_thread;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
