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

#include <opencv2/core/core.hpp>

#include <atomic> // TODO: remove?
#include <condition_variable>
#include <list>
#include <mutex>
#include <thread>

namespace ORB_SLAM2
{

class Tracking;
class LoopClosing;
class Map;
class KeyFrame;
class MapPoint;

// -------------------------------------------------------------------------------------------------
//
//
//

class LocalMapping
{
public:

    // ---------------------------------------------------------------------------------------------
    //
    //
    //

    LocalMapping(Map* pMap, bool bMonocular);
    ~LocalMapping();

    // ---------------------------------------------------------------------------------------------
    //
    //
    //

    void SetLoopCloser(LoopClosing* pLoopCloser)
        { mpLoopCloser = pLoopCloser; }

    void SetTracker(Tracking *pTracker)
        { mpTracker = pTracker; }

    // ---------------------------------------------------------------------------------------------
    //
    //
    //

    bool pause()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        mbStopRequested = true;
        mbAbortBA = true;
        return true;
    }

    bool is_paused() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_state == eState_Paused;
    }

    bool is_pause_requested() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_state == eState_Pausing;
    }




    void RequestReset();

    void Release();

    bool SetNotStop(bool flag)
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        if (flag && m_state == eState_Paused)
            return false;

        mbNotStop = flag;

        return true;
    }

    void InterruptBA()
    {
        // TODO: think about it!
        mbAbortBA = true;
    }

    void RequestFinish()
    {
        // TODO: think about it
        {
            std::unique_lock<std::mutex> lock(m_mutex);
            m_state = eState_Stopping;
        }

        m_kick_condition.notify_all();
    }

    bool isFinished()
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_state == eState_Stopped;
    }

    // ---------------------------------------------------------------------------------------------
    //
    //
    //

    bool AcceptKeyFrames() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_state == eState_Idle;
    }

    std::size_t queued_key_frames() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return mlNewKeyFrames.size();
    }

    void enqueue_key_frame(KeyFrame* pKF);

protected:

    // Main function
    void run();

    bool Stop();

    void ProcessNewKeyFrame();
    void CreateNewMapPoints();

    void MapPointCulling();
    void SearchInNeighbors();

    void KeyFrameCulling();

    static cv::Mat ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2);

    static cv::Mat SkewSymmetricMatrix(const cv::Mat &v);

    void ResetIfRequested();

    bool CheckFinish() const
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_state == eState_Stopping;
    }

    enum eStates
    {
        eState_Stopped,
        eState_Idle,
        eState_Running,
        eState_Stopping,
        eState_Pausing,
        eState_Paused
    };

    bool                    mbResetRequested;
    bool                    mbMonocular;
    Map*                    mpMap;
    LoopClosing*            mpLoopCloser;
    Tracking*               mpTracker;
    KeyFrame*               mpCurrentKeyFrame;
    std::list<KeyFrame*>    mlNewKeyFrames;
    std::list<MapPoint*>    mlpRecentAddedMapPoints;
    bool                    mbAbortBA;
    bool                    mbStopped;
    bool                    mbStopRequested;
    bool                    mbNotStop;


    eStates                 m_state;
    std::mutex mutable      m_mutex;
    std::condition_variable m_kick_condition;
    std::thread             m_thread;
};

} //namespace ORB_SLAM

#endif // LOCALMAPPING_H
