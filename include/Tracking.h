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


#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include"Frame.h"

#include <mutex>
#include <future>
#include <unordered_set>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LoopClosing;
class System;
class MapDrawer;
class KeyFrameDatabase;
class Initializer;

class Tracking
{  
public:

    // Tracking states
    enum eTrackingState
    {
        SYSTEM_NOT_READY    = -1,
        NO_IMAGES_YET       = 0,
        NOT_INITIALIZED     = 1,
        OK                  = 2,
        LOST                = 3
    };

    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
        KeyFrameDatabase* pKFDB, const string &strSettingPath, int sensor);

    ~Tracking();

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    void GrabImageStereo(cv::Mat const& imRectLeft, cv::Mat const& imRectRight, double timestamp);
    void GrabImageRGBD(cv::Mat const& imRGB, cv::Mat const& imD, double timestamp);
    void GrabImageMonocular(cv::Mat const& im, double timestamp);

    void SetLoopClosing(LoopClosing* pLoopClosing)
        { mpLoopClosing = pLoopClosing; }

    void SetViewer(Viewer* pViewer)
        { mpViewer = pViewer; }

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(string const& strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void set_only_tracking(bool flag)
        { mbOnlyTracking = flag; }

    eTrackingState get_last_state() const
        { return mLastProcessedState; }

    Frame const& get_current_frame() const
        { return mCurrentFrame; }

    bool is_only_tracking() const
        { return mbOnlyTracking; }

    std::vector<int> const& get_ini_matches() const
        { return mvIniMatches; }

    Frame const& get_initial_frame() const
        { return mInitialFrame; }

    std::list<cv::Mat> const& get_frame_relative_poses() const
        { return mlRelativeFramePoses; }

    std::list<KeyFrame*> const& get_frame_reference_key_frames() const
        { return mlpReferences; }

    std::list<double> const& get_frame_times() const
        { return mlFrameTimes; }

    std::list<bool> const& get_frame_lost_flags() const
        { return mlbLost; }

    void Reset();

protected:

    // TODO: PAE: added to make extraction work faster
    void track_worker(std::unique_ptr<Frame> frame);

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    bool TrackReferenceKeyFrame();
    void UpdateLastFrame();
    bool TrackWithMotionModel();

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    // PAE: functions moved from local mapping
    void ProcessNewKeyFrame();
    void CreateNewMapPoints();
    void MapPointCulling();
    void SearchInNeighbors();
    void KeyFrameCulling();
    static cv::Mat ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2);
    static cv::Mat SkewSymmetricMatrix(cv::Mat const& v);
    void enqueue_key_frame(KeyFrame* pKF);
    // PAE: members moved from local mapping
    KeyFrame*                   mpCurrentKeyFrame;
    std::list<MapPoint*>        mlpRecentAddedMapPoints;


    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    std::list<cv::Mat>          mlRelativeFramePoses;
    std::list<KeyFrame*>        mlpReferences;
    std::list<double>           mlFrameTimes;
    std::list<bool>             mlbLost;

    // Initialization Variables (Monocular)
    std::vector<int>            mvIniMatches;
    std::vector<cv::Point2f>    mvbPrevMatched;
    std::vector<cv::Point3f>    mvIniP3D;
    Frame                       mInitialFrame;

    // Current Frame
    Frame                       mCurrentFrame;

    // True if local mapping is deactivated and we are performing only localization
    bool                        mbOnlyTracking;

    // Input sensor
    int                         mSensor;

    // current and last processed state
    eTrackingState              mState;
    eTrackingState              mLastProcessedState;

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool                        mbVO;

    // Other Thread Pointers
    LoopClosing*                mpLoopClosing;

    // ORB
    ORBextractor*               mpORBextractorLeft;
    ORBextractor*               mpORBextractorRight;
    ORBextractor*               mpIniORBextractor;

    // BoW
    ORBVocabulary*              mpORBVocabulary;
    KeyFrameDatabase*           mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer*                mpInitializer;

    // Local Map
    KeyFrame*                       mpReferenceKF;
    std::unordered_set<KeyFrame*>   mvpLocalKeyFrames;
    std::unordered_set<MapPoint*>   mvpLocalMapPoints;

    // System
    System*                     mpSystem;
    
    // Drawers
    Viewer*                     mpViewer;
    FrameDrawer*                mpFrameDrawer;
    MapDrawer*                  mpMapDrawer;

    // Map
    Map*                        mpMap;

    //Calibration matrix
    cv::Mat                     mK;
    cv::Mat                     mDistCoef;
    float                       mbf;

    //New KeyFrame rules (according to fps)
    int                         mMinFrames;
    int                         mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float                       mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float                       mDepthMapFactor;

    // Current matches in frame
    int                         mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame*                   mpLastKeyFrame;
    Frame                       mLastFrame;

    uint64_t                    mnLastKeyFrameId;
    uint64_t                    mnLastRelocFrameId;

    // Motion Model
    cv::Mat                     mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool                        mbRGB;

    std::list<MapPoint*>        mlpTemporalPoints;

    // TODO: PAE: need to syncronoze it
    std::future<void>           m_future;


    std::ofstream               m_ofs;

};

} //namespace ORB_SLAM

#endif // TRACKING_H
