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


#ifndef SYSTEM_H
#define SYSTEM_H

#include <opencv2/core/core.hpp>

#include "ORBVocabulary.h"

#include <string>
#include <mutex>

namespace ORB_SLAM2
{

class Tracking;
class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LoopClosing;
class KeyFrameDatabase;
class MapDrawer;

class System
{
public:

    // Input sensor
    enum eSensor
    {
        MONOCULAR   = 0,
        STEREO      = 1,
        RGBD        = 2
    };

public:

    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(std::string const& strVocFile, std::string const& strSettingsFile,
        eSensor sensor, bool bUseViewer = true);

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    void TrackStereo(cv::Mat const& imLeft, cv::Mat const& imRight, double timestamp);

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    void TrackRGBD(cv::Mat const& im, cv::Mat const& depthmap, double timestamp);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    void TrackMonocular(cv::Mat const& im, double timestamp);

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode()
    {
        std::unique_lock<std::mutex> lock(mMutexMode);
        mbActivateLocalizationMode = true;
    }

    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode()
    {
        std::unique_lock<std::mutex> lock(mMutexMode);
        mbDeactivateLocalizationMode = true;
    }

    // Reset the system (clear map)
    void Reset()
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        mbReset = true;
    }

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(std::string const& filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(std::string const& filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(std::string const& filename);

private:

    void locked_handle_activate_localization();
    void locked_handle_deactivate_localization();
    void locked_handle_reset();

    // Input sensor
    eSensor                     mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary*              mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase*           mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map*                        mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking*                   mpTracker;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing*                mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer*                     mpViewer;

    FrameDrawer*                mpFrameDrawer;
    MapDrawer*                  mpMapDrawer;

    // The Tracking thread "lives" in the main execution thread that creates the System object.

    // Reset flag
    std::mutex                  mMutexReset;
    bool                        mbReset;

    // Change mode flags
    std::mutex                  mMutexMode;
    bool                        mbActivateLocalizationMode;
    bool                        mbDeactivateLocalizationMode;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
