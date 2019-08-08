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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{

System::System(string const& strVocFile, string const& strSettingsFile, eSensor sensor, bool bUseViewer)
    :
    mSensor(sensor),
    mpViewer(nullptr),
    mbReset(false),
    mbActivateLocalizationMode(false),
    mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

    if (mSensor == MONOCULAR)
        cout << "Monocular" << endl;
    else if (mSensor == STEREO)
        cout << "Stereo" << endl;
    else if (mSensor == RGBD)
        cout << "RGB-D" << endl;

    // Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

    mpVocabulary = new ORBVocabulary();
    if (!mpVocabulary->loadFromTextFile(strVocFile))
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1);
    }
    cout << "Vocabulary loaded!" << endl << endl;

    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);

    //Initialize the Loop Closing thread and launch
    // NOTE: PAE: thread is opened here!
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);

    //Initialize the Viewer thread and launch
    if (bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer, mpMapDrawer, mpTracker, strSettingsFile);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

void System::TrackStereo(cv::Mat const& imLeft, cv::Mat const& imRight, double timestamp)
{
    assert(mSensor == STEREO);

    // Check mode change
    {
        std::unique_lock<std::mutex> lock(mMutexMode);
        locked_handle_activate_localization();
        locked_handle_deactivate_localization();
    }

    // Check reset
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        locked_handle_reset();
    }

    mpTracker->GrabImageStereo(imLeft,imRight,timestamp);
}

void System::TrackRGBD(cv::Mat const& im, cv::Mat const& depthmap, double timestamp)
{
    assert(mSensor == RGBD);

    // Check mode change
    {
        std::unique_lock<std::mutex> lock(mMutexMode);
        locked_handle_activate_localization();
        locked_handle_deactivate_localization();
    }

    // Check reset
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        locked_handle_reset();
    }

    mpTracker->GrabImageRGBD(im,depthmap,timestamp);
}

void System::TrackMonocular(cv::Mat const& im, double timestamp)
{
    assert(mSensor == MONOCULAR);

    // Check mode change
    {
        std::unique_lock<std::mutex> lock(mMutexMode);
        locked_handle_activate_localization();
        locked_handle_deactivate_localization();
    }

    // Check reset
    {
        std::unique_lock<std::mutex> lock(mMutexReset);
        locked_handle_reset();
    }

    mpTracker->GrabImageMonocular(im, timestamp);
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

    if (mpViewer != nullptr)
    {
        mpViewer->RequestFinish();

        while (!mpViewer->isFinished())
            std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    // Wait until all thread have effectively stopped
    while (!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    if (mpViewer != nullptr)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    assert(mSensor != MONOCULAR);

    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<KeyFrame*> const& frame_ref_key_frames = mpTracker->get_frame_reference_key_frames();
    auto lRit = frame_ref_key_frames.begin();

    std::list<double> const& frame_times = mpTracker->get_frame_times();
    auto lT = frame_times.begin();

    std::list<bool> const& frame_lost_flags = mpTracker->get_frame_lost_flags();
    auto lbL = frame_lost_flags.begin();

    std::list<cv::Mat> const& frame_relative_poses = mpTracker->get_frame_relative_poses();

    for (auto lit = frame_relative_poses.begin(), lend = frame_relative_poses.end();
        lit != lend; ++lit, ++lRit, ++lT, ++lbL)
    {
        if (*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    std::sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    std::ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if (pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    std::vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    std::list<KeyFrame*> const& frame_reference_key_frames = mpTracker->get_frame_reference_key_frames();
    auto lRit = frame_reference_key_frames.begin();

    std::list<double> const& frame_times = mpTracker->get_frame_times();
    auto lT = frame_times.begin();

    std::list<cv::Mat> const& frame_relative_poses = mpTracker->get_frame_relative_poses();

    for (auto lit = frame_relative_poses.begin(), lend = frame_relative_poses.end();
        lit != lend; ++lit, ++lRit, ++lT)
    {
        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while (pKF->isBad())
        {
            Trw = Trw * pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw * pKF->GetPose() * Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::locked_handle_activate_localization()
{
    if (mbActivateLocalizationMode)
    {
        mpLocalMapper->pause();
        mpTracker->InformOnlyTracking(true);
        mbActivateLocalizationMode = false;
    }
}

void System::locked_handle_deactivate_localization()
{
    if (mbDeactivateLocalizationMode)
    {
        mpTracker->InformOnlyTracking(false);
        mpLocalMapper->resume();
        mbDeactivateLocalizationMode = false;
    }
}

void System::locked_handle_reset()
{
    if (mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
}

} //namespace ORB_SLAM
