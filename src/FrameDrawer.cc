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

#include "FrameDrawer.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/types_c.h>

#include "Map.h"
#include "MapPoint.h"
#include "Tracking.h"
#include "ORBExtractor.h"   // TODO: PAE: remove this header from here

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap)
    :
    mpMap(pMap)
{
    mState = Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame()
{
    cv::Mat im;
    std::vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    std::vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    std::vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    std::vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    {
        std::unique_lock<std::mutex> lock(mMutex);

        state = mState;
        if (mState == Tracking::SYSTEM_NOT_READY)
            mState = Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if (mState == Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if (mState == Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if (mState == Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
        }
    }

    if (im.channels() < 3)
        cvtColor(im, im, CV_GRAY2BGR);

    // Draw
    if (state == Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for (std::size_t i = 0; i < vMatches.size(); ++i)
        {
            if (vMatches[i] >= 0)
                cv::line(im, vIniKeys[i].pt, vCurrentKeys[vMatches[i]].pt, cv::Scalar(0, 255, 0));
        }

        for (cv::KeyPoint const& kp : vCurrentKeys)
            cv::circle(im, kp.pt, 2, cv::Scalar(255, 0, 0), -1);
    }
    else if (state == Tracking::OK) //TRACKING
    {
        mnTracked = 0;
        mnTrackedVO = 0;
        
        float const r = 5;
        std::size_t const n = vCurrentKeys.size();

        for (std::size_t i = 0; i < n; ++i)
        {
            if (!vbVO[i] && !vbMap[i])
            {
                cv::circle(im, vCurrentKeys[i].pt, 2, cv::Scalar(0, 0, 255), -1);
                continue;
            }

            cv::Point2f pt1;
            pt1.x = vCurrentKeys[i].pt.x - r;
            pt1.y = vCurrentKeys[i].pt.y - r;
            
            cv::Point2f pt2;
            pt2.x = vCurrentKeys[i].pt.x + r;
            pt2.y = vCurrentKeys[i].pt.y + r;

            cv::Scalar color = vbMap[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0);

            cv::rectangle(im, pt1, pt2, color);
            cv::circle(im, vCurrentKeys[i].pt, 2, color, -1);

            // This is a match to a MapPoint in the map
            if (vbMap[i])
                ++mnTracked;
            else // This is match to a "visual odometry" MapPoint created in the last frame
                ++mnTrackedVO;
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im, state, imWithInfo);

    return imWithInfo;
}

void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;

    if (nState == Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if (nState == Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if (nState == Tracking::OK)
    {
        if (!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";

        std::size_t nKFs = mpMap->KeyFramesInMap();
        std::size_t nMPs = mpMap->MapPointsInMap();

        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;

        if (mnTrackedVO > 0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if (nState == Tracking::LOST)
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    else if (nState==Tracking::SYSTEM_NOT_READY)
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";

    int baseline = 0;
    cv::Size textSize = cv::getTextSize(s.str(), cv::FONT_HERSHEY_PLAIN, 1, 1, &baseline);

    imText = cv::Mat(im.rows + textSize.height + 10, im.cols, im.type());
    im.copyTo(imText.rowRange(0, im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height + 10, im.cols, im.type());
    cv::putText(imText, s.str(), cv::Point(5, imText.rows - 5), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(255,255,255), 1, 8);
}

void FrameDrawer::Update(Tracking *pTracker)
{
    std::unique_lock<std::mutex> lock(mMutex);

    pTracker->get_current_frame().get_image_left().copyTo(mIm);

    mvCurrentKeys = pTracker->get_current_frame().get_key_points();
    N = mvCurrentKeys.size();

    mvbVO = vector<bool>(N);
    mvbMap = vector<bool>(N);

    mbOnlyTracking = pTracker->is_only_tracking();

    if (pTracker->get_last_state() == Tracking::NOT_INITIALIZED)
    {
        mvIniKeys = pTracker->get_initial_frame().get_key_points();
        mvIniMatches = pTracker->get_ini_matches();
    }
    else if (pTracker->get_last_state() == Tracking::OK)
    {
        for (std::size_t i = 0; i < N; ++i)
        {
            MapPoint* pMP = pTracker->get_current_frame().mvpMapPoints[i];

            if (pMP == nullptr || pTracker->get_current_frame().mvbOutlier[i])
                continue;

            if (pMP->Observations() > 0)
                mvbMap[i] = true;
            else
                mvbVO[i] = true;
        }
    }

    mState = (int)pTracker->get_last_state();
}

} //namespace ORB_SLAM
