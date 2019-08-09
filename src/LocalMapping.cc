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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include <mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, bool bMonocular)
    :
    mbMonocular(bMonocular),
    mpMap(pMap),
    m_state(eState_Stopped)
{
    m_thread = std::thread(&LocalMapping::run, this);

    // TODO: PAE: review exception safety here
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        while (m_state == eState_Stopped)
            m_kick_condition.wait(lock);
    }
}

LocalMapping::~LocalMapping()
{
    RequestFinish();

    if (m_thread.joinable())
        m_thread.join();
}

void LocalMapping::run()
{
    // TODO: PAE: review exception safety here!

    std::unique_lock<std::mutex> lock(m_mutex);

    for (;;)
    {
        switch (m_state)
        {
        case eState_Stopped:
        {
            m_state = eState_Idle;
            lock.unlock();
            m_kick_condition.notify_all();
            lock.lock();
        }
        break;
        case eState_Idle:
        case eState_Paused:
            m_kick_condition.wait(lock);
        break;
        case eState_Running:
        {
            if (mlNewKeyFrames.empty())
            {
                m_state = eState_Idle;
                lock.unlock();
                m_kick_condition.notify_all();
                lock.lock();

                continue;
            }

            mpCurrentKeyFrame = mlNewKeyFrames.front();
            mlNewKeyFrames.pop_front();

            // we run our processing in unlocked region ...
            lock.unlock();

            // BoW conversion and insertion in Map
            ProcessNewKeyFrame();

            // Check recent MapPoints
            MapPointCulling();

            // Triangulate new MapPoints
            CreateNewMapPoints();

            // PAE: if (queued_key_frames() == 0)
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();

                // PAE: if (queued_key_frames() == 0)
                {
                    // Local BA
                    if (mpMap->KeyFramesInMap() > 2)
                        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, nullptr/*&mbAbortBA*/, mpMap);

                    // Check redundant local Keyframes
                    KeyFrameCulling();
                }
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame); // PAE: pass to loop closer?

            lock.lock();
        }
        break;
        case eState_Stopping:
        {
            m_state = eState_Stopped;
            lock.unlock();
            m_kick_condition.notify_all();
            lock.lock();
        }
        return; // we quit here ...
        case eState_Pausing:
        {
            m_state = eState_Paused;
            lock.unlock();
            m_kick_condition.notify_all();
            lock.lock();
        }
        break;
        case eState_Reset:
        {
            mlNewKeyFrames.clear();
            mlpRecentAddedMapPoints.clear();
            m_state = eState_Idle;
            lock.unlock();
            m_kick_condition.notify_all();
            lock.lock();
        }
        break;
        }
    }
}

void LocalMapping::enqueue_key_frame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    mlNewKeyFrames.push_back(pKF);

    switch (m_state)
    {
    case eState_Stopped:
    break;
    case eState_Idle:
    {
        m_state = eState_Running;
        lock.unlock();
        m_kick_condition.notify_all();
    }
    break;
    case eState_Running: // TODO: review the way it is done
    break;
    case eState_Stopping:
    case eState_Pausing:
    case eState_Paused:
    break;
    }
}

void LocalMapping::ProcessNewKeyFrame()
{
    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW();

    // Associate MapPoints to the new keyframe and update normal and descriptor
    std::vector<MapPoint*> const& vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0; i < vpMapPointMatches.size(); ++i)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if (pMP == nullptr || pMP->isBad())
            continue;

        if (!pMP->IsInKeyFrame(mpCurrentKeyFrame))
        {
            pMP->AddObservation(mpCurrentKeyFrame, i);
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptors();
        }
        else // this can only happen for new stereo points inserted by the Tracking
        {
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints

    uint64_t const nCurrentKFid = mpCurrentKeyFrame->get_id();

    int const cnThObs = mbMonocular ? 2 : 3;

    for (auto lit = mlpRecentAddedMapPoints.begin(); lit != mlpRecentAddedMapPoints.end();)
    {
        MapPoint* pMP = *lit;

        if (pMP->isBad())
            lit = mlpRecentAddedMapPoints.erase(lit);
        else if (pMP->GetFoundRatio() < 0.25f ||
            (nCurrentKFid >= pMP->mnFirstKFid + 2 && pMP->Observations() <= cnThObs))
        {
            pMP->SetBadFlag();
            lit = mlpRecentAddedMapPoints.erase(lit);
        }
        else if (nCurrentKFid >= pMP->mnFirstKFid + 3)
            lit = mlpRecentAddedMapPoints.erase(lit);
        else
            ++lit;
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = mbMonocular ? 20 : 10;

    std::vector<KeyFrame*> const& vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);

    ORBmatcher matcher(0.6f, false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();

    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0,3));
    tcw1.copyTo(Tcw1.col(3));

    cv::Mat Rwc1 = Rcw1.t();

    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    float const fx1 = mpCurrentKeyFrame->fx;
    float const fy1 = mpCurrentKeyFrame->fy;
    
    float const cx1 = mpCurrentKeyFrame->cx;
    float const cy1 = mpCurrentKeyFrame->cy;

    float const invfx1 = mpCurrentKeyFrame->invfx;
    float const invfy1 = mpCurrentKeyFrame->invfy;

    float const ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    // Search matches with epipolar restriction and triangulate
    for (size_t i = 0; i < vpNeighKFs.size(); ++i)
    {
        if (i > 0 && queued_key_frames() != 0)
            return;

        KeyFrame* pKF2 = vpNeighKFs[i];

        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();

        float const baseline_length = (float)cv::norm(Ow2 - Ow1);

        if (!mbMonocular)
        {
            if (baseline_length < pKF2->mb)
                continue;
        }
        else
        {
            float const medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            float const ratioBaselineDepth = baseline_length / medianDepthKF2;

            if (ratioBaselineDepth < 0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t, size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();

        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        float const fx2 = pKF2->fx;
        float const fy2 = pKF2->fy;

        float const cx2 = pKF2->cx;
        float const cy2 = pKF2->cy;

        float const invfx2 = pKF2->invfx;
        float const invfy2 = pKF2->invfy;

        // Triangulate each match
        std::size_t const nmatches = vMatchedIndices.size();

        for (std::size_t ikp = 0; ikp < nmatches; ++ikp)
        {
            auto const& patch_pair = vMatchedIndices[ikp];

            size_t const idx1 = patch_pair.first;
            size_t const idx2 = patch_pair.second;

            cv::KeyPoint const& kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            float const kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool const bStereo1 = kp1_ur >= 0;

            cv::KeyPoint const& kp2 = pKF2->mvKeysUn[idx2];
            float const kp2_ur = pKF2->mvuRight[idx2];
            bool const bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.f);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.f);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            float const cosParallaxRays = (float)(ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2)));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
                cosParallaxStereo1 = std::cos(2 * std::atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = std::cos(2 * std::atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = (std::min)(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if (cosParallaxRays > 0.f && cosParallaxRays < cosParallaxStereo &&
                (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w;
                cv::Mat u;
                cv::Mat vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
                x3D = pKF2->UnprojectStereo(idx2);
            else
                continue; // No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            // Check triangulation in front of cameras
            float const z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0.f)
                continue;

            float const z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0.f)
                continue;

            //Check reprojection error in first keyframe
            float const sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            float const x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            float const y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            float const invz1 = 1.f / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;

                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;

                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;

                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;

                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8f * sigmaSquare1)
                    continue;
            }

            // Check reprojection error in second keyframe
            float const sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            float const x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            float const y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            float const invz2 = 1.f / z2;

            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;
                
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                
                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;

                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;

                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8f * sigmaSquare2)
                    continue;
            }

            // Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            double const dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            double const dist2 = cv::norm(normal2);

            if (dist1 == 0. || dist2 == 0.)
                continue;

            float const ratioDist = (float)(dist2 / dist1);
            float const ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);

            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    std::vector<KeyFrame*> const& vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(mbMonocular ? 20 : 10);
 
    std::unordered_set<KeyFrame*> target_key_frames;

    for (KeyFrame* pKFCV : vpNeighKFs)
    {
        if (pKFCV->isBad())
            continue;

        auto const& pair = target_key_frames.insert(pKFCV);
        if (!pair.second)
            continue;

        std::vector<KeyFrame*> const& neighbours2 = pKFCV->GetBestCovisibilityKeyFrames(5);
        for (KeyFrame* pKFCV2 : neighbours2)
        {
            if (pKFCV2 == mpCurrentKeyFrame || pKFCV2->isBad())
                continue;

            target_key_frames.insert(pKFCV);
        }
    }

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher(0.6f, true);

    std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    // TODO: PAE: think about removing the vector in the call above and turning it to a set
    std::unordered_set<MapPoint*> fuse_candidates;
    fuse_candidates.insert(vpMapPointMatches.begin(), vpMapPointMatches.end());

    for (KeyFrame* pTKF : target_key_frames)
        matcher.Fuse(pTKF, fuse_candidates, 0.3f);

    // Search matches by projection from target KFs in current KF
    fuse_candidates.clear();
    fuse_candidates.reserve(target_key_frames.size() * vpMapPointMatches.size());

    for (KeyFrame* pTKF : target_key_frames)
    {
        std::vector<MapPoint*> const& vpMapPointsKFi = pTKF->GetMapPointMatches();

        for (MapPoint* pMP : vpMapPointsKFi)
        {
            if (pMP == nullptr || pMP->isBad())
                continue;

            fuse_candidates.insert(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, fuse_candidates, 0.3f);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (MapPoint* pMP : vpMapPointMatches)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    cv::Mat const& K1 = pKF1->mK;
    cv::Mat const& K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    std::vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for (KeyFrame* pKF : vpLocalKeyFrames)
    {
        if (pKF->get_id() == 0)
            continue;

        std::vector<MapPoint*> const vpMapPoints = pKF->GetMapPointMatches();

        int const thObs = 3;
        int nRedundantObservations = 0;

        int nMPs = 0;

        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPoints[i];

            if (pMP == nullptr || pMP->isBad() ||
                (!mbMonocular && (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)))
                continue;

            ++nMPs;

            if (pMP->Observations() <= thObs)
                continue;

            int const octave = pKF->mvKeysUn[i].octave;

            std::unordered_map<KeyFrame*, size_t> observations = pMP->GetObservations();

            int nObs = 0;

            for (auto mit = observations.begin(), mend = observations.end(); mit != mend; ++mit)
            {
                KeyFrame* pKFi = mit->first;

                if (pKFi == pKF)
                    continue;

                int const octave_key_frame = pKFi->mvKeysUn[mit->second].octave;

                if (octave_key_frame <= octave + 1 && ++nObs >= thObs)
                    break;
            }

            if (nObs >= thObs)
                ++nRedundantObservations;
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat LocalMapping::SkewSymmetricMatrix(cv::Mat const& v)
{
    return (cv::Mat_<float>(3,3) <<
        0,                  -v.at<float>(2),    v.at<float>(1),
        v.at<float>(2),     0,                  -v.at<float>(0),
        -v.at<float>(1),    v.at<float>(0),     0);
}

bool LocalMapping::pause()
{
    // TODO: PAE: quick and dirty ... has to be rewritten
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_state != eState_Running || m_state != eState_Idle)
        return false;

    m_state = eState_Pausing;

    do
        m_kick_condition.wait(lock);
    while (m_state == eState_Pausing);

    return true;
}

void LocalMapping::resume()
{
    // TODO: PAE: quick and dirty ... has to be rewritten
    std::unique_lock<std::mutex> lock(m_mutex);

    switch (m_state)
    {
    case eState_Stopped:
        return;
    case eState_Idle:
    case eState_Running:
    case eState_Stopping:
    case eState_Pausing:
        break;
    case eState_Paused:
    {
        m_state = eState_Idle;
        m_kick_condition.notify_all();
    }
    break;
    }

    for (KeyFrame* entry : mlNewKeyFrames)
        delete entry;

    mlNewKeyFrames.clear();

    cout << "Local Mapping RESUME" << endl;
}

void LocalMapping::RequestReset()
{
    // PAE quick and dirty ... has to be rewritten
    std::unique_lock<std::mutex> lock(m_mutex);

    if (m_state != eState_Running)
        return;

    m_state = eState_Reset;

    do
        m_kick_condition.wait(lock);
    while (m_state == eState_Reset);
}

} //namespace ORB_SLAM
