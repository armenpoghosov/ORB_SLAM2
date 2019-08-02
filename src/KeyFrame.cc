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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

uint64_t KeyFrame::s_next_id = 0;

KeyFrame::KeyFrame(Frame& F, Map* pMap, KeyFrameDatabase* pKFDB)
    :
    mnFrameId(F.m_id),
    mTimeStamp(F.mTimeStamp),

    // TODO: PAE: review these
    mnGridCols(Frame::FRAME_GRID_COLS),
    mnGridRows(Frame::FRAME_GRID_ROWS),

    // TODO: PAE: remove these
    mfGridElementWidthInv(F.mfGridElementWidthInv),
    mfGridElementHeightInv(F.mfGridElementHeightInv),

    mnTrackReferenceForFrame(0),
    mnFuseTargetForKF(0),
    mnBALocalForKF(0),
    mnBAFixedForKF(0),
    mnLoopQuery(0),
    mnLoopWords(0),
    mnRelocQuery(0),
    mnRelocWords(0),
    mnBAGlobalForKF(0),

    // TODO: these probably have to be removed too
    fx(F.fx),
    fy(F.fy),
    cx(F.cx),
    cy(F.cy),
    invfx(F.invfx),
    invfy(F.invfy),
    mbf(F.mbf),
    mb(F.mb),
    mThDepth(F.mThDepth),


    m_kf_N(F.m_frame_N),
    mvKeys(F.mvKeys),
    mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight),
    mvDepth(F.mvDepth),
    mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec),
    mFeatVec(F.mFeatVec),
    mnScaleLevels(F.mnScaleLevels),
    mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor),
    mvScaleFactors(F.mvScaleFactors),
    mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2),

    // TODO: these probably have to be removed too
    mnMinX(F.mnMinX),
    mnMinY(F.mnMinY),
    mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY),
    mK(F.mK),

    mvpMapPoints(F.mvpMapPoints),
    mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary),
    mbFirstConnection(true),
    mpParent(nullptr),
    mbNotErase(false),
    mbToBeErased(false),
    mbBad(false),
    mpMap(pMap)
{
    mnId = s_next_id++;

    mGrid.resize(mnGridCols);

    for (int i = 0; i < mnGridCols; ++i)
    {
        mGrid[i].resize(mnGridRows);

        for (int j = 0; j < mnGridRows; ++j)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);
}

void KeyFrame::ComputeBoW()
{
    if (mBowVec.empty() || mFeatVec.empty())
    {
        std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void KeyFrame::SetPose(cv::Mat const& Tcw)
{
    std::unique_lock<std::mutex> lock(mMutexPose);

    Tcw.copyTo(m_Tcw);

    cv::Mat Rcw = m_Tcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat tcw = m_Tcw.rowRange(0, 3).col(3);

    cv::Mat Rwc = Rcw.t();
    cv::Mat Ow = -Rwc * tcw;

    m_Twc = cv::Mat::eye(4, 4, m_Tcw.type());
    Rwc.copyTo(m_Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(m_Twc.rowRange(0, 3).col(3));
}

void KeyFrame::AddConnection(KeyFrame* pKF, std::size_t points)
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);

        auto const& pair = mConnectedKeyFrameWeights.emplace(pKF, points);
        if (!pair.second)
        {
            if (pair.first->second == points)
                return;

            pair.first->second = points;
        }
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    std::vector<std::pair<std::size_t, KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());

    for (auto const& pair : mConnectedKeyFrameWeights)
       vPairs.emplace_back(pair.second, pair.first);

    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame*> vkf; vkf.reserve(vPairs.size());
    std::vector<std::size_t> vws; vws.reserve(vPairs.size());

    for (auto it = vPairs.crbegin(), itEnd = vPairs.crend(); it != itEnd; ++it)
    {
        vkf.push_back(it->second);
        vws.push_back(it->first);
    }

    mvpOrderedConnectedKeyFrames = std::move(vkf);
    mvOrderedWeights = std::move(vws);
}

std::unordered_set<KeyFrame*> KeyFrame::GetConnectedKeyFrames() const
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    std::unordered_set<KeyFrame*> s;

    for (auto const& pair : mConnectedKeyFrameWeights)
        s.insert(pair.first);

    return s;
}

std::vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(std::size_t N) const
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;

    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

std::vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(int w) const
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    auto it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, std::greater<std::size_t>());
    if (it == mvOrderedWeights.end())
        return std::vector<KeyFrame*>();

    std::size_t n = it - mvOrderedWeights.begin();
    return std::vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    std::size_t index = pMP->GetIndexInKeyFrame(this);
    if (index != (std::size_t)-1)
        mvpMapPoints[index] = nullptr;
}

std::unordered_set<MapPoint*> KeyFrame::GetMapPoints() const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);

    std::unordered_set<MapPoint*> s;

    for (MapPoint* pMP : mvpMapPoints)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        s.insert(pMP);
    }

    return s;
}

int KeyFrame::TrackedMapPoints(std::size_t minObservations) const
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);

    int nPoints = 0;

    for (MapPoint* pMP : mvpMapPoints)
    {
        if (pMP != nullptr && !pMP->isBad() &&
            (minObservations <= 0 || pMP->Observations() >= minObservations))
        {
            ++nPoints;
        }
    }

    return nPoints;
}

void KeyFrame::UpdateConnections()
{
    std::vector<MapPoint*> vpMP;
    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    std::unordered_map<KeyFrame*, std::size_t> KFcounter;

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (MapPoint* pMP : vpMP)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        std::unordered_map<KeyFrame*, size_t> const& observations = pMP->GetObservations();

        for (auto const& pair : observations)
        {
            if (pair.first->mnId == mnId)
                continue;

            ++KFcounter[pair.first];
        }
    }

    // This should not happen
    if (KFcounter.empty())
        return;

    // If the counter is greater than threshold add connection
    // In case no keyframe counter is over threshold add the one with maximum counter
    std::size_t const threshold = 15;

    std::size_t nmax = 0;
    KeyFrame* pKFmax = nullptr;

    std::vector<std::pair<std::size_t, KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());

    for (auto const& pair : KFcounter)
    {
        if (pair.second > nmax)
        {
            nmax = pair.second;
            pKFmax = pair.first;
        }

        if (pair.second >= threshold)
        {
            vPairs.emplace_back(pair.second, pair.first);
            pair.first->AddConnection(this, pair.second);
        }
    }

    if (vPairs.empty())
    {
        vPairs.emplace_back(nmax, pKFmax);
        pKFmax->AddConnection(this, nmax);
    }

    std::sort(vPairs.begin(), vPairs.end());

    // TODO: PAE: refactored but still crappy!
    std::vector<KeyFrame*> vpOrderedConnectedKeyFrames;
    vpOrderedConnectedKeyFrames.reserve(vPairs.size());

    std::vector<std::size_t> vOrderedWeights;
    vOrderedWeights.reserve(vPairs.size());

    for (auto it = vPairs.crbegin(), itEnd = vPairs.crend(); it != itEnd; ++it)
    {
        vpOrderedConnectedKeyFrames.push_back(it->second);
        vOrderedWeights.push_back(it->first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = std::move(KFcounter);
        mvpOrderedConnectedKeyFrames = std::move(vpOrderedConnectedKeyFrames);
        mvOrderedWeights = std::move(vOrderedWeights);

        if (mbFirstConnection && mnId != 0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }
    }
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

void KeyFrame::SetErase()
{
    {
        std::unique_lock<std::mutex> lock(mMutexConnections);

        if (mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if (mbToBeErased)
        SetBadFlag();
}

void KeyFrame::SetBadFlag()
{
    {
        unique_lock<mutex> lock(mMutexConnections);

        if (mnId == 0)
            return;

        if (mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for (auto const& pair : mConnectedKeyFrameWeights)
        pair.first->EraseConnection(this);

    for (MapPoint* pMP : mvpMapPoints)
        if (pMP != nullptr)
            pMP->EraseObservation(this);

    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while (!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for (KeyFrame* pKF : mspChildrens)
            {
                if (pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                std::vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i<iend; i++)
                {
                    for (KeyFrame* pcKF : sParentCandidates)
                    {
                        if (vpConnected[i]->mnId != pcKF->mnId)
                            continue;

                        int w = pKF->GetWeight(vpConnected[i]);
                        if (w > max)
                        {
                            pC = pKF;
                            pP = vpConnected[i];
                            max = w;
                            bContinue = true;
                        }
                    }
                }
            }

            if (bContinue)
                break;

            pC->ChangeParent(pP);
            sParentCandidates.insert(pC);
            mspChildrens.erase(pC);
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        for (KeyFrame* pKF : mspChildrens)
            pKF->ChangeParent(mpParent);

        mpParent->EraseChild(this);
        mTcp = m_Tcw * mpParent->GetPoseInverse();
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate;
    {
        std::unique_lock<mutex> lock(mMutexConnections);
        bUpdate = mConnectedKeyFrameWeights.erase(pKF) != 0;
    }

    if (bUpdate)
        UpdateBestCovisibles();
}

std::vector<size_t> KeyFrame::GetFeaturesInArea(float x, float y, float r) const
{
    std::vector<std::size_t> vIndices;

    int const nMinCellX = (std::max)(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= mnGridCols)
        return vIndices;

    int const nMaxCellX = (std::min)((int)mnGridCols - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    int const nMinCellY = (std::max)(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= mnGridRows)
        return vIndices;

    int const nMaxCellY = (std::min)((int)mnGridRows - 1,(int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    vIndices.reserve(m_kf_N);

    for (int ix = nMinCellX; ix <= nMaxCellX; ++ix)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; ++iy)
        {
            vector<size_t> const& vCell = mGrid[ix][iy];

            for (std::size_t index : vCell)
            {
                cv::KeyPoint const& kpUn = mvKeysUn[index];

                if (fabs(kpUn.pt.x - x) < r && fabs(kpUn.pt.y - y) < r)
                    vIndices.push_back(index);
            }
        }
    }

    return vIndices;
}

cv::Mat KeyFrame::UnprojectStereo(std::size_t i)
{
    float const z = mvDepth[i];

    if (z <= 0)
        return cv::Mat();

    float const u = mvKeys[i].pt.x;
    float const v = mvKeys[i].pt.y;

    float const x = (u - cx) * z * invfx;
    float const y = (v - cy) * z * invfy;

    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

    std::unique_lock<std::mutex> lock(mMutexPose);
    return m_Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + m_Twc.rowRange(0, 3).col(3);
}

float KeyFrame::ComputeSceneMedianDepth(int q) const
{
    cv::Mat Tcw;
    std::vector<MapPoint*> vpMapPoints;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPose);
        Tcw = m_Tcw.clone();
        vpMapPoints = mvpMapPoints;
    }

    std::vector<float> vDepths;
    vDepths.reserve(m_kf_N);

    cv::Mat Rcw2 = Tcw.row(2).colRange(0, 3).t();
    float const zcw = Tcw.at<float>(2, 3);

    for (std::size_t i = 0; i < m_kf_N; ++i)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if (pMP == nullptr)
            continue;

        cv::Mat x3Dw = pMP->GetWorldPos();
        vDepths.push_back((float)Rcw2.dot(x3Dw));
    }

    // TODO: PAE: what if it is empty!?
    std::size_t const index = (vDepths.size() - 1) / q;
    std::nth_element(vDepths.begin(), vDepths.begin() + index, vDepths.end());
    return vDepths[index] + zcw;
}

} //namespace ORB_SLAM
