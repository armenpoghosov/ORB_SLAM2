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
    mnGridCols(FRAME_GRID_COLS),
    mnGridRows(FRAME_GRID_ROWS),
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
    fx(F.fx),
    fy(F.fy),
    cx(F.cx),
    cy(F.cy),
    invfx(F.invfx),
    invfy(F.invfy),
    mbf(F.mbf),
    mb(F.mb),
    mThDepth(F.mThDepth),
    N(F.N),
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
    mnMinX(F.mnMinX),
    mnMinY(F.mnMinY),
    mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY),
    mK(F.mK),
    mvpMapPoints(F.mvpMapPoints),
    mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary),
    mbFirstConnection(true),
    mpParent(NULL),
    mbNotErase(false),
    mbToBeErased(false),
    mbBad(false),
    mHalfBaseline(F.mb / 2),
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
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void KeyFrame::SetPose(cv::Mat const& Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);

    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc * tcw;

    Twc = cv::Mat::eye(4, 4, Tcw.type());
    Rwc.copyTo(Twc.rowRange(0, 3).colRange(0, 3));
    Ow.copyTo(Twc.rowRange(0, 3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.clone();
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose);
    return Twc.clone();
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Ow.clone();
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).colRange(0,3).clone();
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose);
    return Tcw.rowRange(0,3).col(3).clone();
}

void KeyFrame::AddConnection(KeyFrame* pKF, int weight)
{
    {
        unique_lock<mutex> lock(mMutexConnections);

        auto const& pair = mConnectedKeyFrameWeights.insert(std::make_pair(pKF, weight));

        if (!pair.second)
        {
            if (pair.first->second == weight)
            {
                return;
            }

            pair.first->second = weight;
        }
    }

    UpdateBestCovisibles();
}

void KeyFrame::UpdateBestCovisibles()
{
    std::unique_lock<std::mutex> lock(mMutexConnections);

    std::vector<std::pair<int, KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());

    for (auto const& pair : mConnectedKeyFrameWeights)
       vPairs.emplace_back(pair.second, pair.first);

    std::sort(vPairs.begin(), vPairs.end());

    std::vector<KeyFrame*> vkf; vkf.reserve(vPairs.size());
    vector<int> vws; vws.reserve(vPairs.size());

    for (auto it = vPairs.crbegin(), itEnd = vPairs.crend(); it != itEnd; ++it)
    {
        vkf.push_back(it->second);
        vws.push_back(it->first);
    }

    mvpOrderedConnectedKeyFrames = std::move(vkf);
    mvOrderedWeights = std::move(vws);
}

std::set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);

    std::set<KeyFrame*> s;

    for (auto const& pair : mConnectedKeyFrameWeights)
        s.insert(pair.first);

    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections);

    if ((int)mvpOrderedConnectedKeyFrames.size() < N)
        return mvpOrderedConnectedKeyFrames;

    return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + N);
}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(int w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if (mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = std::upper_bound(mvOrderedWeights.begin(), mvOrderedWeights.end(), w, KeyFrame::weightComp);
    if (it == mvOrderedWeights.end())
        return vector<KeyFrame*>();

    int n = it - mvOrderedWeights.begin();
    return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin() + n);
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    return mConnectedKeyFrameWeights.count(pKF) ? mConnectedKeyFrameWeights[pKF] : 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = pMP;
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mvpMapPoints[idx] = nullptr;
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if (idx >= 0)
        mvpMapPoints[idx] = nullptr;
}

std::set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);

    std::set<MapPoint*> s;

    for (MapPoint* pMP : mvpMapPoints)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        s.insert(pMP);
    }

    return s;
}

int KeyFrame::TrackedMapPoints(int minObservations)
{
    unique_lock<mutex> lock(mMutexFeatures);

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

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints;
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mvpMapPoints[idx];
}

void KeyFrame::UpdateConnections()
{
    std::vector<MapPoint*> vpMP;
    {
        unique_lock<mutex> lockMPs(mMutexFeatures);
        vpMP = mvpMapPoints;
    }

    std::map<KeyFrame*, int> KFcounter;

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for (MapPoint* pMP : vpMP)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        std::map<KeyFrame*, size_t> observations = pMP->GetObservations();

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

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax = 0;
    KeyFrame* pKFmax = nullptr;
    int th = 15;

    vector<pair<int, KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());

    for (auto const& pair : KFcounter)
    {
        if (pair.second > nmax)
        {
            nmax = pair.second;
            pKFmax = pair.first;
        }

        if (pair.second >= th)
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

    sort(vPairs.begin(), vPairs.end());

    // TODO: PAE: refactor this crap!
    list<KeyFrame*> lKFs;
    list<int> lWs;
    
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second);
        lWs.push_front(vPairs[i].first);
    }

    {
        unique_lock<mutex> lockCon(mMutexConnections);

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter;
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

        if(mbFirstConnection && mnId!=0)
        {
            mpParent = mvpOrderedConnectedKeyFrames.front();
            mpParent->AddChild(this);
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);

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
        
        if (mnId==0)
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
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for (size_t i = 0, iend = vpConnected.size(); i<iend; i++)
                {
                    for (KeyFrame* pcKF : sParentCandidates)
                    {
                        if (vpConnected[i]->mnId == pcKF->mnId)
                        {
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
        mTcp = Tcw * mpParent->GetPoseInverse();
        mbBad = true;
    }

    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mbBad;
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

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;

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

    vIndices.reserve(N);

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

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    float const z = mvDepth[i];

    if (z <= 0)
        return cv::Mat();

    float const u = mvKeys[i].pt.x;
    float const v = mvKeys[i].pt.y;

    float const x = (u - cx) * z * invfx;
    float const y = (v - cy) * z * invfy;

    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);

    unique_lock<mutex> lock(mMutexPose);
    return Twc.rowRange(0, 3).colRange(0, 3) * x3Dc + Twc.rowRange(0, 3).col(3);
}

float KeyFrame::ComputeSceneMedianDepth(int q)
{
    vector<MapPoint*> vpMapPoints;

    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);

    cv::Mat Rcw2 = Tcw_.row(2).colRange(0, 3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2, 3);

    for (int i = 0; i < N; ++i)
    {
        MapPoint* pMP = mvpMapPoints[i];
        if (pMP != nullptr)
        {
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw) + zcw;
            vDepths.push_back(z);
        }
    }

    // TODO: PAE: get rid of this crap with sorting!
    std::sort(vDepths.begin(), vDepths.end());

    return vDepths[(vDepths.size() - 1) / q];
}

} //namespace ORB_SLAM
