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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

std::atomic<uint64_t> MapPoint::s_next_id;

mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(cv::Mat const& Pos, KeyFrame *pRefKF, Map* pMap)
    :
    mnFirstKFid(pRefKF->mnId),
    mnFirstFrame(pRefKF->mnFrameId),
    nObs(0),
    mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0),
    mnBALocalForKF(0),
    mnFuseCandidateForKF(0),
    mnLoopPointForKF(0),
    mnCorrectedByKF(0),
    mnCorrectedReference(0),
    mnBAGlobalForKF(0),
    mpRefKF(pRefKF),
    mnVisible(1),
    mnFound(1),
    mbBad(false),
    mpReplaced(nullptr),
    mfMinDistance(0),
    mfMaxDistance(0),
    mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);

    mnId = s_next_id++;
}

MapPoint::MapPoint(cv::Mat const& Pos, Map* pMap, Frame* pFrame, int idxF)
    :
    mnFirstKFid(-1),
    mnFirstFrame(pFrame->m_id),
    nObs(0),
    mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0),
    mnBALocalForKF(0),
    mnFuseCandidateForKF(0),
    mnLoopPointForKF(0),
    mnCorrectedByKF(0),
    mnCorrectedReference(0),
    mnBAGlobalForKF(0),
    mpRefKF(static_cast<KeyFrame*>(NULL)),
    mnVisible(1),
    mnFound(1),
    mbBad(false),
    mpReplaced(NULL),
    mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = (float)cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    mnId = s_next_id++;
}

void MapPoint::SetWorldPos(cv::Mat const& Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}


void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures);

    auto const& pair = mObservations.emplace(pKF, idx);
    if (pair.second)
    {
        if (pKF->mvuRight[idx] >= 0.f)
            nObs += 2;
        else
            ++nObs;
    }
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        unique_lock<mutex> lock(mMutexFeatures);

        auto it = mObservations.find(pKF);
        if (it != mObservations.end())
        {
            if (pKF->mvuRight[it->second] >= 0)
                nObs -= 2;
            else
                --nObs;

            mObservations.erase(it);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if (nObs <= 2)
                bBad = true;
        }
    }

    if (bBad)
        SetBadFlag();
}

void MapPoint::SetBadFlag()
{
    std::unordered_map<KeyFrame*, size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad = true;
        obs.swap(mObservations);
    }

    for (auto const& pair : obs)
        pair.first->EraseMapPointMatch(pair.second);

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if (pMP->mnId == mnId)
        return;

    int nvisible;
    int nfound;
    std::unordered_map<KeyFrame*, size_t> obs;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);

        obs.swap(mObservations);
        mbBad = true;

        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for (auto const& pair : obs)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = pair.first;

        if (!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(pair.second, pMP);
            pMP->AddObservation(pKF, pair.second);
        }
        else
        {
            pKF->EraseMapPointMatch(pair.second);
        }
    }

    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures);
    unique_lock<mutex> lock2(mMutexPos);
    return mbBad;
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible += n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound += n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (float)mnFound / mnVisible;
}

void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;

    std::unordered_map<KeyFrame*, size_t> observations;

    {
        unique_lock<mutex> lock1(mMutexFeatures);

        if (mbBad)
            return;

        observations = mObservations;
    }

    if (observations.empty())
        return;

    vDescriptors.reserve(observations.size());

    for (auto const& pair : observations)
    {
        KeyFrame* pKF = pair.first;

        if (!pKF->isBad())
            vDescriptors.push_back(pKF->mDescriptors.row((int)pair.second));
    }


    if (vDescriptors.empty())
        return;

    // Compute distances between them
    size_t const N = vDescriptors.size();

    std::vector<int> Distances(N * N);

    for (size_t i=0; i < N; ++i)
    {
        Distances[i * N + i] = 0;

        for (size_t j = i + 1; j < N; ++j)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i], vDescriptors[j]);
            Distances[i * N + j] = distij;
            Distances[j * N + i] = distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX;
    
    std::size_t BestIdx = 0;

    for (std::size_t i = 0; i < N; ++i)
    {
        vector<int> vDists(&Distances[i * N], &Distances[i * N] + N);

        sort(vDists.begin(), vDists.end());
        int median = vDists[(std::size_t)(0.5 * (N - 1))];

        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        unique_lock<mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

std::size_t MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    auto it = mObservations.find(pKF);
    return it != mObservations.end() ? it->second : (std::size_t)-1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations.count(pKF) != 0;
}

void MapPoint::UpdateNormalAndDepth()
{
    std::unordered_map<KeyFrame*, size_t> observations;
    KeyFrame* pRefKF;
    cv::Mat Pos;

    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        if (mbBad)
            return;

        observations = mObservations;
        pRefKF = mpRefKF;
        Pos = mWorldPos.clone();
    }

    std::size_t const n = observations.size();

    if (n == 0)
        return;

    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F);

    for (auto const& pair : observations)
    {
        cv::Mat Owi = pair.first->GetCameraCenter();
        cv::Mat normali = mWorldPos - Owi;
        normal = normal + normali/cv::norm(normali);
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    const float dist = (float)cv::norm(PC);
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level];
    const int nLevels = pRefKF->mnScaleLevels;

    {
        unique_lock<mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / (double)n;
    }
}

int MapPoint::PredictScale(float currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = (int)std::ceil(std::log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

int MapPoint::PredictScale(float currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = (int)std::ceil(std::log(ratio) / pF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pF->mnScaleLevels)
        nScale = pF->mnScaleLevels - 1;

    return nScale;
}

} //namespace ORB_SLAM
