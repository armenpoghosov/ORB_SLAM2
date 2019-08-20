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

#include "KeyFrame.h"
#include "Map.h"
#include "ORBmatcher.h"

namespace ORB_SLAM2
{

std::atomic<uint64_t> MapPoint::s_next_id;

std::mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(cv::Mat const& worldPos, KeyFrame* pRefKF, Map* pMap)
    :
    mTrackProjX(0.f),
    mTrackProjY(0.f),
    mTrackProjXR(0.f),
    mbTrackInView(false),
    mnTrackScaleLevel(0),
    mTrackViewCos(0.f),
    mnLastFrameSeen((uint64_t)-1),
    mnCorrectedByKF((uint64_t)-1),
    mnCorrectedReference((uint64_t)-1),
    mnFirstKFid(pRefKF->get_id()),
    m_observe_count(0),
    mnId(s_next_id++),
    mpRefKF(pRefKF),
    mnVisible(1),
    mnFound(1),
    mbBad(false),
    mpReplaced(nullptr),
    mfMinDistance(0.f),
    mfMaxDistance(0.f),
    mpMap(pMap)
{
    worldPos.copyTo(mWorldPos);
    mNormalVector = cv::Mat::zeros(3, 1, CV_32F);
}

MapPoint::MapPoint(cv::Mat const& Pos, Map* pMap, Frame* pFrame, int idxF)
    :
    mTrackProjX(0.f),
    mTrackProjY(0.f),
    mTrackProjXR(0.f),
    mbTrackInView(false),
    mnTrackScaleLevel(0),
    mTrackViewCos(0.f),
    mnLastFrameSeen((uint64_t)-1),
    mnCorrectedByKF((uint64_t)-1),
    mnCorrectedReference((uint64_t)-1),
    mnFirstKFid((uint64_t)-1),
    m_observe_count(0),
    mnId(s_next_id++),
    mpRefKF(nullptr),
    mnVisible(1),
    mnFound(1),
    mbBad(false),
    mpReplaced(nullptr),
    mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector / cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    float const dist = (float)cv::norm(PC);
    int const level = pFrame->mvKeysUn[idxF].octave;
    float const levelScaleFactor =  pFrame->mvScaleFactors[level];
    int const nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist * levelScaleFactor;
    mfMinDistance = mfMaxDistance / pFrame->mvScaleFactors[nLevels - 1];

    pFrame->get_descriptor(idxF).copyTo(mDescriptor);
}

void MapPoint::SetWorldPos(cv::Mat const& Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

bool MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    std::unique_lock<std::mutex> lock(mMutexFeatures);

    auto const& pair = mObservations.emplace(pKF, idx);

    if (pair.second)
    {
        if (pKF->mvuRight[idx] >= 0.f)
            m_observe_count += 2;
        else
            ++m_observe_count;
    }

    return pair.second;
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad = false;
    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);

        auto it = mObservations.find(pKF);
        if (it != mObservations.end())
        {
            if (pKF->mvuRight[it->second] >= 0)
                m_observe_count -= 2;
            else
                --m_observe_count;

            mObservations.erase(it);

            if (mpRefKF == pKF)
                mpRefKF = mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if (m_observe_count <= 2)
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
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);
        mbBad = true;
        obs.swap(mObservations);
    }

    for (auto const& pair : obs)
        pair.first->EraseMapPointMatch(pair.second);

    mpMap->EraseMapPoint(this);
}

void MapPoint::Replace(MapPoint* pMP)
{
    if (pMP->mnId == mnId)
        return;

    int nvisible;
    int nfound;
    std::unordered_map<KeyFrame*, size_t> obs;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);

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

void MapPoint::ComputeDistinctiveDescriptors()
{
    std::unordered_map<KeyFrame*, size_t> observations;

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);

        if (mbBad)
            return;

        observations = mObservations;
    }

    if (observations.empty())
        return;

    // Retrieve all observed descriptors
    std::vector<cv::Mat> vDescriptors;
    vDescriptors.reserve(observations.size());

    for (auto const& pair : observations)
    {
        KeyFrame* pKF = pair.first;

        if (!pKF->isBad())
            vDescriptors.emplace_back(pKF->mDescriptors.row((int)pair.second));
    }

    if (vDescriptors.empty())
        return;

    // Compute distances between them
    size_t const N = vDescriptors.size();

    std::vector<int> Distances(N * N);

    for (size_t i = 0; i < N; ++i)
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
        int* it = &Distances[i * N];
        int* itEnd = it + N;

        std::sort(it, itEnd);

        int median = it[(std::size_t)(0.5 * (N - 1))];
        if (median < BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    {
        std::unique_lock<std::mutex> lock(mMutexFeatures);
        mDescriptor = vDescriptors[BestIdx].clone();
    }
}

void MapPoint::UpdateNormalAndDepth()
{
    cv::Mat Pos;
    KeyFrame* pRefKF;
    std::unordered_map<KeyFrame*, std::size_t> observations;

    {
        std::unique_lock<std::mutex> lock1(mMutexFeatures);
        std::unique_lock<std::mutex> lock2(mMutexPos);

        if (mbBad)
            return;

        Pos = mWorldPos.clone();
        pRefKF = mpRefKF;
        observations = mObservations;
    }

    std::size_t const n = observations.size();
    if (n == 0)
        return;

    cv::Mat normal = cv::Mat::zeros(3, 1, CV_32F);

    for (auto const& pair : observations)
    {
        cv::Mat Owi = pair.first->GetCameraCenter();
        cv::Mat normali = Pos - Owi;
        normal += normali / cv::norm(normali);
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter();
    float const dist = (float)cv::norm(PC);
    int const level = pRefKF->mvKeysUn[observations[pRefKF]].octave;
    float const levelScaleFactor =  pRefKF->mvScaleFactors[level];
    int const nLevels = pRefKF->mnScaleLevels;

    {
        std::unique_lock<std::mutex> lock3(mMutexPos);
        mfMaxDistance = dist * levelScaleFactor;
        mfMinDistance = mfMaxDistance / pRefKF->mvScaleFactors[nLevels - 1];
        mNormalVector = normal / (double)n;
    }
}

int MapPoint::PredictScale(float currentDist, KeyFrame* pKF) const
{
    float ratio;
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
        ratio = mfMaxDistance / currentDist;
    }

    int nScale = (int)std::ceil(std::log(ratio) / pKF->mfLogScaleFactor);
    if (nScale < 0)
        nScale = 0;
    else if (nScale >= pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels - 1;

    return nScale;
}

int MapPoint::PredictScale(float currentDist, Frame const* pF) const
{
    float ratio;
    {
        std::unique_lock<std::mutex> lock(mMutexPos);
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
