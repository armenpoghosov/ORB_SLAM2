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

#include "ORBmatcher.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"

#include <cstdint>
#include <limits>
#include <unordered_set>

using namespace std;

namespace ORB_SLAM2
{

ORBmatcher::ORBmatcher(float nnratio, bool check_orientation)
    :
    mfNNratio(nnratio),
    m_check_orientation(check_orientation)
{}

std::size_t ORBmatcher::SearchByProjection(Frame& F,
    std::unordered_set<MapPoint*> const& vpMapPoints, float th, std::ofstream* ofs)
{
    // -----------------------------------------------------------------------------
    // TODO: PAE: this is just to ensure we avoid randomization
    std::vector<MapPoint*> points;
    points.reserve(vpMapPoints.size());
    points.insert(points.end(), vpMapPoints.begin(), vpMapPoints.end());
    std::sort(points.begin(), points.end(),
        [] (MapPoint const* p1, MapPoint const* p2) -> bool
        {
            return p1->get_id() < p2->get_id();
        });
    // -----------------------------------------------------------------------------

    std::size_t nmatches = 0;

    for (MapPoint* pMP : points)
    {
        if (!pMP->mbTrackInView || pMP->isBad())
            continue;

        // The size of the window will depend on the viewing direction
        float const r = RadiusByViewingCos(pMP->mTrackViewCos) * th;

        int const nPredictedLevel = pMP->mnTrackScaleLevel;

        std::vector<std::size_t> const& vIndices = F.GetFeaturesInArea(pMP->mTrackProjX, pMP->mTrackProjY,
            r * F.mvScaleFactors[nPredictedLevel], nPredictedLevel - 1, nPredictedLevel);

        /*
        if (F.get_id() == 221 && pMP->get_id() == 1368 && ofs != nullptr)
        {
            (*ofs) << "--------------------------------------------------" << std::endl;
            (*ofs) << pMP->mTrackProjX << ',' << pMP->mTrackProjY << std::endl;
            (*ofs) << pMP->GetMinDistanceInvariance() << ',' << pMP->GetMaxDistanceInvariance() << std::endl;
            (*ofs) << r * F.mvScaleFactors[nPredictedLevel] << ',' << nPredictedLevel - 1 << ',' << nPredictedLevel << std::endl;
            (*ofs) << "indices" << std::endl;
            for (std::size_t index : vIndices)
                (*ofs) << index << '|';
            (*ofs) << std::endl << "--------------------------------------------------" << std::endl;
        }*/

        if (vIndices.empty())
            continue;

        cv::Mat const& MPdescriptor = pMP->GetDescriptor();

        int bestDist = TH_HIGH;
        int bestLevel = -1;

        int bestDist2 = TH_HIGH;
        int bestLevel2 = -1;

        int bestIdx = -1;

        // Get best and second matches with near keypoints
        for (size_t const idx : vIndices)
        {
            MapPoint const* pMPIdx = F.mvpMapPoints[idx];

            // TODO: PAE: non-determinizm possible if we remove the sorting of points above!
            if (pMPIdx != nullptr && pMPIdx->Observations() != 0)
                continue;

            if (F.mvuRight[idx] > 0)
            {
                float const er = std::fabs(pMP->mTrackProjXR - F.mvuRight[idx]);
                if (er > r * F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            cv::Mat const& d = F.get_descriptor((int)idx);

            int const dist = DescriptorDistance(MPdescriptor, d);
            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestLevel2 = bestLevel;

                bestDist = dist;
                bestLevel = F.mvKeysUn[idx].octave;

                bestIdx = (int)idx;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
                bestLevel2 = F.mvKeysUn[idx].octave;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if (bestDist < TH_HIGH)
        {
            if (bestLevel == bestLevel2 && bestDist > mfNNratio * bestDist2)
                continue;

            F.mvpMapPoints[bestIdx] = pMP;
            ++nmatches;
        }
    }

    return nmatches;
}

bool ORBmatcher::CheckDistEpipolarLine(cv::KeyPoint const& kp1,
    cv::KeyPoint const& kp2, cv::Mat const& F12, KeyFrame const* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    float const a = kp1.pt.x * F12.at<float>(0, 0) + kp1.pt.y * F12.at<float>(1, 0) + F12.at<float>(2, 0);
    float const b = kp1.pt.x * F12.at<float>(0, 1) + kp1.pt.y * F12.at<float>(1, 1) + F12.at<float>(2, 1);
    float const c = kp1.pt.x * F12.at<float>(0, 2) + kp1.pt.y * F12.at<float>(1, 2) + F12.at<float>(2, 2);

    float const num = a * kp2.pt.x + b * kp2.pt.y + c;

    float const den = a * a + b * b;

    if (den == 0)
        return false;

    float const dsqr = num * num / den;

    return dsqr < 3.84 * pKF2->mvLevelSigma2[kp2.octave];
}

std::size_t ORBmatcher::SearchByBoW(KeyFrame const* pKF, Frame const& F, std::vector<MapPoint*>& vMatches)
{
    std::size_t nmatches = 0;

    std::vector<int> rotHist[HISTO_LENGTH];
    if (m_check_orientation)
    {
        for (int i = 0; i < HISTO_LENGTH; ++i)
            rotHist[i].reserve(500);
    }

    vMatches = std::vector<MapPoint*>(F.get_frame_N());

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    DBoW2::FeatureVector const& vFeatVecKF = pKF->get_BoW_features();
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();

    DBoW2::FeatureVector const& vFeatVecFrame = F.get_BoW_features();
    DBoW2::FeatureVector::const_iterator Fit = vFeatVecFrame.begin();
    DBoW2::FeatureVector::const_iterator Fend = vFeatVecFrame.end();

    std::vector<MapPoint*> const& vpMapPointsKF = pKF->GetMapPointMatches();

    while (KFit != KFend && Fit != Fend)
    {
        if (KFit->first == Fit->first)
        {
            std::vector<uint32_t> const& vIndicesKF = KFit->second;
            std::vector<uint32_t> const& vIndicesF = Fit->second;

            for (uint32_t const realIdxKF : vIndicesKF)
            {
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if (pMP == nullptr || pMP->isBad())
                    continue;

                cv::Mat const& dKF= pKF->mDescriptors.row(realIdxKF);

                int bestDist1 = TH_LOW;
                int bestDist2 = TH_LOW;
                std::size_t bestIdxF = (std::size_t)-1;

                for (uint32_t realIdxF : vIndicesF)
                {
                    if (vMatches[realIdxF] != nullptr)
                        continue;

                    cv::Mat const& dF = F.get_descriptor(realIdxF);

                    int const dist =  DescriptorDistance(dKF, dF);
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdxF = realIdxF;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 < TH_LOW && (float)bestDist1 < mfNNratio * (float)bestDist2)
                {
                    vMatches[bestIdxF] = pMP;

                    cv::KeyPoint const& kp = pKF->mvKeysUn[realIdxKF];

                    if (m_check_orientation)
                    {
                        float rot = kp.angle - F.get_key_points()[bestIdxF].angle;
                        if (rot < 0.f)
                            rot += 360.f;

                        int bin = (int)(rot * (HISTO_LENGTH / 360.f));
                        if (bin == HISTO_LENGTH)
                            bin = 0;

                        assert(bin >= 0 && bin < HISTO_LENGTH);
                        rotHist[bin].push_back((int)bestIdxF);
                    }

                    ++nmatches;
                }
            }

            ++KFit;
            ++Fit;
        }
        else if (KFit->first < Fit->first)
            KFit = vFeatVecKF.lower_bound(Fit->first);
        else
            Fit = vFeatVecFrame.lower_bound(KFit->first);
    }

    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int> const& v_histo_entry = rotHist[i];

            for (int histo_index : v_histo_entry)
                vMatches[histo_index] = nullptr;

            nmatches -= v_histo_entry.size();
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw,
    std::unordered_set<MapPoint*> const& vpPoints, std::vector<MapPoint*>& vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    float const fx = pKF->fx;
    float const fy = pKF->fy;
    float const cx = pKF->cx;
    float const cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    float const scw = std::sqrtf((float)sRcw.row(0).dot(sRcw.row(0)));

    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0, 3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw;

    // Set of MapPoints already found in the KeyFrame
    int nmatches = 0;

    // For each Candidate MapPoint Project and Match
    for (MapPoint* pMP : vpPoints)
    {
        // Discard Bad MapPoints and already found
        if (pMP->isBad())
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.f)
            continue;

        // Project into Image
        float const invz = 1 / p3Dc.at<float>(2);
        float const x = p3Dc.at<float>(0) * invz;
        float const y = p3Dc.at<float>(1) * invz;

        float const u = fx * x + cx;
        float const v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        // Depth must be inside the scale invariance region of the point
        float const maxDistance = pMP->GetMaxDistanceInvariance();
        float const minDistance = pMP->GetMinDistanceInvariance();

        cv::Mat PO = p3Dw - Ow;
        float const dist = (float)cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        float const radius = th * pKF->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> const vIndices = pKF->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        cv::Mat const dMP = pMP->GetDescriptor();

        int bestDist = 256;
        std::size_t bestIdx = -1;

        for (size_t const idx : vIndices)
        {
            if (vpMatched[idx] != nullptr)
                continue;

            int const& kpLevel= pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            cv::Mat const& dKF = pKF->mDescriptors.row((int)idx);

            int const dist = DescriptorDistance(dMP,dKF);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if (bestDist <= TH_LOW)
        {
            vpMatched[bestIdx] = pMP;
            ++nmatches;
        }
    }

    return nmatches;
}

std::size_t ORBmatcher::SearchForInitialization(Frame const& F1, Frame const& F2,
    std::vector<cv::Point2f>& vbPrevMatched, vector<int>& vnMatches12, int windowSize)
{
    std::vector<int> rotHist[HISTO_LENGTH];

    if (m_check_orientation)
    {
        for (int i = 0; i < HISTO_LENGTH; ++i)
            rotHist[i].reserve(500);
    }

    std::size_t const size_keys_1 = F1.mvKeysUn.size();
    std::size_t const size_keys_2 = F2.mvKeysUn.size();

    std::size_t nmatches = 0;
    vnMatches12 = std::vector<int>(size_keys_1, -1);

    std::vector<int> vnMatches21(size_keys_2, -1);
    std::vector<int> vMatchedDistance(size_keys_2, INT_MAX);

    for (size_t i1 = 0, iend1 = size_keys_1; i1 < iend1; ++i1)
    {
        cv::KeyPoint const& kp1 = F1.mvKeysUn[i1];

        int const level1 = kp1.octave;
        //if (level1 > 0) ///TODO: PAE: hmmm matching done on features in the first octave only!
        //    continue;

        std::vector<size_t> vIndices2 = F2.GetFeaturesInArea(
            vbPrevMatched[i1].x, vbPrevMatched[i1].y, windowSize, level1, level1);

        if (vIndices2.empty())
            continue;

        cv::Mat const& d1 = F1.get_descriptor((int)i1);

        int bestDist = std::numeric_limits<int>::max();

        std::size_t bestIdx2 = -1;
        int bestDist2 = std::numeric_limits<int>::max();

        for (std::size_t i2 : vIndices2)
        {
            cv::Mat const& d2 = F2.get_descriptor((int)i2);

            int const dist = DescriptorDistance(d1, d2);
            if (vMatchedDistance[i2] <= dist)
                continue;

            if (dist < bestDist)
            {
                bestDist2 = bestDist;
                bestDist = dist;
                bestIdx2 = i2;
            }
            else if (dist < bestDist2)
            {
                bestDist2 = dist;
            }
        }

        if (bestDist > TH_LOW || bestDist >= bestDist2 * mfNNratio)
            continue;

        if (vnMatches21[bestIdx2] >= 0)
        {
            vnMatches12[vnMatches21[bestIdx2]] = -1;
            --nmatches;
        }

        vnMatches12[i1] = bestIdx2;
        vnMatches21[bestIdx2] = (int)i1; // TODO: PAE: look at it deeper
        vMatchedDistance[bestIdx2] = bestDist;
        ++nmatches;

        if (m_check_orientation)
        {
            float rot = F1.mvKeysUn[i1].angle - F2.mvKeysUn[bestIdx2].angle;

            if (rot < 0.f)
                rot += 360.f;

            int bin = (int)(rot * (HISTO_LENGTH / 360.f));
            if (bin == HISTO_LENGTH)
                bin = 0;

            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(i1);
        }
    }

    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int> const& v_histo_entry = rotHist[i];

            for (int idx1 : v_histo_entry)
            {
                int& r_match_entry = vnMatches12[idx1];
                if (r_match_entry >= 0)
                {
                    r_match_entry = -1;
                    --nmatches;
                }
            }
        }
    }

    // Update prev matched
    for (size_t i1 = 0; i1 < size_keys_1; ++i1)
    {
        int const match_12 = vnMatches12[i1];
        if (match_12 >= 0)
            vbPrevMatched[i1] = F2.mvKeysUn[match_12].pt;
    }

    return nmatches;
}

int ORBmatcher::SearchByBoW(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint*>& vpMatches12)
{
    std::vector<cv::KeyPoint> const& vKeysUn1 = pKF1->mvKeysUn;
    std::vector<cv::KeyPoint> const& vKeysUn2 = pKF2->mvKeysUn;

    DBoW2::FeatureVector const& vFeatVec1 = pKF1->get_BoW_features();
    DBoW2::FeatureVector const& vFeatVec2 = pKF2->get_BoW_features();

    std::vector<MapPoint*> const vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<MapPoint*> const vpMapPoints2 = pKF2->GetMapPointMatches();

    cv::Mat const& Descriptors1 = pKF1->mDescriptors;
    cv::Mat const& Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = std::vector<MapPoint*>(vpMapPoints1.size());
    std::vector<bool> vbMatched2(vpMapPoints2.size());

    std::vector<int> rotHist[HISTO_LENGTH];
    for (int i = 0; i < HISTO_LENGTH; ++i)
        rotHist[i].reserve(500);

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();

    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (std::size_t const idx1 : f1it->second)
            {
                MapPoint* pMP1 = vpMapPoints1[idx1];
                if (pMP1 == nullptr || pMP1->isBad())
                    continue;

                cv::Mat const& d1 = Descriptors1.row((int)idx1);

                int bestDist1 = 256;
                int bestIdx2 = -1 ;
                int bestDist2 = 256;

                for (std::size_t const idx2 : f2it->second)
                {
                    if (vbMatched2[idx2])
                        continue;

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if (pMP2 == nullptr || pMP2->isBad())
                        continue;

                    cv::Mat const& d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1, d2);
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }

                if (bestDist1 < TH_LOW && bestDist1 < mfNNratio * bestDist2)
                {
                    vpMatches12[idx1] = vpMapPoints2[bestIdx2];
                    vbMatched2[bestIdx2] = true;

                    if (m_check_orientation)
                    {
                        float rot = vKeysUn1[idx1].angle - vKeysUn2[bestIdx2].angle;
                        if (rot < 0.0)
                            rot += 360.0f;

                        int bin = (int)(rot * (HISTO_LENGTH / 360.f));
                        if (bin == HISTO_LENGTH)
                            bin = 0;

                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }

                    ++nmatches;
                }
            }

            ++f1it;
            ++f2it;
        }
        else if (f1it->first < f2it->first)
            f1it = vFeatVec1.lower_bound(f2it->first);
        else
            f2it = vFeatVec2.lower_bound(f1it->first);
    }

    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int>& v_histo_entry = rotHist[i];

            for (int histo_index : v_histo_entry)
                vpMatches12[histo_index] = nullptr;

            nmatches -= (int)v_histo_entry.size();
        }
    }

    return nmatches;
}

std::size_t ORBmatcher::SearchForTriangulation(KeyFrame* pKF1, KeyFrame* pKF2,
    cv::Mat const& F12, std::vector<std::pair<std::size_t, std::size_t> >& matches, bool bOnlyStereo)
{
    // Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    
    cv::Mat C2 = R2w * Cw + t2w;

    float const invz = 1.f / C2.at<float>(2);
    float const ex = pKF2->fx * C2.at<float>(0) * invz + pKF2->cx;
    float const ey = pKF2->fy * C2.at<float>(1) * invz + pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    std::size_t nmatches = 0;
    std::vector<std::size_t> vMatches12(pKF1->m_kf_N, (std::size_t)-1);

    std::vector<int> rotHist[HISTO_LENGTH];
    if (m_check_orientation)
    {
        for (int i = 0; i < HISTO_LENGTH; ++i)
            rotHist[i].reserve(500);
    }

    DBoW2::FeatureVector const& vFeatVec1 = pKF1->get_BoW_features();
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator const f1end = vFeatVec1.end();

    DBoW2::FeatureVector const& vFeatVec2 = pKF2->get_BoW_features();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator const f2end = vFeatVec2.end();

    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (std::size_t const idx1 : f1it->second)
            {
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);

                // If there is already a MapPoint skip
                if (pMP1 != nullptr)
                    continue;

                bool const bStereo1 = pKF1->mvuRight[idx1] >= 0;
                if (bOnlyStereo && !bStereo1)
                    continue;

                cv::KeyPoint const& kp1 = pKF1->mvKeysUn[idx1];

                cv::Mat const& d1 = pKF1->mDescriptors.row(idx1);

                int bestDist = TH_LOW;
                std::size_t bestIdx2 = (std::size_t)-1;

                for (std::size_t const idx2 : f2it->second)
                {
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);

                    // If we have already matched or there is a MapPoint skip
                    if (pMP2 != nullptr)
                        continue;

                    bool const bStereo2 = pKF2->mvuRight[idx2] >= 0;
                    if (bOnlyStereo && !bStereo2)
                        continue;
                    
                    cv::Mat const& d2 = pKF2->mDescriptors.row(idx2);
                    
                    int const dist = DescriptorDistance(d1, d2);
                    if (dist >= bestDist)
                        continue;

                    cv::KeyPoint const& kp2 = pKF2->mvKeysUn[idx2];

                    if (!bStereo1 && !bStereo2)
                    {
                        float const distex = ex - kp2.pt.x;
                        float const distey = ey - kp2.pt.y;

                        if (distex * distex + distey * distey < 100 * pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if (CheckDistEpipolarLine(kp1, kp2, F12, pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }

                if (bestDist >= TH_LOW)
                    continue;

                cv::KeyPoint const& kp2 = pKF2->mvKeysUn[bestIdx2];
                vMatches12[idx1] = bestIdx2;
                ++nmatches;

                if (m_check_orientation)
                {
                    float rot = kp1.angle - kp2.angle;
                    if (rot < 0.f)
                        rot += 360.f;

                    int bin = (int)(rot * (HISTO_LENGTH / 360.f));
                    if (bin == HISTO_LENGTH)
                        bin = 0;

                    assert(bin >= 0 && bin < HISTO_LENGTH);
                    rotHist[bin].push_back(idx1);
                }
            }

            ++f1it;
            ++f2it;
        }
        else if (f1it->first < f2it->first)
            f1it = vFeatVec1.lower_bound(f2it->first);
        else
            f2it = vFeatVec2.lower_bound(f1it->first);
    }

    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int> const& histo_entry = rotHist[i];

            for (int histo_index : histo_entry)
                vMatches12[histo_index] = (std::size_t)-1;

            nmatches -= histo_entry.size();
        }
    }

    matches.clear();
    matches.reserve(nmatches);

    for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        if (vMatches12[i] == (std::size_t)-1)
            continue;

        matches.emplace_back(i, vMatches12[i]);
    }

    return nmatches;
}

std::size_t ORBmatcher::Fuse(KeyFrame* pKF, std::unordered_set<MapPoint*> const& map_points, float th)
{
    float const fx = pKF->fx;
    float const fy = pKF->fy;
    float const cx = pKF->cx;
    float const cy = pKF->cy;
    float const bf = pKF->mbf;

    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();
    cv::Mat Ow = pKF->GetCameraCenter();

    std::size_t nFused = 0;

    //-------------------------------------------------------------------------------------------
    // TODO: PAE: removing non determinizm introduced by unordered_set here
    std::vector<MapPoint*> map_points_vector;
    map_points_vector.reserve(map_points.size());
    map_points_vector.insert(map_points_vector.end(), map_points.begin(), map_points.end());
    std::sort(map_points_vector.begin(), map_points_vector.end(),
        [] (MapPoint const* p1, MapPoint const* p2)->bool
        {
            return p1->get_id() < p2->get_id();
        });
    //-------------------------------------------------------------------------------------------

    for (MapPoint* pMP : map_points_vector)
    {
        if (pMP == nullptr || pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.f)
            continue;

        float const invz = 1.f / p3Dc.at<float>(2);
        float const x = p3Dc.at<float>(0) * invz;
        float const y = p3Dc.at<float>(1) * invz;

        float const u = fx * x + cx;
        float const v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        float const ur = u - bf * invz;

        float const maxDistance = pMP->GetMaxDistanceInvariance();
        float const minDistance = pMP->GetMinDistanceInvariance();

        cv::Mat PO = p3Dw - Ow;
        float const dist3D = (float)cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();
        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        int const nPredictedLevel = pMP->PredictScale(dist3D, pKF);

        // Search in a radius
        float const radius = th * pKF->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> const& vIndices = pKF->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        cv::Mat const& dMP = pMP->GetDescriptor();

        int bestDist = TH_LOW;
        std::size_t bestIdx = -1;

        for (size_t const idx : vIndices)
        {
            cv::KeyPoint const& kp = pKF->mvKeysUn[idx];

            int const kpLevel= kp.octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            float const kpx = kp.pt.x;
            float const kpy = kp.pt.y;
            float const kpr = pKF->mvuRight[idx];

            float const ex = u - kpx;
            float const ey = v - kpy;

            if (kpr >= 0.f)
            {
                // Check reprojection error in stereo
                float const er = ur - kpr;
                float const e2 = ex * ex + ey * ey + er * er;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 7.8)
                    continue;
            }
            else
            {
                const float e2 = ex * ex + ey * ey;

                if (e2 * pKF->mvInvLevelSigma2[kpLevel] > 5.99)
                    continue;
            }

            cv::Mat const& dKF = pKF->mDescriptors.row((int)idx);

            int const dist = DescriptorDistance(dMP, dKF);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist < TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);

            if (pMPinKF == nullptr)
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            else if (!pMPinKF->isBad())
            {
                if (pMPinKF->Observations() > pMP->Observations())
                    pMP->Replace(pMPinKF);
                else
                    pMPinKF->Replace(pMP);
            }

            ++nFused;
        }
    }

    return nFused;
}

void ORBmatcher::Fuse(KeyFrame* pKF, cv::Mat Scw, std::unordered_set<MapPoint*> const& vpPoints,
    float th, std::unordered_map<MapPoint*, MapPoint*>& vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    float const fx = pKF->fx;
    float const fy = pKF->fy;
    float const cx = pKF->cx;
    float const cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    float const scw = std::sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw / scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3) / scw;
    cv::Mat Ow = -Rcw.t() * tcw;

    // Set of MapPoints already found in the KeyFrame
    std::unordered_set<MapPoint*> const& spAlreadyFound = pKF->GetMapPoints();

    // For each candidate MapPoint project and match
    for (MapPoint* pMP : vpPoints)
    {
        // Discard Bad MapPoints and already found
        if (pMP->isBad() || spAlreadyFound.find(pMP) != spAlreadyFound.end())
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw * p3Dw + tcw;

        // Depth must be positive
        if (p3Dc.at<float>(2) < 0.f)
            continue;

        // Project into Image
        float const invz = 1.f / p3Dc.at<float>(2);
        float const x = p3Dc.at<float>(0) * invz;
        float const y = p3Dc.at<float>(1) * invz;

        float const u = fx * x + cx;
        float const v = fy * y + cy;

        // Point must be inside the image
        if (!pKF->IsInImage(u, v))
            continue;

        // Depth must be inside the scale pyramid of the image
        float const maxDistance = pMP->GetMaxDistanceInvariance();
        float const minDistance = pMP->GetMinDistanceInvariance();

        cv::Mat PO = p3Dw - Ow;
        double const dist3D = cv::norm(PO);

        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if (PO.dot(Pn) < 0.5 * dist3D)
            continue;

        // Compute predicted scale level
        int const nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        float const radius = th * pKF->mvScaleFactors[nPredictedLevel];

        std::vector<size_t> const& vIndices = pKF->GetFeaturesInArea(u, v, radius);
        if (vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        cv::Mat const& dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;

        for (size_t const idx : vIndices)
        {
            int const kpLevel = pKF->mvKeysUn[idx].octave;

            if (kpLevel < nPredictedLevel - 1 || kpLevel > nPredictedLevel)
                continue;

            cv::Mat const& dKF = pKF->mDescriptors.row(idx);

            int const dist = DescriptorDistance(dMP, dKF);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist <= TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);

            if (pMPinKF == nullptr)
            {
                pMP->AddObservation(pKF, bestIdx);
                pKF->AddMapPoint(pMP, bestIdx);
            }
            else if (!pMPinKF->isBad())
                vpReplacePoint.emplace(pMP, pMPinKF);
        }
    }
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        
        if (pMP == nullptr)
            continue;

        vbAlreadyMatched1[i] = true;
        int idx2 = pMP->GetIndexInKeyFrame(pKF2);
        if (idx2 >= 0 && idx2 < N2)
            vbAlreadyMatched2[idx2] = true;
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            cv::KeyPoint const& kp = pKF2->mvKeysUn[idx];

            if (kp.octave < nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}

std::size_t ORBmatcher::SearchByProjection(Frame& rCF, Frame const& rLF, float th, bool bMono)
{
    std::size_t nmatches = 0;

    std::vector<int> rotHist[HISTO_LENGTH];
    if (m_check_orientation)
    {
        for (int i = 0; i < HISTO_LENGTH; ++i)
            rotHist[i].reserve(500);
    }

    cv::Mat const Rcw = rCF.mTcw.rowRange(0, 3).colRange(0, 3);
    cv::Mat const tcw = rCF.mTcw.rowRange(0, 3).col(3);

    bool bForward;
    bool bBackward;

    if (!bMono)
    {
        cv::Mat const Rlw = rLF.mTcw.rowRange(0, 3).colRange(0, 3);
        cv::Mat const tlw = rLF.mTcw.rowRange(0, 3).col(3);

        cv::Mat const twc = -Rcw.t() * tcw;
        cv::Mat const tlc = Rlw * twc + tlw;

        bForward = tlc.at<float>(2) > rCF.get_mb();
        bBackward = -tlc.at<float>(2) > rCF.get_mb();
    }
    else
    {
        bForward = false;
        bBackward = false;
    }

    for (std::size_t i = 0; i < rLF.get_frame_N(); ++i)
    {
        MapPoint* pMP = rLF.mvpMapPoints[i];

        if (pMP == nullptr || rLF.mvbOutlier[i] || pMP->isBad())
            continue;

        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        float const invzc = 1.f / x3Dc.at<float>(2);
        if (invzc < 0)
            continue;

        float const xc = x3Dc.at<float>(0);
        float const yc = x3Dc.at<float>(1);

        float u = rCF.fx * xc * invzc + rCF.cx;
        float v = rCF.fy * yc * invzc + rCF.cy;

        if (u < rCF.mnMinX || u > rCF.mnMaxX ||
            v < rCF.mnMinY || v > rCF.mnMaxY)
            continue;

        int const nLastOctave = rLF.get_key_points()[i].octave;

        // Search in a window. Size depends on scale
        float const radius = th * rCF.mvScaleFactors[nLastOctave];

        std::vector<std::size_t> vIndices2;

        if (bForward)
            vIndices2 = rCF.GetFeaturesInArea(u, v, radius, nLastOctave);
        else if (bBackward)
            vIndices2 = rCF.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
        else
            vIndices2 = rCF.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

        if (vIndices2.empty())
            continue;

        cv::Mat const dMP = pMP->GetDescriptor();

        int bestDist = TH_HIGH;
        std::size_t bestIdx2 = (std::size_t)-1;

        for (size_t const i2 : vIndices2)
        {
            MapPoint* pMP_i2 = rCF.mvpMapPoints[i2];

            if (pMP_i2 != nullptr && pMP_i2->Observations() != 0)
                continue;

            if (rCF.mvuRight[i2] > 0)
            {
                float const ur = u - rCF.get_mbf() * invzc;
                float const er = std::fabs(ur - rCF.mvuRight[i2]);
                if (er > radius)
                    continue;
            }

            cv::Mat const& d = rCF.get_descriptor(i2);

            int const dist = DescriptorDistance(dMP, d);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx2 = i2;
            }
        }

        if (bestDist >= TH_HIGH)
            continue;

        rCF.mvpMapPoints[bestIdx2] = pMP;
        ++nmatches;

        if (m_check_orientation)
        {
            float rot = rLF.mvKeysUn[i].angle - rCF.mvKeysUn[bestIdx2].angle;
            if (rot < 0.f)
                rot += 360.f;

            int bin = (int)(rot * (HISTO_LENGTH / 360.f));

            if (bin == HISTO_LENGTH)
                bin = 0;

            assert(bin >= 0 && bin < HISTO_LENGTH);
            rotHist[bin].push_back(bestIdx2);
        }
    }

    // Apply rotation consistency
    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int> const& histo_entry = rotHist[i];

            for (int histo_index : histo_entry)
                rCF.mvpMapPoints[histo_index] = nullptr;

            nmatches -= histo_entry.size();
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF,
    std::unordered_set<MapPoint*> const& sAlreadyFound, float th , int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    std::vector<int> rotHist[HISTO_LENGTH];
    if (m_check_orientation)
    {
        for (int i = 0; i < HISTO_LENGTH; ++i)
            rotHist[i].reserve(500);
    }

    std::vector<MapPoint*> const& vpMPs = pKF->GetMapPointMatches();

    for (size_t i = 0, iend = vpMPs.size(); i < iend; ++i)
    {
        MapPoint* pMP = vpMPs[i];

        if (pMP == nullptr|| pMP->isBad() || sAlreadyFound.count(pMP) != 0)
            continue;

        // Project
        cv::Mat x3Dw = pMP->GetWorldPos();
        cv::Mat x3Dc = Rcw * x3Dw + tcw;

        const float xc = x3Dc.at<float>(0);
        const float yc = x3Dc.at<float>(1);
        const float invzc = 1.f / x3Dc.at<float>(2);

        const float u = CurrentFrame.fx * xc * invzc + CurrentFrame.cx;
        const float v = CurrentFrame.fy * yc * invzc + CurrentFrame.cy;

        if (u < CurrentFrame.mnMinX || u > CurrentFrame.mnMaxX ||
            v < CurrentFrame.mnMinY || v > CurrentFrame.mnMaxY)
            continue;

        // Compute predicted scale level
        cv::Mat PO = x3Dw-Ow;
        float dist3D = (float)cv::norm(PO);

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();

        // Depth must be inside the scale pyramid of the image
        if (dist3D < minDistance || dist3D > maxDistance)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

        // Search in a window
        const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);
        if (vIndices2.empty())
            continue;

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx2 = -1;

        for (std::size_t const i2 : vIndices2)
        {
            if (CurrentFrame.mvpMapPoints[i2])
                continue;

            cv::Mat const& d = CurrentFrame.get_descriptor((int)i2);

            int const dist = DescriptorDistance(dMP, d);
            if (dist < bestDist)
            {
                bestDist = dist;
                bestIdx2 = i2;
            }
        }

        if (bestDist <= ORBdist)
        {
            CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
            ++nmatches;

            if (m_check_orientation)
            {
                float rot = pKF->mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
                if (rot < 0.f)
                    rot += 360.0f;

                int bin = (int)(rot * (HISTO_LENGTH / 360.f));
                if (bin == HISTO_LENGTH)
                    bin = 0;

                assert(bin >= 0 && bin < HISTO_LENGTH);
                rotHist[bin].push_back(bestIdx2);
            }
        }
    }

    if (m_check_orientation)
    {
        int ind1 = -1;
        int ind2 = -1;
        int ind3 = -1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for (int i = 0; i < HISTO_LENGTH; ++i)
        {
            if (i == ind1 || i == ind2 || i == ind3)
                continue;

            std::vector<int> const& v_histo_entry = rotHist[i];

            for (int histo_index : v_histo_entry)
                CurrentFrame.mvpMapPoints[histo_index] = nullptr;

            nmatches -= (int)v_histo_entry.size();;
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(std::vector<int> const* histo, int L, int &ind1, int &ind2, int &ind3)
{
    std::size_t max1 = 0;
    std::size_t max2 = 0;
    std::size_t max3 = 0;

    for (int i = 0; i < L; ++i)
    {
        std::size_t const s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(cv::Mat const& a, cv::Mat const& b)
{
    uint32_t const* const pa = a.ptr<uint32_t>();
    uint32_t const* const pb = b.ptr<uint32_t>();

    int dist = 0;

    for (int i = 0; i < 8; ++i)
    {
        uint32_t v = pa[i] ^ pb[i];
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (int)((((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24);
    }

    return dist;
}

} //namespace ORB_SLAM
