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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include <future>

namespace ORB_SLAM2
{

Initializer::Initializer(Frame const& rRF, float sigma, int iterations)
{
    mK = rRF.get_K().clone();

    mvKeys1 = rRF.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma * sigma;

    mMaxIterations = iterations;
}

bool Initializer::Initialize(Frame const& rCF, std::vector<int> const& vMatches12,
    cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3f>& vP3D, std::vector<bool>& vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2
    mvKeys2 = rCF.mvKeysUn;

    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());

    mvbMatched1.resize(mvKeys1.size());

    for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        bool const matched = vMatches12[i] >= 0;

        mvbMatched1[i] = matched;

        if (matched)
            mvMatches12.emplace_back((int)i, vMatches12[i]);
    }

    std::size_t const N = mvMatches12.size();

    // Generate sets of 8 points for each RANSAC iteration
    mvSets = std::vector<std::size_t[8]>(mMaxIterations);

    DUtils::Random::SeedRandOnce(0);

    for (int it = 0; it < mMaxIterations; ++it)
    {
        std::size_t(&index_set)[8] = mvSets[it];

        std::size_t* const it_s = &index_set[0];
        std::size_t* const it_e = it_s + 8;

        for (std::size_t* it = it_s; it != it_e; ++it)
        {
            std::size_t index;

            for (index = DUtils::Random::RandomInt(0, N - 1);
                std::find(it_s, it, index) != it;
                index = DUtils::Random::RandomInt(0, N - 1))
            {}

            *it = index;
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> vbMatchesInliersH;
    vector<bool> vbMatchesInliersF;

    float SH;
    float SF;

    cv::Mat H;
    cv::Mat F;

    auto future = std::async(launch::async, &Initializer::FindHomography, this, ref(vbMatchesInliersH), ref(SH), ref(H));
    FindFundamental(vbMatchesInliersF, SF, F);
    future.wait(); // Wait until both have finished

    // Compute ratio of scores
    float RH = SH / (SH + SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if (RH > 0.40)
        return ReconstructH(vbMatchesInliersH, H, mK, R21, t21, vP3D, vbTriangulated, 1.f, 10);
    else //if (pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF, F, mK, R21, t21, vP3D, vbTriangulated, 1.f, 10);

    // return false;
}

void Initializer::FindHomography(vector<bool>& vbMatchesInliers, float& score, cv::Mat& H21)
{
    // Normalize coordinates
    cv::Mat T1;
    std::vector<cv::Point2f> vPn1;
    Normalize(mvKeys1, vPn1, T1);

    cv::Mat T2;
    std::vector<cv::Point2f> vPn2;
    Normalize(mvKeys2, vPn2, T2);

    cv::Mat T2inv = T2.inv();

    // Iteration variables
    cv::Point2f vPn1i[4];
    cv::Point2f vPn2i[4];

    cv::Mat H21i;
    cv::Mat H12i;

    // Number of putative matches
    std::size_t const N = mvMatches12.size();

    vector<bool> vbCurrentInliers(N);

    // Best Results variables
    score = 0.f;
    vbMatchesInliers = vector<bool>(N);

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < mMaxIterations; ++it)
    {
        std::size_t const (&index_set)[8] = mvSets[it];

        // Select a minimum set
        for (size_t j = 0; j < 4; ++j)
        {
            std::pair<int, int> const& match = mvMatches12[index_set[j]];
            vPn1i[j] = vPn1[match.first];
            vPn2i[j] = vPn2[match.second];
        }

        cv::Mat Hn = ComputeH21(vPn1i, vPn2i);
        H21i = T2inv * Hn * T1;
        H12i = H21i.inv();

        float const currentScore = CheckHomography(H21i, H12i, vbCurrentInliers, mSigma);
        if (currentScore > score)
        {
            H21 = H21i.clone();
            vbMatchesInliers = std::move(vbCurrentInliers);
            score = currentScore;
        }
    }
}


void Initializer::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches
    std::size_t const N = vbMatchesInliers.size();

    // Normalize coordinates
    cv::Mat T1;
    vector<cv::Point2f> vPn1;
    Normalize(mvKeys1, vPn1, T1);

    cv::Mat T2;
    vector<cv::Point2f> vPn2;
    Normalize(mvKeys2, vPn2, T2);

    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N);

    // Iteration variables
    cv::Point2f vPn1i[8];
    cv::Point2f vPn2i[8];
    
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N);

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < mMaxIterations; ++it)
    {
        std::size_t const (&index_set)[8] = mvSets[it];

        // Select a minimum set
        for (int j = 0; j < 8; ++j)
        {
            auto const& pair_match = mvMatches12[index_set[j]];
            vPn1i[j] = vPn1[pair_match.first];
            vPn2i[j] = vPn2[pair_match.second];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        F21i = T2t * Fn * T1;

        float const currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);
        if (currentScore > score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

cv::Mat Initializer::ComputeH21(cv::Point2f const (&vP1)[4], cv::Point2f const (&vP2)[4])
{
    cv::Mat A(18, 9, CV_32F);

    for (int i = 0; i < 4; ++i) // NOTE: PAE: it was previously computed with 8 points!? why?
    {
        float const u1 = vP1[i].x;
        float const v1 = vP1[i].y;

        float const u2 = vP2[i].x;
        float const v2 = vP2[i].y;

        A.at<float>(2 * i, 0) = 0.f;
        A.at<float>(2 * i, 1) = 0.f;
        A.at<float>(2 * i, 2) = 0.f;
        A.at<float>(2 * i, 3) = - u1;
        A.at<float>(2 * i, 4) = - v1;
        A.at<float>(2 * i, 5) = - 1.f;
        A.at<float>(2 * i, 6) = v2 * u1;
        A.at<float>(2 * i, 7) = v2 * v1;
        A.at<float>(2 * i, 8) = v2;

        A.at<float>(2 * i + 1,0) = u1;
        A.at<float>(2 * i + 1,1) = v1;
        A.at<float>(2 * i + 1,2) = 1.f;
        A.at<float>(2 * i + 1,3) = 0.f;
        A.at<float>(2 * i + 1,4) = 0.f;
        A.at<float>(2 * i + 1,5) = 0.f;
        A.at<float>(2 * i + 1,6) = -u2 * u1;
        A.at<float>(2 * i + 1,7) = -u2 * v1;
        A.at<float>(2 * i + 1,8) = -u2;
    }

    cv::Mat u;
    cv::Mat w;
    cv::Mat vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat Initializer::ComputeF21(cv::Point2f const (&vP1)[8], cv::Point2f const (&vP2)[8])
{
    cv::Mat A(8, 9, CV_32F);

    for (int i = 0; i < 8; ++i)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;

        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i, 0) = u2 * u1;
        A.at<float>(i, 1) = u2 * v1;
        A.at<float>(i, 2) = u2;
        A.at<float>(i, 3) = v2 * u1;
        A.at<float>(i, 4) = v2 * v1;
        A.at<float>(i, 5) = v2;
        A.at<float>(i, 6) = u1;
        A.at<float>(i, 7) = v1;
        A.at<float>(i, 8) = 1;
    }

    cv::Mat u;
    cv::Mat w;
    cv::Mat vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2) = 0;

    return  u * cv::Mat::diag(w) * vt;
}

float Initializer::CheckHomography(cv::Mat const& H21, cv::Mat const& H12, vector<bool>& vbMatchesInliers, float sigma)
{
    std::size_t const N = mvMatches12.size();

    float const h11 = H21.at<float>(0, 0);
    float const h12 = H21.at<float>(0, 1);
    float const h13 = H21.at<float>(0, 2);
    float const h21 = H21.at<float>(1, 0);
    float const h22 = H21.at<float>(1, 1);
    float const h23 = H21.at<float>(1, 2);
    float const h31 = H21.at<float>(2, 0);
    float const h32 = H21.at<float>(2, 1);
    float const h33 = H21.at<float>(2, 2);

    float const h11inv = H12.at<float>(0, 0);
    float const h12inv = H12.at<float>(0, 1);
    float const h13inv = H12.at<float>(0, 2);
    float const h21inv = H12.at<float>(1, 0);
    float const h22inv = H12.at<float>(1, 1);
    float const h23inv = H12.at<float>(1, 2);
    float const h31inv = H12.at<float>(2, 0);
    float const h32inv = H12.at<float>(2, 1);
    float const h33inv = H12.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0;

    float const th = 5.991f;

    float const invSigmaSquare = 1.f / (sigma * sigma);

    for (std::size_t i = 0; i < N; ++i)
    {
        bool bIn = true;

        auto const& match_pair = mvMatches12[i];

        cv::KeyPoint const& kp1 = mvKeys1[match_pair.first];
        cv::KeyPoint const& kp2 = mvKeys2[match_pair.second];

        float const u1 = kp1.pt.x;
        float const v1 = kp1.pt.y;

        float const u2 = kp2.pt.x;
        float const v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        float const w2in1inv = 1.f / (h31inv * u2 + h32inv * v2 + h33inv);
        float const u2in1 = (h11inv * u2 + h12inv * v2 + h13inv) * w2in1inv;
        float const v2in1 = (h21inv * u2 + h22inv * v2 + h23inv) * w2in1inv;

        float const squareDist1 = (u1 - u2in1) * (u1 - u2in1) + (v1 - v2in1) * (v1 - v2in1);

        float const chiSquare1 = squareDist1 * invSigmaSquare;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        float const w1in2inv = 1.f / (h31 * u1 + h32 * v1 + h33);
        float const u1in2 = (h11 * u1 + h12 * v1 + h13) * w1in2inv;
        float const v1in2 = (h21 * u1 + h22 * v1 + h23) * w1in2inv;

        float const squareDist2 = (u2 - u1in2)*(u2 - u1in2) + (v2 - v1in2)*(v2 - v1in2);

        float const chiSquare2 = squareDist2*invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        vbMatchesInliers[i] = bIn;
    }

    return score;
}

float Initializer::CheckFundamental(cv::Mat const& F21, std::vector<bool>& vbMatchesInliers, float sigma)
{
    std::size_t const N = mvMatches12.size();

    float const f11 = F21.at<float>(0, 0);
    float const f12 = F21.at<float>(0, 1);
    float const f13 = F21.at<float>(0, 2);
    float const f21 = F21.at<float>(1, 0);
    float const f22 = F21.at<float>(1, 1);
    float const f23 = F21.at<float>(1, 2);
    float const f31 = F21.at<float>(2, 0);
    float const f32 = F21.at<float>(2, 1);
    float const f33 = F21.at<float>(2, 2);

    vbMatchesInliers.resize(N);

    float score = 0.f;

    float const th = 3.841f;
    float const thScore = 5.991f;

    float const invSigmaSquare = 1.f / (sigma * sigma);

    for (std::size_t i = 0; i < N; ++i)
    {
        auto const& match_pair = mvMatches12[i];

        cv::KeyPoint const& kp1 = mvKeys1[match_pair.first];
        cv::KeyPoint const& kp2 = mvKeys2[match_pair.second];

        float const u1 = kp1.pt.x;
        float const v1 = kp1.pt.y;

        float const u2 = kp2.pt.x;
        float const v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2 = F21x1 = (a2, b2, c2)
        float const a2 = f11 * u1 + f12 * v1 + f13;
        float const b2 = f21 * u1 + f22 * v1 + f23;
        float const c2 = f31 * u1 + f32 * v1 + f33;

        float const num2 = a2 * u2 + b2 * v2 + c2;

        float const squareDist1 = (num2 * num2) / (a2 * a2 + b2 * b2);

        float const chiSquare1 = squareDist1 * invSigmaSquare;

        bool bIn = true;

        if (chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 = x2tF21 = (a1, b1, c1)
        float const a1 = f11 * u2 + f21 * v2 + f31;
        float const b1 = f12 * u2 + f22 * v2 + f32;
        const float c1 = f13 * u2 + f23 * v2 + f33;

        const float num1 = a1 * u1 + b1 * v1 + c1;

        const float squareDist2 = (num1 * num1) / (a1 * a1 + b1 * b1);

        const float chiSquare2 = squareDist2 * invSigmaSquare;

        if (chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        vbMatchesInliers[i] = bIn;
    }

    return score;
}

bool Initializer::ReconstructF(std::vector<bool>& vbMatchesInliers, cv::Mat& F21, cv::Mat const& K,
    cv::Mat&R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated,
    float minParallax, int minTriangulated)
{
    int N = 0;
    
    for (bool const is_inlier : vbMatchesInliers)
        N += is_inlier ? 1 : 0;

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t() * F21 * K;

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21, R1, R2, t);  

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    std::vector<cv::Point3f> vP3D1;
    std::vector<cv::Point3f> vP3D2;
    std::vector<cv::Point3f> vP3D3;
    std::vector<cv::Point3f> vP3D4;

    std::vector<bool> vbTriangulated1;
    std::vector<bool> vbTriangulated2;
    std::vector<bool> vbTriangulated3;
    std::vector<bool> vbTriangulated4;

    float parallax1;
    float parallax2;
    float parallax3;
    float parallax4;

    int nGood1 = CheckRT(R1, t1, mvKeys1, mvKeys2, mvMatches12,
        vbMatchesInliers, K, vP3D1, 4.f * mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, mvMatches12,
        vbMatchesInliers, K, vP3D2, 4.f * mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, mvMatches12,
        vbMatchesInliers, K, vP3D3, 4.f * mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, mvMatches12,
        vbMatchesInliers, K, vP3D4, 4.f * mSigma2, vbTriangulated4, parallax4);

    int maxGood = (std::max)(nGood1, (std::max)(nGood2, (std::max)(nGood3, nGood4)));

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = (std::max)((int)(0.9 * N), minTriangulated);

    int nsimilar = 0;
    if (nGood1 > 0.7 * maxGood)
        ++nsimilar;
    if (nGood2 > 0.7 * maxGood)
        ++nsimilar;
    if (nGood3 > 0.7 * maxGood)
        ++nsimilar;
    if (nGood4 > 0.7 * maxGood)
        ++nsimilar;

    // If there is not a clear winner or not enough triangulated points reject initialization
    if (maxGood < nMinGood || nsimilar > 1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if (maxGood == nGood1)
    {
        if (parallax1 > minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood2)
    {
        if (parallax2 > minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood3)
    {
        if (parallax3 > minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }
    else if (maxGood == nGood4)
    {
        if (parallax4 > minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

bool Initializer::ReconstructH(std::vector<bool>& vbMatchesInliers, cv::Mat& H21, cv::Mat& K,
    cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3f>& vP3D, std::vector<bool>& vbTriangulated,
    float minParallax, int minTriangulated)
{
    int N = 0;
    for (size_t i = 0, iend = vbMatchesInliers.size() ; i < iend; ++i)
        if (vbMatchesInliers[i])
            ++N;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK * H21 * K;

    cv::Mat U;
    cv::Mat w;
    cv::Mat Vt;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    cv::Mat V = Vt.t();

    float const d1 = w.at<float>(0);
    float const d2 = w.at<float>(1);
    float const d3 = w.at<float>(2);

    if (d1 / d2 < 1.00001f || d2 / d3 < 1.00001f)
    {
        return false;
    }

    float s = (float)(cv::determinant(U) * cv::determinant(Vt));

    std::vector<cv::Mat> vR;
    vR.reserve(8);

    std::vector<cv::Mat> vt;
    vt.reserve(8);

    std::vector<cv::Mat> vn;
    vn.reserve(8);

    // n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = std::sqrt((d1 * d1 - d2 * d2) /(d1 * d1 - d3 * d3));
    float aux3 = std::sqrt((d2 * d2 - d3 * d3) /(d1 * d1 - d3 * d3));
    float x1[] = { aux1, aux1, -aux1, -aux1 };
    float x3[] = { aux3, -aux3, aux3, -aux3 };

    // case d'=d2
    float aux_stheta = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 + d3) * d2);

    float ctheta = (d2 * d2 + d1 * d3) / ((d1 + d3) * d2);
    float stheta[] = { aux_stheta, -aux_stheta, -aux_stheta, aux_stheta };

    for (int i = 0; i < 4; ++i)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = ctheta;
        Rp.at<float>(0, 2) = -stheta[i];
        Rp.at<float>(2, 0) = stheta[i];
        Rp.at<float>(2, 2) = ctheta;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = -x3[i];
        tp *= d1 - d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;

        vn.push_back(n);
    }

    // case d'=-d2
    float aux_sphi = std::sqrt((d1 * d1 - d2 * d2) * (d2 * d2 - d3 * d3)) / ((d1 - d3) * d2);

    float cphi = (d1 * d3 - d2 * d2) / ((d1 - d3) * d2);
    float sphi[] = { aux_sphi, -aux_sphi, -aux_sphi, aux_sphi };

    for (int i = 0; i < 4; ++i)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = cphi;
        Rp.at<float>(0, 2) = sphi[i];
        Rp.at<float>(1, 1) = -1;
        Rp.at<float>(2, 0) = sphi[i];
        Rp.at<float>(2, 2) = -cphi;

        cv::Mat R = s * U * Rp * Vt;
        vR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = x3[i];
        tp *= d1 + d3;

        cv::Mat t = U * tp;
        vt.push_back(t / cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];

        cv::Mat n = V * np;
        if (n.at<float>(2) < 0)
            n = -n;

        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;
    std::size_t bestSolutionIdx = -1;
    float bestParallax = -1;
    std::vector<cv::Point3f> bestP3D;
    std::vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper
    // (which could fail for points seen with low parallax) reconstruct all hypotheses
    // and check in terms of triangulated points and parallax
    for (size_t i = 0; i < 8; ++i)
    {
        float parallaxi;
        std::vector<cv::Point3f> vP3Di;
        std::vector<bool> vbTriangulatedi;

        int nGood = CheckRT(vR[i], vt[i], mvKeys1, mvKeys2, mvMatches12, vbMatchesInliers,
            K, vP3Di, 4.f * mSigma2, vbTriangulatedi, parallaxi);

        if (nGood > bestGood)
        {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if (nGood > secondBestGood)
        {
            secondBestGood = nGood;
        }
    }

    if (secondBestGood < 0.75f * bestGood &&
        bestParallax >= minParallax && bestGood > minTriangulated && bestGood > 0.9f * N)
    {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        vP3D = bestP3D;
        vbTriangulated = bestTriangulated;
        return true;
    }

    return false;
}

void Initializer::Triangulate(cv::KeyPoint const& kp1, cv::KeyPoint const& kp2,
    cv::Mat const& P1, cv::Mat const& P2, cv::Mat& x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
    A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
    A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
    A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

    cv::Mat u;
    cv::Mat w;
    cv::Mat vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
}

void Initializer::Normalize(std::vector<cv::KeyPoint> const& vKeys,
    std::vector<cv::Point2f>& vNormalizedPoints, cv::Mat& transform)
{
    float meanX = 0.f;
    float meanY = 0.f;

    for (cv::KeyPoint const& kp : vKeys)
    {
        meanX += kp.pt.x;
        meanY += kp.pt.y;
    }

    std::size_t const keys_count = vKeys.size();

    meanX /= keys_count;
    meanY /= keys_count;

    float meanDevX = 0.f;
    float meanDevY = 0.f;

    vNormalizedPoints.reserve(keys_count);

    for (cv::KeyPoint const& kp : vKeys)
    {
        vNormalizedPoints.emplace_back(kp.pt.x - meanX, kp.pt.y - meanY);
        cv::Point2f& np = vNormalizedPoints.back();
        meanDevX += std::fabs(np.x);
        meanDevY += std::fabs(np.y);
    }

    meanDevX /= keys_count;
    meanDevY /= keys_count;

    float const sX = 1.f / meanDevX;
    float const sY = 1.f / meanDevY;

    for (cv::Point2f& np : vNormalizedPoints)
    {
        np.x *= sX;
        np.y *= sY;
    }

    transform = cv::Mat::eye(3, 3, CV_32F);
    transform.at<float>(0, 0) = sX;
    transform.at<float>(1, 1) = sY;
    transform.at<float>(0, 2) = - meanX * sX;
    transform.at<float>(1, 2) = - meanY * sY;
}


int Initializer::CheckRT(cv::Mat const& R, cv::Mat const& t,
    std::vector<cv::KeyPoint> const& vKeys1, std::vector<cv::KeyPoint> const& vKeys2,
    std::vector<pair<int, int> > const& vMatches12, std::vector<bool>& vbMatchesInliers,
    cv::Mat const& K, std::vector<cv::Point3f>& vP3D, float th2, std::vector<bool>& vbGood, float& parallax)
{
    // Calibration parameters
    float const fx = K.at<float>(0, 0);
    float const fy = K.at<float>(1, 1);
    float const cx = K.at<float>(0, 2);
    float const cy = K.at<float>(1, 2);

    std::size_t const keys1_size = vKeys1.size();

    vbGood = std::vector<bool>(keys1_size);
    vP3D.resize(keys1_size);

    vector<float> vCosParallax;
    vCosParallax.reserve(keys1_size);

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));
    K.copyTo(P1.rowRange(0, 3).colRange(0, 3));

    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t.copyTo(P2.rowRange(0, 3).col(3));
    P2 = K * P2;

    cv::Mat O2 = - R.t() * t;

    int nGood = 0;

    for (size_t i = 0, iend = vMatches12.size(); i < iend; ++i)
    {
        if (!vbMatchesInliers[i])
            continue;

        auto const& pair_match = vMatches12[i];

        cv::KeyPoint const& kp1 = vKeys1[pair_match.first];
        cv::KeyPoint const& kp2 = vKeys2[pair_match.second];

        cv::Mat p3dC1;

        Triangulate(kp1, kp2, P1, P2, p3dC1);

        if (!std::isfinite(p3dC1.at<float>(0)) ||
            !std::isfinite(p3dC1.at<float>(1)) ||
            !std::isfinite(p3dC1.at<float>(2)))
        {
            vbGood[pair_match.first] = false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        double const dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        double const dist2 = cv::norm(normal2);

        double const cosParallax = normal1.dot(normal2) / (dist1 * dist2);

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        if (p3dC1.at<float>(2) <= 0.f && cosParallax < 0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R * p3dC1 + t;

        if (p3dC2.at<float>(2) <= 0.f && cosParallax < 0.99998)
            continue;

        // Check reprojection error in first image
        float const invZ1 = 1.f / p3dC1.at<float>(2);
        float const im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
        float const im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;
        float const squareError1 = (im1x - kp1.pt.x) *(im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);
        if (squareError1 > th2)
            continue;

        // Check reprojection error in second image
        float const invZ2 = 1.f / p3dC2.at<float>(2);
        float const im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
        float const im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;
        float const squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);
        if (squareError2 > th2)
            continue;

        vCosParallax.push_back((float)cosParallax);
        vP3D[pair_match.first] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        ++nGood;

        if (cosParallax < 0.99998)
            vbGood[pair_match.first] = true;
    }

    if (nGood > 0)
    {
        std::sort(vCosParallax.begin(), vCosParallax.end());
        std::size_t idx = vCosParallax.empty() ? 50 : (std::min)(vCosParallax.size() - 1, (std::size_t)50);
        parallax = (float)(std::acos(vCosParallax[idx]) * 180. / CV_PI);
    }
    else
        parallax = 0.f;

    return nGood;
}

void Initializer::DecomposeE(cv::Mat const& E, cv::Mat& R1, cv::Mat& R2, cv::Mat& t)
{
    cv::Mat u;
    cv::Mat w;
    cv::Mat vt;
    cv::SVD::compute(E, w, u, vt);

    u.col(2).copyTo(t);
    t = t / cv::norm(t);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    R1 = u * W * vt;
    if (cv::determinant(R1) < 0)
        R1 = -R1;

    R2 = u * W.t() * vt;
    if (cv::determinant(R2) < 0)
        R2 = -R2;
}

} //namespace ORB_SLAM
