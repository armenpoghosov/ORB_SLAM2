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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"

#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"

#include <future>

namespace ORB_SLAM2
{

uint64_t Frame::s_next_id = 0;

bool Frame::mbInitialComputations = true;

float Frame::cx;
float Frame::cy;
float Frame::fx;
float Frame::fy;
float Frame::invfx;
float Frame::invfy;
float Frame::mnMinX;
float Frame::mnMinY;
float Frame::mnMaxX;
float Frame::mnMaxY;
float Frame::mfGridElementWidthInv;
float Frame::mfGridElementHeightInv;

Frame::Frame(Frame const& frame)
    :
    mpORBvocabulary(frame.mpORBvocabulary),
    mpORBextractorLeft(frame.mpORBextractorLeft),
    mpORBextractorRight(frame.mpORBextractorRight),
    mTimeStamp(frame.mTimeStamp),
    mK(frame.mK.clone()),
    mDistCoef(frame.mDistCoef.clone()),
    mbf(frame.mbf),
    mb(frame.mb),
    mThDepth(frame.mThDepth),
    m_frame_N(frame.m_frame_N),
    mvKeys(frame.mvKeys),
    mvKeysRight(frame.mvKeysRight),
    mvKeysUn(frame.mvKeysUn),
    mvuRight(frame.mvuRight),
    mvDepth(frame.mvDepth),
    mBowVec(frame.mBowVec),
    mFeatVec(frame.mFeatVec),
    mDescriptors(frame.mDescriptors.clone()),
    mDescriptorsRight(frame.mDescriptorsRight.clone()),
    mvpMapPoints(frame.mvpMapPoints),
    mvbOutlier(frame.mvbOutlier),
    m_id(frame.m_id),
    mpReferenceKF(frame.mpReferenceKF),
    mnScaleLevels(frame.mnScaleLevels),
    mfScaleFactor(frame.mfScaleFactor),
    mfLogScaleFactor(frame.mfLogScaleFactor),
    mvScaleFactors(frame.mvScaleFactors),
    mvInvScaleFactors(frame.mvInvScaleFactors),
    mvLevelSigma2(frame.mvLevelSigma2),
    mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for (int i = 0; i < FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j] = frame.mGrid[i][j];

    if (!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

Frame::Frame(cv::Mat const& imLeft, cv::Mat const& imRight,
        double timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight,
        ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, float bf, float thDepth)
    :
    mpORBvocabulary(voc),
    mpORBextractorLeft(extractorLeft),
    mpORBextractorRight(extractorRight),
    mTimeStamp(timeStamp),
    mK(K.clone()),
    mDistCoef(distCoef.clone()),
    mbf(bf),
    mThDepth(thDepth),
    mpReferenceKF(nullptr)
{
    // Frame ID
    m_id = s_next_id++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    auto future = std::async(&Frame::ExtractORB, this, 0, imLeft);
    ExtractORB(1, imRight);
    future.get();

    m_frame_N = mvKeys.size();

    if (mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(m_frame_N);
    mvbOutlier = vector<bool>(m_frame_N);

    // This is done only for the first Frame (or after a change in the calibration)
    ensure_initial_computations(imLeft, K);

    mb = mbf / fx;

    AssignFeaturesToGrid();
}

Frame::Frame(cv::Mat const& imGray, cv::Mat const& imDepth, double timeStamp,
        ORBextractor* extractor, ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef,
        float bf, float thDepth)
    :
    mpORBvocabulary(voc),
    mpORBextractorLeft(extractor),
    mpORBextractorRight(nullptr),
    mTimeStamp(timeStamp),
    mK(K.clone()),
    mDistCoef(distCoef.clone()),
    mbf(bf),
    mThDepth(thDepth)
{
    // Frame ID
    m_id = s_next_id++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0, imGray);

    m_frame_N = mvKeys.size();

    if (mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(m_frame_N);
    mvbOutlier = vector<bool>(m_frame_N);

    // This is done only for the first Frame (or after a change in the calibration)
    ensure_initial_computations(imGray, K);

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(cv::Mat const& imGray, double timeStamp, ORBextractor* extractor,
        ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, float bf, float thDepth)
    :
    mpORBvocabulary(voc),
    mpORBextractorLeft(extractor),
    mpORBextractorRight(nullptr),
    mTimeStamp(timeStamp),
    mK(K.clone()),
    mDistCoef(distCoef.clone()),
    mbf(bf),
    mThDepth(thDepth)
{
    // Frame ID
    m_id = s_next_id++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    m_imLeft = imGray; // PAE: added to make it faster
    ExtractORB(0, imGray);

    m_frame_N = mvKeys.size();

    if (mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = std::vector<float>(m_frame_N, -1.f);
    mvDepth = std::vector<float>(m_frame_N, -1.f);

    mvpMapPoints = std::vector<MapPoint*>(m_frame_N);
    mvbOutlier = std::vector<bool>(m_frame_N);

    // This is done only for the first Frame (or after a change in the calibration)
    ensure_initial_computations(imGray, K);

    mb = mbf / fx;

    AssignFeaturesToGrid();
}

void Frame::ensure_initial_computations(cv::Mat const& img_first, cv::Mat const& K)
{
    if (mbInitialComputations)
    {
        ComputeImageBounds(img_first);

        mfGridElementWidthInv = FRAME_GRID_COLS / (mnMaxX - mnMinX);
        mfGridElementHeightInv = FRAME_GRID_ROWS / (mnMaxY - mnMinY);

        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);

        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);

        invfx = 1.f / fx;
        invfy = 1.f / fy;

        mbInitialComputations = false;
    }
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f * m_frame_N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);

    for (int i = 0; i < FRAME_GRID_COLS; ++i)
        for (int j = 0; j < FRAME_GRID_ROWS; ++j)
            mGrid[i][j].reserve(nReserve);

    for (std::size_t i = 0; i < m_frame_N; ++i)
    {
        cv::KeyPoint const& kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if (PosInGrid(kp, nGridPosX, nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, cv::Mat const& im)
{
    if (flag == 0)
        (*mpORBextractorLeft)(im, mvKeys, mDescriptors);
    else
        (*mpORBextractorRight)(im, mvKeysRight, mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0, 3).col(3);
    mOw = -mRwc * mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) const
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    cv::Mat const Pc = mRcw * P + mtcw;
    float const PcX = Pc.at<float>(0);
    float const PcY= Pc.at<float>(1);
    float const PcZ = Pc.at<float>(2);

    // Check positive depth
    if (PcZ < 0.0f)
        return false;

    // Project in image and check it is not outside
    float const invz = 1.0f / PcZ;
    float const u = fx * PcX * invz + cx;
    float const v = fy * PcY * invz + cy;

    if (u < mnMinX || u > mnMaxX || v < mnMinY || v > mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    float const maxDistance = pMP->GetMaxDistanceInvariance();
    float const minDistance = pMP->GetMinDistanceInvariance();
    cv::Mat const PO = P-mOw;
    float const dist = cv::norm(PO);

    if (dist < minDistance || dist > maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if (viewCos < viewingCosLimit)
        return false;

    // Predict scale in the image
    int const nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf * invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

std::vector<size_t> Frame::GetFeaturesInArea(float x, float y, float r, int minLevel, int maxLevel) const
{
    std::vector<size_t> vIndices;

    int const nMinCellX = (std::max)(0, (int)floor((x - mnMinX - r) * mfGridElementWidthInv));
    if (nMinCellX >= FRAME_GRID_COLS)
        return vIndices;

    int const nMaxCellX = (std::min)(FRAME_GRID_COLS - 1, (int)ceil((x - mnMinX + r) * mfGridElementWidthInv));
    if (nMaxCellX < 0)
        return vIndices;

    int const nMinCellY = (std::max)(0, (int)floor((y - mnMinY - r) * mfGridElementHeightInv));
    if (nMinCellY >= FRAME_GRID_ROWS)
        return vIndices;

    int const nMaxCellY = (std::min)(FRAME_GRID_ROWS - 1, (int)ceil((y - mnMinY + r) * mfGridElementHeightInv));
    if (nMaxCellY < 0)
        return vIndices;

    vIndices.reserve(m_frame_N);

    for (int ix = nMinCellX; ix <= nMaxCellX; ++ix)
    {
        for (int iy = nMinCellY; iy <= nMaxCellY; ++iy)
        {
            std::vector<size_t> const& vCell = mGrid[ix][iy];

            if (vCell.empty())
                continue;

            for (size_t const key_index : vCell)
            {
                cv::KeyPoint const& kpUn = mvKeysUn[key_index];

                if ((minLevel > 0 && kpUn.octave < minLevel) ||
                    (maxLevel >= 0 && kpUn.octave > maxLevel))
                    continue;

                if (std::fabs(kpUn.pt.x - x) < r && std::fabs(kpUn.pt.y - y) < r)
                    vIndices.push_back(key_index);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(cv::KeyPoint const& kp, int& posX, int& posY) const
{
    posX = (int)std::round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
    posY = (int)std::round((kp.pt.y - mnMinY) * mfGridElementHeightInv);
    // Keypoint's coordinates are undistorted, which could cause to go out of the image
    return posX >= 0 && posX < FRAME_GRID_COLS && posY >= 0 && posY < FRAME_GRID_ROWS;
}

void Frame::ComputeBoW()
{
    if (mBowVec.empty())
    {
        std::vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
    }
}

void Frame::UndistortKeyPoints()
{
    if (mDistCoef.at<float>(0) == 0.f)
    {
        mvKeysUn = mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(m_frame_N, 2, CV_32F);
    for(int i = 0; i < m_frame_N; ++i)
    {
        mat.at<float>(i, 0) = mvKeys[i].pt.x;
        mat.at<float>(i, 1) = mvKeys[i].pt.y;
    }

    // Undistort points
    mat = mat.reshape(2);
    cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
    mat = mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(m_frame_N);
    for (int i = 0; i < m_frame_N; ++i)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x = mat.at<float>(i, 0);
        kp.pt.y = mat.at<float>(i, 1);
        mvKeysUn[i] = kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if (mDistCoef.at<float>(0) != 0.)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.;
        mat.at<float>(0, 1) = 0.;
        mat.at<float>(1, 0) = imLeft.cols;
        mat.at<float>(1, 1) = 0.;
        mat.at<float>(2, 0) = 0.;
        mat.at<float>(2, 1) = imLeft.rows;
        mat.at<float>(3, 0) = imLeft.cols;
        mat.at<float>(3, 1) = imLeft.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        mnMinX = (std::min)(mat.at<float>(0, 0), mat.at<float>(2, 0));
        mnMaxX = (std::max)(mat.at<float>(1, 0), mat.at<float>(3, 0));
        mnMinY = (std::min)(mat.at<float>(0, 1), mat.at<float>(1, 1));
        mnMaxY = (std::max)(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        mnMinX = 0.f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvDepth = vector<float>(m_frame_N, -1.0f);
    mvuRight = vector<float>(m_frame_N, -1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(m_frame_N);

    for(int iL=0; iL<m_frame_N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        cv::Mat const& dL = get_descriptor(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w, w) * cv::Mat::ones(IL.rows, IL.cols, CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(), vDistIdx.end());

    const float median = vDistIdx[vDistIdx.size() / 2].first;
    const float thDist = 1.5f * 1.4f * median;

    for (int i = vDistIdx.size() - 1; i >= 0 && vDistIdx[i].first >= thDist; --i)
    {
        mvuRight[vDistIdx[i].second] = -1;
        mvDepth[vDistIdx[i].second] = -1;
    }
}

void Frame::ComputeStereoFromRGBD(cv::Mat const& imDepth)
{
    mvuRight = std::vector<float>(m_frame_N, -1.f);
    mvDepth = std::vector<float>(m_frame_N, -1.f);

    for (int i = 0; i < m_frame_N; ++i)
    {
        cv::KeyPoint const& kp = mvKeys[i];
        cv::KeyPoint const& kpU = mvKeysUn[i];

        float const d = imDepth.at<float>((int)kp.pt.y, (int)kp.pt.x);
        if (d > 0.)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x - mbf / d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(std::size_t kp_index) const
{
    float const z = mvDepth[kp_index];
    if (z <= 0)
        return cv::Mat();

    cv::KeyPoint const kp = mvKeysUn[kp_index];
    float const x = (kp.pt.x - cx) * z * invfx;
    float const y = (kp.pt.y - cy) * z * invfy;

    cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
    return mRwc * x3Dc + mOw;
}

} //namespace ORB_SLAM
