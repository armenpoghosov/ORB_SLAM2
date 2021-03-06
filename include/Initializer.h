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
#ifndef INITIALIZER_H
#define INITIALIZER_H

#include<opencv2/opencv.hpp>

#include "Frame.h"

namespace ORB_SLAM2
{

// THIS IS THE INITIALIZER FOR MONOCULAR SLAM. NOT USED IN THE STEREO OR RGBD CASE.
class Initializer
{
public:

    // Fix the reference frame
    Initializer(Frame const& ReferenceFrame, float sigma = 1.f, int iterations = 200);

    // Computes in parallel a fundamental matrix and a homography
    // Selects a model and tries to recover the motion and the structure from motion
    bool Initialize(Frame const& CurrentFrame, vector<int> const& vMatches12,
        cv::Mat& R21, cv::Mat& t21, vector<cv::Point3f>& vP3D, vector<bool>& vbTriangulated);

private:

    void FindHomography(vector<bool>& vbMatchesInliers, float& score, cv::Mat& H21);
    void FindFundamental(vector<bool>& vbInliers, float& score, cv::Mat& F21);

    // PAE: NOTE: homography computation was done previously with 8 point!? why?
    static cv::Mat ComputeH21(cv::Point2f const (&vP1)[4], cv::Point2f const (&vP2)[4]);
    static cv::Mat ComputeF21(cv::Point2f const (&vP1)[8], cv::Point2f const (&vP2)[8]);

    float CheckHomography(cv::Mat const& H21, cv::Mat const& H12,
        vector<bool>& vbMatchesInliers, float sigma);
    float CheckFundamental(cv::Mat const& F21,
        vector<bool>& vbMatchesInliers, float sigma);

    bool ReconstructF(std::vector<bool>& vbMatchesInliers, cv::Mat& F21, cv::Mat const& K,
        cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3f>& vP3D, vector<bool>& vbTriangulated,
        float minParallax, int minTriangulated);

    bool ReconstructH(std::vector<bool>& vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
        cv::Mat& R21, cv::Mat& t21, std::vector<cv::Point3f>& vP3D,
        std::vector<bool>& vbTriangulated, float minParallax, int minTriangulated);

    static void Triangulate(cv::KeyPoint const& kp1, cv::KeyPoint const& kp2,
        cv::Mat const& P1, cv::Mat const& P2, cv::Mat& x3D);

    void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);

    int CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<pair<int, int> > &vMatches12, vector<bool> &vbInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax);

    void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);


    // Keypoints from Reference Frame (Frame 1)
    std::vector<cv::KeyPoint>       mvKeys1;

    // Keypoints from Current Frame (Frame 2)
    std::vector<cv::KeyPoint>       mvKeys2;

    // Current Matches from Reference to Current
    std::vector<pair<int, int> >    mvMatches12;
    std::vector<bool>               mvbMatched1;

    // Calibration
    cv::Mat                         mK;

    // Standard Deviation and Variance
    float                           mSigma;
    float                           mSigma2;

    // Ransac max iterations
    int                             mMaxIterations;

    // Ransac sets
    std::vector<size_t[8]>          mvSets;
};

} //namespace ORB_SLAM

#endif // INITIALIZER_H
