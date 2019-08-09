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


#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <opencv2/opencv.hpp>

#include <unordered_set>
#include <vector>

namespace ORB_SLAM2
{

class Frame;
class KeyFrame;
class MapPoint;

class ORBmatcher
{
public:

    enum : int
    {
        TH_HIGH         = 100,
        TH_LOW          = 50,
        HISTO_LENGTH    = 30
    };

    ORBmatcher(float nnratio, bool check_orientation);

    // Computes the Hamming distance between two ORB descriptors
    static int DescriptorDistance(cv::Mat const& a, cv::Mat const& b);

    // Search matches between Frame keypoints and projected MapPoints. Returns number of matches
    // Used to track the local map (Tracking)
    int SearchByProjection(Frame& F, std::vector<MapPoint*> const& vpMapPoints, float th = 3);

    // Project MapPoints tracked in last frame into the current frame and search matches.
    // Used to track from previous frame (Tracking)
    int SearchByProjection(Frame& CurrentFrame, Frame const& LastFrame, float th, bool bMono);

    // Project MapPoints seen in KeyFrame into the Frame and search matches.
    // Used in relocalisation (Tracking)
    int SearchByProjection(Frame& CurrentFrame, KeyFrame* pKF,
        std::unordered_set<MapPoint*> const& sAlreadyFound, float th, int ORBdist);

    // Project MapPoints using a Similarity Transformation and search matches.
    // Used in loop detection (Loop Closing)
     int SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPoint*> &vpPoints, std::vector<MapPoint*> &vpMatched, int th);

    // Search matches between MapPoints in a KeyFrame and ORB in a Frame.
    // Brute force constrained to ORB that belong to the same vocabulary node (at a certain level)
    // Used in Relocalisation and Loop Detection
    std::size_t SearchByBoW(KeyFrame *pKF, Frame &F, std::vector<MapPoint*> &vpMapPointMatches);
    int SearchByBoW(KeyFrame *pKF1, KeyFrame* pKF2, std::vector<MapPoint*> &vpMatches12);

    // Matching for the Map Initialization (only used in the monocular case)
    int SearchForInitialization(Frame const& F1, Frame  const&F2,
        std::vector<cv::Point2f>& vbPrevMatched, std::vector<int>& vnMatches12, int windowSize = 10);

    // Matching to triangulate new MapPoints. Check Epipolar Constraint.
    std::size_t SearchForTriangulation(KeyFrame *pKF1, KeyFrame* pKF2, cv::Mat const& F12,
        std::vector<std::pair<size_t, size_t> >& vMatchedPairs, bool bOnlyStereo);

    // Search matches between MapPoints seen in KF1 and KF2 transforming by a Sim3 [s12*R12|t12]
    // In the stereo and RGB-D case, s12=1
    int SearchBySim3(KeyFrame* pKF1, KeyFrame* pKF2, std::vector<MapPoint *> &vpMatches12,
        const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th);

    // Project MapPoints into KeyFrame and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, std::unordered_set<MapPoint*> const& map_points, float th);

    // Project MapPoints into KeyFrame using a given Sim3 and search for duplicated MapPoints.
    int Fuse(KeyFrame* pKF, cv::Mat Scw, std::vector<MapPoint*> const& vpPoints,
        float th, std::vector<MapPoint*>& vpReplacePoint);

protected:

    bool CheckDistEpipolarLine(cv::KeyPoint const& kp1, cv::KeyPoint const& kp2,
        cv::Mat const& F12, KeyFrame const* pKF);

    static float RadiusByViewingCos(float viewCos)
        { return viewCos > 0.998f ? 2.5f : 4.0f; }

    void ComputeThreeMaxima(std::vector<int> const* histo, int L, int &ind1, int &ind2, int &ind3);

    float   mfNNratio;
    bool    m_check_orientation;
};

}// namespace ORB_SLAM

#endif // ORBMATCHER_H
