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

#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H

#include <vector>
#include <list>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace ORB_SLAM2
{

class ORBextractor
{
public:
    
    enum
    {
        HARRIS_SCORE    = 0,
        FAST_SCORE      = 1
    };

    enum : int
    {
        PATCH_SIZE      = 31,
        HALF_PATCH_SIZE = 15,
        EDGE_THRESHOLD  = 19
    };

    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    void operator () (cv::InputArray image, cv::InputArray mask,
        std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    int GetLevels() const
        { return nlevels; }

    float GetScaleFactor() const
        { return m_scaleFactor; }

    // TODO: PAE: to be removed
    std::vector<float> GetScaleFactors() const
    {
        std::vector<float> result(nlevels);
        result[0] = 1.f;
        for (int level = 1; level < nlevels; ++level)
            result[level] = result[level - 1] * m_scaleFactor;
        return result;
    }

    // TODO: PAE: to be removed
    std::vector<float> GetInverseScaleFactors() const
    {
        std::vector<float> result = GetScaleFactors();
        for (float& sf : result)
            sf = 1.f / sf;
        return result;
    }

    std::vector<float> GetScaleSigmaSquares()
    {
        std::vector<float> result = GetScaleFactors();
        for (float& sf : result)
            sf *= sf;
        return result;
    }

    std::vector<float> GetInverseScaleSigmaSquares()
    {
        std::vector<float> result = GetScaleFactors();
        for (float& sf : result)
            sf = 1.f / (sf * sf);
        return result;
    }

    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    

    static  std::vector<cv::KeyPoint> DistributeOctTree(
        std::vector<cv::KeyPoint>& vToDistributeKeys,
        int minX, int maxX, int minY, int maxY, int nFeatures);

    int                     nfeatures;
    float                   m_scaleFactor;
    int                     nlevels;
    int                     iniThFAST;
    int                     minThFAST;
    std::vector<int>        mnFeaturesPerLevel;
    std::vector<float>      mvScaleFactor;
    std::vector<float>      mvLevelSigma2;
};

} //namespace ORB_SLAM

#endif

