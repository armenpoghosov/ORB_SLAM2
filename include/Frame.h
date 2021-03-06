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

#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>

#include "Thirdparty/DBoW2/DBoW2/BowVector.h"
#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include "ORBVocabulary.h"

#include <cstdint>
#include <vector>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;
class ORBextractor;

class Frame
{
public:

    enum : int
    {
        FRAME_GRID_ROWS = 48,
        FRAME_GRID_COLS = 64
    };

    Frame() // TODO: PAE: may need complete removal
    {}

    // Copy constructor.
    Frame(Frame const& frame);

    // Constructor for stereo cameras.
    Frame(cv::Mat const& imLeft, cv::Mat const& imRight, double timeStamp,
        ORBextractor* extractorLeft, ORBextractor* extractorRight,
        ORBVocabulary* voc, cv::Mat& K, cv::Mat& distCoef, float bf, float thDepth);

    // Constructor for RGB-D cameras.
    Frame(cv::Mat const& imGray, cv::Mat const& imDepth,
        double timeStamp, ORBextractor* extractor, ORBVocabulary* voc,
        cv::Mat &K, cv::Mat& distCoef, float bf, float thDepth);

    // Constructor for Monocular cameras.
    Frame(cv::Mat const& imGray, double timeStamp, ORBextractor* extractor,
        ORBVocabulary* voc, cv::Mat &K, cv::Mat& distCoef, float bf, float thDepth);

    // Extract ORB on the image. 0 for left image and 1 for right image.
    void ExtractORB(int flag, cv::Mat const& im);

    // Compute Bag of Words representation.
    void ComputeBoW();

    // Set the camera pose.
    void SetPose(cv::Mat Tcw);

    // Computes rotation, translation and camera center matrices from the camera pose.
    void UpdatePoseMatrices();

    // Returns the camera center.
    cv::Mat GetCameraCenter() const
        { return mOw.clone(); }

    // Returns inverse of rotation
    cv::Mat GetRotationInverse() const
        { return mRwc.clone(); }

    // Check if a MapPoint is in the frustum of the camera
    // and fill variables of the MapPoint to be used by the tracking
    bool isInFrustum(MapPoint* pMP, float viewingCosLimit) const;

    // Compute the cell of a keypoint (return false if outside the grid)
    bool PosInGrid(cv::KeyPoint const& kp, int& posX, int& posY) const;

    std::vector<std::size_t> GetFeaturesInArea(float x, float y, float r,
        int minLevel = -1, int maxLevel = -1) const;

    // Search a match for each keypoint in the left image to a keypoint in the right image.
    // If there is a match, depth is computed and the right coordinate associated to the left keypoint is stored.
    void ComputeStereoMatches();

    // Associate a "right" coordinate to a keypoint if there is valid depth in the depthmap.
    void ComputeStereoFromRGBD(cv::Mat const& imDepth);

    // Backprojects a keypoint (if stereo/depth info available) into 3D world coordinates.
    cv::Mat UnprojectStereo(std::size_t kp_index) const;

    cv::Mat const& get_image_left() const
        { return m_imLeft;  }

    cv::Mat const& get_descriptors() const
        { return mDescriptors; }

    cv::Mat const get_descriptor(int index) const
        { return mDescriptors.row(index); }

    ORBVocabulary* get_orb_vocabulary() const
        { return mpORBvocabulary; }

    cv::Mat const& get_K() const
        { return mK; }

    double get_time_stamp() const
        { return mTimeStamp; }

    static void reset_id_counter()
        { s_next_id = 0; }

    uint64_t get_id() const
        { return m_id; }

    float get_mbf() const
        { return mbf; }

    float get_mb() const
        { return mb; }

    std::vector<cv::KeyPoint> const& get_key_points() const
        { return mvKeys; }

    std::size_t get_frame_N() const
        { return m_frame_N; }

    DBoW2::BowVector const& get_BoW() const
    {
        const_cast<Frame*>(this)->ensure_BoW_computed();
        return mBowVec;
    }

    DBoW2::FeatureVector const& get_BoW_features() const
    {
        const_cast<Frame*>(this)->ensure_BoW_computed();
        return mFeatVec;
    }

    std::vector<MapPoint*> const& get_map_points() const
        { return mvpMapPoints; }
    // TODO: PAE: this one is a bad idea!!!
    std::vector<MapPoint*>& get_map_points()
        { return mvpMapPoints; }



    void commit_replaced_map_points();

public:

    // NOTE: PAE: stupid function to be refactored ... just for the sake of reducing code size
    void ensure_initial_computations(cv::Mat const& img_first, cv::Mat const& K);

    static float                fx;
    static float                fy;
    static float                cx;
    static float                cy;
    static float                invfx;
    static float                invfy;

    // Keypoints are assigned to cells in a grid to reduce matching complexity when projecting MapPoints.
    static float                mfGridElementWidthInv;
    static float                mfGridElementHeightInv;

    // Undistorted Image Bounds (computed once).
    static float                mnMinX;
    static float                mnMaxX;
    static float                mnMinY;
    static float                mnMaxY;

    static bool                 mbInitialComputations;

    // Threshold close/far points. Close points are inserted from 1 view.
    // Far points are inserted as in the monocular case from 2 views.
    float                       mThDepth; // PAE: in fact it is always const

    // Number of KeyPoints.
    std::size_t                 m_frame_N;

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint>   mvKeysRight;
    std::vector<cv::KeyPoint>   mvKeysUn;

    // Corresponding stereo coordinate and depth for each keypoint.
    // "Monocular" keypoints have a negative value.
    std::vector<float>          mvuRight;
    std::vector<float>          mvDepth;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat                     mDescriptorsRight;

    // MapPoints associated to keypoints, nullptr if no association.
    std::vector<MapPoint*>      mvpMapPoints;

    // Flag to identify outlier associations.
    std::vector<bool>           mvbOutlier;

    std::vector<std::size_t>    mGrid[FRAME_GRID_COLS][FRAME_GRID_ROWS];

    // Camera pose.
    cv::Mat                     mTcw;

    // Reference Keyframe.
    KeyFrame*                   mpReferenceKF;

    // TODO: PAE: those have to die probably
    // Scale pyramid info.
    int                         mnScaleLevels;
    float                       mfScaleFactor;
    float                       mfLogScaleFactor;
    std::vector<float>          mvScaleFactors;
    std::vector<float>          mvInvScaleFactors;
    std::vector<float>          mvLevelSigma2;
    std::vector<float>          mvInvLevelSigma2;

private:


    // Compute Bag of Words representation.
    void ensure_BoW_computed()
    {
        if (mBowVec.empty())
            compute_BoW();
    }

    void compute_BoW();

    // Undistort keypoints given OpenCV distortion parameters.
    // Only for the RGB-D case. Stereo must be already rectified!
    // (called in the constructor).
    void UndistortKeyPoints();

    // Computes image bounds for the undistorted image (called in the constructor).
    void ComputeImageBounds(cv::Mat const& imLeft);

    // Assign keypoints to the grid for speed up feature matching (called in the constructor).
    void AssignFeaturesToGrid();

    // Vector of keypoints (original for visualization) and undistorted (actually used by the system).
    // In the stereo case, mvKeysUn is redundant as images must be rectified.
    // In the RGB-D case, RGB images can be distorted.
    std::vector<cv::KeyPoint>   mvKeys;

    // Bag of Words Vector structures.
    DBoW2::BowVector            mBowVec;
    DBoW2::FeatureVector        mFeatVec;

    // Stereo baseline multiplied by fx
    float                       mbf;

    // Stereo baseline in meters.
    float                       mb;

    // Current and Next Frame id.
    static uint64_t             s_next_id;
    uint64_t                    m_id;

    // Frame timestamp.
    double                      mTimeStamp;

    // Calibration matrix and OpenCV distortion parameters.
    cv::Mat                     mK;
    cv::Mat                     mDistCoef;

    // Feature extractor. The right is used only in the stereo case.
    ORBextractor*               mpORBextractorLeft;
    ORBextractor*               mpORBextractorRight;

    // Vocabulary used for relocalization.
    ORBVocabulary*              mpORBvocabulary;

    // PAE: added to make it a bit faster
    cv::Mat                     m_imLeft;

    // ORB descriptor, each row associated to a keypoint.
    cv::Mat                     mDescriptors;

    // Rotation, translation and camera center
    cv::Mat                     mRcw;
    cv::Mat                     mtcw;
    cv::Mat                     mRwc;
    cv::Mat                     mOw; //==mtwc
};

}// namespace ORB_SLAM

#endif // FRAME_H
