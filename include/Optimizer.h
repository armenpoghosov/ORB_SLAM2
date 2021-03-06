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

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "LoopClosing.h"

namespace ORB_SLAM2
{

class Frame;
class KeyFrame;
class Map;
class MapPoint;

class Optimizer
{
public:

    // either update keyframes and map points at once or return the updated positions in the result pair
    static std::pair<std::unordered_map<KeyFrame*, cv::Mat>, std::unordered_map<MapPoint*, cv::Mat> >
    GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool *pbStopFlag, bool bUpdateAtOnce, bool bRobus);

    static void LocalBundleAdjustment(KeyFrame* pKF, bool *pbStopFlag, Map *pMap);

    static std::size_t PoseOptimization(Frame* pFrame);

    // if bFixScale is true, 6DoF optimization (stereo,rgbd), 7DoF otherwise (mono)
    static void OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
        LoopClosing::KeyFrameAndPose const& NonCorrectedSim3, LoopClosing::KeyFrameAndPose const& CorrectedSim3,
        std::unordered_map<KeyFrame*, std::unordered_set<KeyFrame*> > const& LoopConnections, bool bFixScale);

    // if bFixScale is true, optimize SE3 (stereo,rgbd), Sim3 otherwise (mono)
    static int OptimizeSim3(KeyFrame* pKF1, KeyFrame* pKF2,
        std::vector<MapPoint*>& vpMatches1, g2o::Sim3& g2oS12, float th2, bool bFixScale);
};

} //namespace ORB_SLAM

#endif // OPTIMIZER_H
