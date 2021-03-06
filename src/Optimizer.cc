﻿/**
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

#include "Optimizer.h"

#include "Map.h"
#include "KeyFrame.h"

#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

#include <Eigen/StdVector>

#include "Converter.h"

namespace ORB_SLAM2
{

std::pair<std::unordered_map<KeyFrame*, cv::Mat>, std::unordered_map<MapPoint*, cv::Mat> >
Optimizer::GlobalBundleAdjustemnt(Map* pMap, int nIterations, bool* pbStopFlag, bool bUpdateAtOnce, bool bRobust)
{
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    std::vector<KeyFrame*> const& vpKFs = pMap->GetAllKeyFrames();
    std::vector<MapPoint*> const& vpMP = pMap->GetAllMapPoints();

    std::vector<bool> vbNotIncludedMP;
    vbNotIncludedMP.resize(vpMP.size());

    uint64_t maxKFid = 0;

    // Set KeyFrame vertices
    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        KeyFrame* pKF = vpKFs[i];
        if (pKF->isBad())
            continue;

        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKF->GetPose()));
        vSE3->setId(pKF->get_id());
        vSE3->setFixed(pKF->get_id() == 0);

        optimizer.addVertex(vSE3);

        if (pKF->get_id() > maxKFid)
            maxKFid = pKF->get_id();
    }

    float const thHuber2D = std::sqrtf(5.99f);
    float const thHuber3D = std::sqrtf(7.815f);

    // Set MapPoint vertices
    for (size_t i = 0; i < vpMP.size(); ++i)
    {
        MapPoint* pMP = vpMP[i];
        if (pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));
        int const id = (int)(pMP->get_id() + maxKFid + 1);
        vPoint->setId(id);
        vPoint->setMarginalized(true);

        optimizer.addVertex(vPoint);

        std::unordered_map<KeyFrame*, size_t> const & observations = pMP->GetObservations();

        int nEdges = 0;

        // SET EDGES
        for (auto const& pair : observations)
        {
            KeyFrame* pKF = pair.first;

            if (pKF->isBad() || pKF->get_id() > maxKFid)
                continue;

            ++nEdges;

            cv::KeyPoint const& kpUn = pKF->mvKeysUn[pair.second];

            if (pKF->mvuRight[pair.second] < 0)
            {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, optimizer.vertex(id));
                e->setVertex(1, optimizer.vertex(pKF->get_id()));
                e->setMeasurement(obs);

                float const invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber2D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;

                optimizer.addEdge(e);
            }
            else
            {
                Eigen::Matrix<double, 3, 1> obs;
                float const kp_ur = pKF->mvuRight[pair.second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();
                e->setVertex(0, optimizer.vertex(id));
                e->setVertex(1, optimizer.vertex(pKF->get_id()));
                e->setMeasurement(obs);
                float const invSigma2 = pKF->mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity() * invSigma2;
                e->setInformation(Info);

                if (bRobust)
                {
                    g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                    e->setRobustKernel(rk);
                    rk->setDelta(thHuber3D);
                }

                e->fx = pKF->fx;
                e->fy = pKF->fy;
                e->cx = pKF->cx;
                e->cy = pKF->cy;
                e->bf = pKF->mbf;

                optimizer.addEdge(e);
            }
        }

        if (nEdges == 0)
        {
            optimizer.removeVertex(vPoint);
            vbNotIncludedMP[i] = true;
        }
        else
        {
            vbNotIncludedMP[i] = false;
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(nIterations);

    // Recover optimized data

    std::pair<std::unordered_map<KeyFrame*, cv::Mat>, std::unordered_map<MapPoint*, cv::Mat> > result;

    // Keyframes
    for (KeyFrame* pKF : vpKFs)
    {
        if (pKF->isBad())
            continue;

        g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(pKF->get_id()));
        g2o::SE3Quat SE3quat = vSE3->estimate();

        if (bUpdateAtOnce)
        {
            pKF->SetPose(Converter::toCvMat(SE3quat));
        }
        else
        {
            auto const& pair = result.first.emplace(pKF, Converter::toCvMat(SE3quat));
            assert(pair.second);
        }
    }

    // Points
    for (size_t i = 0; i < vpMP.size(); ++i)
    {
        if (vbNotIncludedMP[i])
            continue;

        MapPoint* pMP = vpMP[i];

        if (pMP->isBad())
            continue;

        g2o::VertexSBAPointXYZ* vPoint = static_cast<g2o::VertexSBAPointXYZ*>(optimizer.vertex(pMP->get_id() + maxKFid+1));

        if (bUpdateAtOnce)
        {
            pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
            pMP->UpdateNormalAndDepth();
        }
        else
        {
            auto const& pair = result.second.emplace(pMP, Converter::toCvMat(vPoint->estimate()));
            assert(pair.second);
        }
    }

    return result;
}

std::size_t Optimizer::PoseOptimization(Frame* pFrame)
{
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(new g2o::OptimizationAlgorithmLevenberg(solver_ptr));

    // Set Frame vertex
    g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    std::size_t const N = pFrame->get_frame_N();

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vpEdgesMono.reserve(N);

    std::vector<std::size_t> vnIndexEdgeMono;
    vnIndexEdgeMono.reserve(N);

    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vpEdgesStereo.reserve(N);

    std::vector<std::size_t> vnIndexEdgeStereo;
    vnIndexEdgeStereo.reserve(N);

    float const deltaMono = std::sqrt(5.991f);
    float const deltaStereo = std::sqrt(7.815f);

    std::size_t nInitialCorrespondences = 0;

    {
        std::unique_lock<std::mutex> lock(MapPoint::mGlobalMutex);

        for (std::size_t i = 0; i < N; ++i)
        {
            MapPoint* pMP = pFrame->mvpMapPoints[i];

            if (pMP == nullptr)
                continue;

            ++nInitialCorrespondences;
            pFrame->mvbOutlier[i] = false;

            cv::KeyPoint const& kpUn = pFrame->mvKeysUn[i];
            float const invSigma2 = pFrame->mvInvLevelSigma2[kpUn.octave];

            // Monocular observation
            if (pFrame->mvuRight[i] < 0)
            {
                Eigen::Matrix<double, 2, 1> obs;

                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                e->setId((int)pMP->get_id());
                e->setVertex(0, optimizer.vertex(0));
                e->setMeasurement(obs);

                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;

                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation
            {
                // SET EDGE
                Eigen::Matrix<double, 3, 1> obs;
                float const kp_ur = pFrame->mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                e->setId((int)pMP->get_id());
                e->setVertex(0, optimizer.vertex(0));
                e->setMeasurement(obs);

                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = pFrame->fx;
                e->fy = pFrame->fy;
                e->cx = pFrame->cx;
                e->cy = pFrame->cy;
                e->bf = pFrame->get_mbf();

                cv::Mat Xw = pMP->GetWorldPos();
                e->Xw[0] = Xw.at<float>(0);
                e->Xw[1] = Xw.at<float>(1);
                e->Xw[2] = Xw.at<float>(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }

    if (nInitialCorrespondences < 3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation
    // as inlier/outlier, At the next optimization, outliers are not included, but
    // at the end they can be classified as inliers again.
    static float const chi2Mono[4] = { 5.991, 4.891, 3.791, 2.691 };
    static float const chi2Stereo[4] = { 7.815f, 7.815f, 7.815f, 7.815f };
    static int const its[4] = { 10, 10, 10, 10 };

    std::size_t nBad = 0;

    for (size_t it = 0; it < 4; ++it)
    {
        vSE3->setEstimate(Converter::toSE3Quat(pFrame->mTcw));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;

        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            size_t const idx = vnIndexEdgeMono[i];

            if (pFrame->mvbOutlier[idx])
                e->computeError();

            bool const is_bad = e->chi2() > chi2Mono[it];
            pFrame->mvbOutlier[idx] = is_bad;
            e->setLevel((int)is_bad);
            nBad += (int)is_bad;

            if (it == 2)
                e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            size_t const idx = vnIndexEdgeStereo[i];

            if (pFrame->mvbOutlier[idx])
                e->computeError();

            bool const is_bad = e->chi2() > chi2Stereo[it];
            pFrame->mvbOutlier[idx] = is_bad;
            e->setLevel((int)is_bad);
            nBad += (int)is_bad;

            if (it == 2)
                e->setRobustKernel(0);
        }

        if (optimizer.edges().size() < 10)
            break;
    }

    // Recover optimized pose and return number of inliers
    pFrame->SetPose(Converter::toCvMat(vSE3->estimate()));

    return nInitialCorrespondences - nBad;
}

void Optimizer::LocalBundleAdjustment(KeyFrame *pKF, bool* pbStopFlag, Map* pMap)
{
    // Local KeyFrames: First Breath Search from Current Keyframe
    std::unordered_set<KeyFrame*> local_key_frames;
    local_key_frames.emplace(pKF);

    std::vector<KeyFrame*> const& covisible_frames = pKF->GetVectorCovisibleKeyFrames();
    for (KeyFrame* pKFCV : covisible_frames)
    {
        if (pKFCV->isBad())
            continue;

        local_key_frames.emplace(pKFCV);
    }

    // Local MapPoints seen in Local KeyFrames
    std::unordered_set<MapPoint*> local_map_points;
    for (KeyFrame* pLKF : local_key_frames)
    {
        std::vector<MapPoint*> const& lkf_map_points = pLKF->GetMapPointMatches();
        for (MapPoint* pMP : lkf_map_points)
        {
            if (pMP == nullptr || pMP->isBad())
                continue;

            local_map_points.emplace(pMP);
        }
    }

    std::unordered_set<KeyFrame*> fixel_frames;
    for (MapPoint *pLMP : local_map_points)
    {
        std::unordered_map<KeyFrame*, size_t> const& observations = pLMP->GetObservations();
        for (auto const& pair : observations)
        {
            if (pair.first->isBad() || local_key_frames.find(pair.first) != local_key_frames.end())
                continue;

            fixel_frames.emplace(pair.first);
        }
    }

    // Setup optimizer
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    if (pbStopFlag)
        optimizer.setForceStopFlag(pbStopFlag);

    uint64_t maxKFid = 0;

    // set local KeyFrame vertices
    for (KeyFrame* pLKF : local_key_frames)
    {
        g2o::VertexSE3Expmap* vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pLKF->GetPose()));
        vSE3->setId(pLKF->get_id());
        vSE3->setFixed(pLKF->get_id() == 0);

        optimizer.addVertex(vSE3);

        if (pLKF->get_id() > maxKFid)
            maxKFid = pLKF->get_id();
    }

    // Set Fixed KeyFrame vertices
    for (KeyFrame* pKFF : fixel_frames)
    {
        g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setEstimate(Converter::toSE3Quat(pKFF->GetPose()));
        vSE3->setId(pKFF->get_id());
        vSE3->setFixed(true);

        optimizer.addVertex(vSE3);

        if (pKFF->get_id() > maxKFid)
            maxKFid = pKFF->get_id();
    }

    // Set MapPoint vertices
    std::size_t const nExpectedSize = (local_key_frames.size() + fixel_frames.size()) * local_map_points.size();

    std::vector<g2o::EdgeSE3ProjectXYZ*> vpEdgesMono;
    vpEdgesMono.reserve(nExpectedSize);

    std::vector<KeyFrame*> vpEdgeKFMono;
    vpEdgeKFMono.reserve(nExpectedSize);

    std::vector<MapPoint*> vpMapPointEdgeMono;
    vpMapPointEdgeMono.reserve(nExpectedSize);

    std::vector<g2o::EdgeStereoSE3ProjectXYZ*> vpEdgesStereo;
    vpEdgesStereo.reserve(nExpectedSize);

    std::vector<KeyFrame*> vpEdgeKFStereo;
    vpEdgeKFStereo.reserve(nExpectedSize);

    std::vector<MapPoint*> vpMapPointEdgeStereo;
    vpMapPointEdgeStereo.reserve(nExpectedSize);

    double const thHuberMono = std::sqrt(5.991);
    double const thHuberStereo = std::sqrt(7.815);

    for (MapPoint* pMP : local_map_points)
    {
        g2o::VertexSBAPointXYZ* vPoint = new g2o::VertexSBAPointXYZ();
        vPoint->setEstimate(Converter::toVector3d(pMP->GetWorldPos()));

        int const id = pMP->get_id() + maxKFid + 1;

        vPoint->setId(id);
        vPoint->setMarginalized(true);
        optimizer.addVertex(vPoint);

        // Set edges
        std::unordered_map<KeyFrame*, size_t> const& observations = pMP->GetObservations();
        for (auto const& pair : observations)
        {
            KeyFrame* pKFO = pair.first;

            if (pKFO->isBad())
                continue;

            cv::KeyPoint const& kpUn = pKFO->mvKeysUn[pair.second];

            // Monocular observation
            if (pKFO->mvuRight[pair.second] < 0)
            {
                Eigen::Matrix<double, 2, 1> obs;
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();

                e->setVertex(0, optimizer.vertex(id));
                e->setVertex(1, optimizer.vertex(pKFO->get_id()));
                e->setMeasurement(obs);

                float const invSigma2 = pKFO->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberMono);

                e->fx = pKFO->fx;
                e->fy = pKFO->fy;
                e->cx = pKFO->cx;
                e->cy = pKFO->cy;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vpEdgeKFMono.push_back(pKFO);
                vpMapPointEdgeMono.push_back(pMP);
            }
            else // Stereo observation
            {
                Eigen::Matrix<double, 3, 1> obs;
                float const kp_ur = pKFO->mvuRight[pair.second];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;

                g2o::EdgeStereoSE3ProjectXYZ* e = new g2o::EdgeStereoSE3ProjectXYZ();

                e->setVertex(0, optimizer.vertex(id));
                e->setVertex(1, optimizer.vertex(pKFO->get_id()));
                e->setMeasurement(obs);

                float const invSigma2 = pKFO->mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(thHuberStereo);

                e->fx = pKFO->fx;
                e->fy = pKFO->fy;
                e->cx = pKFO->cx;
                e->cy = pKFO->cy;
                e->bf = pKFO->mbf;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vpEdgeKFStereo.push_back(pKFO);
                vpMapPointEdgeStereo.push_back(pMP);
            }
        }
    }

    if (pbStopFlag != nullptr && *pbStopFlag)
        return;

    optimizer.initializeOptimization();
    optimizer.optimize(5);

    if (pbStopFlag == nullptr || !(*pbStopFlag))
    {
        // Check inlier observations
        for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointEdgeMono[i];

            if (pMP->isBad()) // TODO: PAE: HM can it get changed?
                continue;

            g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];

            if (e->chi2() > 5.991 || !e->isDepthPositive())
                e->setLevel(1);

            e->setRobustKernel(0);
        }

        for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPointEdgeStereo[i];

            if (pMP->isBad()) // TODO: PAE: HM can it get changed?
                continue;

            g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
            if (e->chi2() > 7.815 || !e->isDepthPositive())
                e->setLevel(1);

            e->setRobustKernel(0);
        }

        // Optimize again without the outliers
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
    }

    std::vector<std::pair<KeyFrame*, MapPoint*> > vToErase;
    vToErase.reserve(vpEdgesMono.size() + vpEdgesStereo.size());

    // Check inlier observations
    for (size_t i = 0, iend = vpEdgesMono.size(); i < iend; ++i)
    {
        MapPoint* pMP = vpMapPointEdgeMono[i];

        if (pMP->isBad()) // TODO: PAE: hm can it get changed?
            continue;

        g2o::EdgeSE3ProjectXYZ* e = vpEdgesMono[i];
        if (e->chi2() > 5.991 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFMono[i];
            vToErase.emplace_back(pKFi, pMP);
        }
    }

    for (size_t i = 0, iend = vpEdgesStereo.size(); i < iend; ++i)
    {
        MapPoint* pMP = vpMapPointEdgeStereo[i];

        if (pMP->isBad())  // TODO: PAE: hm can it get changed?
            continue;

        g2o::EdgeStereoSE3ProjectXYZ* e = vpEdgesStereo[i];
        if (e->chi2() > 7.815 || !e->isDepthPositive())
        {
            KeyFrame* pKFi = vpEdgeKFStereo[i];
            vToErase.emplace_back(pKFi, pMP);
        }
    }

    // Get Map Mutex
    // PAE: threading removed std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

    for (size_t i = 0; i < vToErase.size(); ++i)
    {
        KeyFrame* pKFi = vToErase[i].first;
        MapPoint* pMPi = vToErase[i].second;
        pKFi->EraseMapPointMatch(pMPi);
        pMPi->EraseObservation(pKFi);
    }

    // Recover optimized data

    for (KeyFrame* pKF : local_key_frames)
    {
        g2o::VertexSE3Expmap* vSE3 = (g2o::VertexSE3Expmap*)(optimizer.vertex(pKF->get_id()));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        pKF->SetPose(Converter::toCvMat(SE3quat));
    }

    for (MapPoint* pMP : local_map_points)
    {
        g2o::VertexSBAPointXYZ* vPoint = (g2o::VertexSBAPointXYZ*)(optimizer.vertex(pMP->get_id() + maxKFid + 1));
        pMP->SetWorldPos(Converter::toCvMat(vPoint->estimate()));
        pMP->UpdateNormalAndDepth();
    }
}


void Optimizer::OptimizeEssentialGraph(Map* pMap, KeyFrame* pLoopKF, KeyFrame* pCurKF,
    LoopClosing::KeyFrameAndPose const& NonCorrectedSim3, LoopClosing::KeyFrameAndPose const& CorrectedSim3,
    std::unordered_map<KeyFrame*, std::unordered_set<KeyFrame*> > const& LoopConnections, bool bFixScale)
{
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);
    g2o::BlockSolver_7_3::LinearSolverType * linearSolver =
           new g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>();
    g2o::BlockSolver_7_3 * solver_ptr= new g2o::BlockSolver_7_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    solver->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(solver);

    const vector<KeyFrame*> vpKFs = pMap->GetAllKeyFrames();
    const vector<MapPoint*> vpMPs = pMap->GetAllMapPoints();

    uint64_t const nMaxKFid = pMap->GetMaxKFid();

    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vScw(nMaxKFid + 1);
    vector<g2o::Sim3, Eigen::aligned_allocator<g2o::Sim3> > vCorrectedSwc(nMaxKFid + 1);
    vector<g2o::VertexSim3Expmap*> vpVertices(nMaxKFid + 1);

    const int minFeat = 100;

    // Set KeyFrame vertices
    for(size_t i=0, iend=vpKFs.size(); i<iend;i++)
    {
        KeyFrame* pKF = vpKFs[i];
        if(pKF->isBad())
            continue;
        g2o::VertexSim3Expmap* VSim3 = new g2o::VertexSim3Expmap();

        const int nIDi = pKF->get_id();

        LoopClosing::KeyFrameAndPose::const_iterator it = CorrectedSim3.find(pKF);

        if(it!=CorrectedSim3.end())
        {
            vScw[nIDi] = it->second;
            VSim3->setEstimate(it->second);
        }
        else
        {
            Eigen::Matrix<double,3,3> Rcw = Converter::toMatrix3d(pKF->GetRotation());
            Eigen::Matrix<double,3,1> tcw = Converter::toVector3d(pKF->GetTranslation());
            g2o::Sim3 Siw(Rcw,tcw,1.0);
            vScw[nIDi] = Siw;
            VSim3->setEstimate(Siw);
        }

        if(pKF==pLoopKF)
            VSim3->setFixed(true);

        VSim3->setId(nIDi);
        VSim3->setMarginalized(false);
        VSim3->_fix_scale = bFixScale;

        optimizer.addVertex(VSim3);

        vpVertices[nIDi]=VSim3;
    }

    // NOTE: PAE: need to change to unordered_set but have to add hasher first
    std::set<std::pair<uint64_t, uint64_t> > sInsertedEdges;

    const Eigen::Matrix<double,7,7> matLambda = Eigen::Matrix<double,7,7>::Identity();

    // Set Loop edges
    for (auto mit = LoopConnections.begin(), mend=LoopConnections.end(); mit != mend; ++mit)
    {
        KeyFrame* pKF = mit->first;

        uint64_t const nIDi = pKF->get_id();
        std::unordered_set<KeyFrame*> const& spConnections = mit->second;

        const g2o::Sim3 Siw = vScw[nIDi];
        const g2o::Sim3 Swi = Siw.inverse();

        for (auto sit = spConnections.begin(), send = spConnections.end(); sit != send; ++sit)
        {
            uint64_t const nIDj = (*sit)->get_id();

            if ((nIDi != pCurKF->get_id() || nIDj != pLoopKF->get_id()) && pKF->GetWeight(*sit) < minFeat)
                continue;

            const g2o::Sim3 Sjw = vScw[nIDj];
            const g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, optimizer.vertex(nIDj));
            e->setVertex(0, optimizer.vertex(nIDi));
            e->setMeasurement(Sji);

            e->information() = matLambda;

            optimizer.addEdge(e);

            sInsertedEdges.emplace((std::min)(nIDi, nIDj), (std::max)(nIDi, nIDj));
        }
    }

    // Set normal edges
    for(size_t i=0, iend=vpKFs.size(); i<iend; i++)
    {
        KeyFrame* pKF = vpKFs[i];

        const int nIDi = pKF->get_id();

        g2o::Sim3 Swi;

        LoopClosing::KeyFrameAndPose::const_iterator iti = NonCorrectedSim3.find(pKF);

        if(iti!=NonCorrectedSim3.end())
            Swi = (iti->second).inverse();
        else
            Swi = vScw[nIDi].inverse();

        KeyFrame* pParentKF = pKF->GetParent();

        // Spanning tree edge
        if(pParentKF)
        {
            int nIDj = pParentKF->get_id();

            g2o::Sim3 Sjw;

            LoopClosing::KeyFrameAndPose::const_iterator itj = NonCorrectedSim3.find(pParentKF);

            if(itj!=NonCorrectedSim3.end())
                Sjw = itj->second;
            else
                Sjw = vScw[nIDj];

            g2o::Sim3 Sji = Sjw * Swi;

            g2o::EdgeSim3* e = new g2o::EdgeSim3();
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDj)));
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(nIDi)));
            e->setMeasurement(Sji);

            e->information() = matLambda;
            optimizer.addEdge(e);
        }

        // Loop edges
        const std::unordered_set<KeyFrame*> sLoopEdges = pKF->GetLoopEdges();
        for (KeyFrame* pLKF : sLoopEdges)
        {
            if (pLKF->get_id() >= pKF->get_id())
                continue;

            g2o::Sim3 Slw;

            LoopClosing::KeyFrameAndPose::const_iterator itl = NonCorrectedSim3.find(pLKF);

            if(itl!=NonCorrectedSim3.end())
                Slw = itl->second;
            else
                Slw = vScw[pLKF->get_id()];

            g2o::Sim3 Sli = Slw * Swi;
            g2o::EdgeSim3* el = new g2o::EdgeSim3();
            el->setVertex(1, optimizer.vertex(pLKF->get_id()));
            el->setVertex(0, optimizer.vertex(nIDi));
            el->setMeasurement(Sli);
            el->information() = matLambda;
            optimizer.addEdge(el);
        }

        // Covisibility graph edges
        const vector<KeyFrame*> vpConnectedKFs = pKF->GetCovisiblesByWeight(minFeat);
        for (KeyFrame* pKFn : vpConnectedKFs)
        {
            if (pKFn == nullptr || pKFn == pParentKF || pKF->hasChild(pKFn) ||
                sLoopEdges.count(pKFn) != 0 || pKFn->isBad() || pKFn->get_id() >= pKF->get_id())
                continue;

            if (sInsertedEdges.count(std::make_pair((std::min)(pKF->get_id(), pKFn->get_id()), (std::max)(pKF->get_id(), pKFn->get_id()))))
                continue;

            g2o::Sim3 Snw;

            LoopClosing::KeyFrameAndPose::const_iterator itn = NonCorrectedSim3.find(pKFn);

            if(itn!=NonCorrectedSim3.end())
                Snw = itn->second;
            else
                Snw = vScw[pKFn->get_id()];

            g2o::Sim3 Sni = Snw * Swi;

            g2o::EdgeSim3* en = new g2o::EdgeSim3();
            en->setVertex(1, optimizer.vertex(pKFn->get_id()));
            en->setVertex(0, optimizer.vertex(nIDi));
            en->setMeasurement(Sni);
            en->information() = matLambda;
            optimizer.addEdge(en);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    std::unique_lock<std::mutex> lock(pMap->mMutexMapUpdate);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (size_t i = 0; i < vpKFs.size(); ++i)
    {
        KeyFrame* pKFi = vpKFs[i];

        const int nIDi = pKFi->get_id();

        g2o::VertexSim3Expmap* VSim3 = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(nIDi));
        g2o::Sim3 CorrectedSiw =  VSim3->estimate();
        vCorrectedSwc[nIDi]=CorrectedSiw.inverse();
        Eigen::Matrix3d eigR = CorrectedSiw.rotation().toRotationMatrix();
        Eigen::Vector3d eigt = CorrectedSiw.translation();
        double s = CorrectedSiw.scale();

        eigt *=(1./s); //[R t/s;0 1]

        cv::Mat Tiw = Converter::toCvSE3(eigR,eigt);

        pKFi->SetPose(Tiw);
    }

    // Correct points. Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if (pMP->isBad())
            continue;

        int nIDr;
        if (pMP->mnCorrectedByKF == pCurKF->get_id())
            nIDr = pMP->mnCorrectedReference;
        else
        {
            KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
            nIDr = pRefKF->get_id();
        }


        g2o::Sim3 Srw = vScw[nIDr];
        g2o::Sim3 correctedSwr = vCorrectedSwc[nIDr];

        cv::Mat P3Dw = pMP->GetWorldPos();
        Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
        Eigen::Matrix<double,3,1> eigCorrectedP3Dw = correctedSwr.map(Srw.map(eigP3Dw));

        cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
        pMP->SetWorldPos(cvCorrectedP3Dw);

        pMP->UpdateNormalAndDepth();
    }
}

int Optimizer::OptimizeSim3(KeyFrame *pKF1, KeyFrame *pKF2,
    std::vector<MapPoint*>& vpMatches1, g2o::Sim3& g2oS12, float th2, bool bFixScale)
{
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolverX::PoseMatrixType>();

    g2o::BlockSolverX * solver_ptr = new g2o::BlockSolverX(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    // Calibration
    const cv::Mat &K1 = pKF1->mK;
    const cv::Mat &K2 = pKF2->mK;

    // Camera poses
    const cv::Mat R1w = pKF1->GetRotation();
    const cv::Mat t1w = pKF1->GetTranslation();
    const cv::Mat R2w = pKF2->GetRotation();
    const cv::Mat t2w = pKF2->GetTranslation();

    // Set Sim3 vertex
    g2o::VertexSim3Expmap * vSim3 = new g2o::VertexSim3Expmap();    
    vSim3->_fix_scale=bFixScale;
    vSim3->setEstimate(g2oS12);
    vSim3->setId(0);
    vSim3->setFixed(false);
    vSim3->_principle_point1[0] = K1.at<float>(0,2);
    vSim3->_principle_point1[1] = K1.at<float>(1,2);
    vSim3->_focal_length1[0] = K1.at<float>(0,0);
    vSim3->_focal_length1[1] = K1.at<float>(1,1);
    vSim3->_principle_point2[0] = K2.at<float>(0,2);
    vSim3->_principle_point2[1] = K2.at<float>(1,2);
    vSim3->_focal_length2[0] = K2.at<float>(0,0);
    vSim3->_focal_length2[1] = K2.at<float>(1,1);
    optimizer.addVertex(vSim3);

    // Set MapPoint vertices
    const int N = vpMatches1.size();
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    vector<g2o::EdgeSim3ProjectXYZ*> vpEdges12;
    vector<g2o::EdgeInverseSim3ProjectXYZ*> vpEdges21;
    vector<size_t> vnIndexEdge;

    vnIndexEdge.reserve(2*N);
    vpEdges12.reserve(2*N);
    vpEdges21.reserve(2*N);

    const float deltaHuber = sqrt(th2);

    int nCorrespondences = 0;

    for (int i = 0; i < N; ++i)
    {
        if (!vpMatches1[i])
            continue;

        MapPoint* pMP1 = vpMapPoints1[i];
        MapPoint* pMP2 = vpMatches1[i];

        const int id1 = 2*i+1;
        const int id2 = 2*(i+1);

        if (pMP1 == nullptr || pMP1->isBad() ||
            pMP2 == nullptr || pMP2->isBad())
            continue;

        const int i2 = pMP2->GetIndexInKeyFrame(pKF2);

        if (i2 < 0)
            continue;

        g2o::VertexSBAPointXYZ* vPoint1 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D1w = pMP1->GetWorldPos();
        cv::Mat P3D1c = R1w*P3D1w + t1w;
        vPoint1->setEstimate(Converter::toVector3d(P3D1c));
        vPoint1->setId(id1);
        vPoint1->setFixed(true);
        optimizer.addVertex(vPoint1);

        g2o::VertexSBAPointXYZ* vPoint2 = new g2o::VertexSBAPointXYZ();
        cv::Mat P3D2w = pMP2->GetWorldPos();
        cv::Mat P3D2c = R2w*P3D2w + t2w;
        vPoint2->setEstimate(Converter::toVector3d(P3D2c));
        vPoint2->setId(id2);
        vPoint2->setFixed(true);
        optimizer.addVertex(vPoint2);

        nCorrespondences++;

        // Set edge x1 = S12*X2
        Eigen::Matrix<double,2,1> obs1;
        const cv::KeyPoint &kpUn1 = pKF1->mvKeysUn[i];
        obs1 << kpUn1.pt.x, kpUn1.pt.y;

        g2o::EdgeSim3ProjectXYZ* e12 = new g2o::EdgeSim3ProjectXYZ();
        e12->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id2)));
        e12->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e12->setMeasurement(obs1);
        const float &invSigmaSquare1 = pKF1->mvInvLevelSigma2[kpUn1.octave];
        e12->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare1);

        g2o::RobustKernelHuber* rk1 = new g2o::RobustKernelHuber;
        e12->setRobustKernel(rk1);
        rk1->setDelta(deltaHuber);
        optimizer.addEdge(e12);

        // Set edge x2 = S21*X1
        Eigen::Matrix<double,2,1> obs2;
        const cv::KeyPoint &kpUn2 = pKF2->mvKeysUn[i2];
        obs2 << kpUn2.pt.x, kpUn2.pt.y;

        g2o::EdgeInverseSim3ProjectXYZ* e21 = new g2o::EdgeInverseSim3ProjectXYZ();

        e21->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(id1)));
        e21->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e21->setMeasurement(obs2);
        float invSigmaSquare2 = pKF2->mvInvLevelSigma2[kpUn2.octave];
        e21->setInformation(Eigen::Matrix2d::Identity()*invSigmaSquare2);

        g2o::RobustKernelHuber* rk2 = new g2o::RobustKernelHuber;
        e21->setRobustKernel(rk2);
        rk2->setDelta(deltaHuber);
        optimizer.addEdge(e21);

        vpEdges12.push_back(e12);
        vpEdges21.push_back(e21);
        vnIndexEdge.push_back(i);
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers
    int nBad=0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];
        if(!e12 || !e21)
            continue;

        if(e12->chi2()>th2 || e21->chi2()>th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx]=static_cast<MapPoint*>(NULL);
            optimizer.removeEdge(e12);
            optimizer.removeEdge(e21);
            vpEdges12[i]=static_cast<g2o::EdgeSim3ProjectXYZ*>(NULL);
            vpEdges21[i]=static_cast<g2o::EdgeInverseSim3ProjectXYZ*>(NULL);
            nBad++;
        }
    }

    int const nMoreIterations = nBad > 0 ? 10 : 5;

    if (nCorrespondences - nBad < 10)
        return 0;

    // Optimize again only with inliers

    optimizer.initializeOptimization();
    optimizer.optimize(nMoreIterations);

    int nIn = 0;
    for(size_t i=0; i<vpEdges12.size();i++)
    {
        g2o::EdgeSim3ProjectXYZ* e12 = vpEdges12[i];
        g2o::EdgeInverseSim3ProjectXYZ* e21 = vpEdges21[i];

        if (e12 == nullptr || e21 == nullptr)
            continue;

        if (e12->chi2() > th2 || e21->chi2() > th2)
        {
            size_t idx = vnIndexEdge[i];
            vpMatches1[idx] = nullptr;
        }
        else
            nIn++;
    }

    // Recover optimized Sim3
    g2o::VertexSim3Expmap* vSim3_recov = static_cast<g2o::VertexSim3Expmap*>(optimizer.vertex(0));
    g2oS12= vSim3_recov->estimate();

    return nIn;
}


} //namespace ORB_SLAM
