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

#include "LoopClosing.h"

#include "Converter.h"
#include "Map.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "Sim3Solver.h"

namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, bool bFixScale)
    :
    mbResetRequested(false),
    mbFinishRequested(false),
    mbFinished(true),
    mpMap(pMap),
    mpKeyFrameDB(pDB),
    mpORBVocabulary(pVoc),
    mpMatchedKF(NULL),
    mLastLoopKFid(0),
    mbRunningGBA(false),
    mbFinishedGBA(true),
    mbStopGBA(false),
    mpThreadGBA(nullptr),
    mbFixScale(bFixScale),
    mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
    m_thread = std::thread(&LoopClosing::Run, this);
}

LoopClosing::~LoopClosing()
{
    RequestFinish();

    if (m_thread.joinable())
        m_thread.join();
}

void LoopClosing::Run()
{
    mbFinished = false;

    for (;;)
    {
        /* PAE: removed loop closing
        // Check if there are keyframes in the queue
        if (CheckNewKeyFrames() &&
            // Detect loop candidates and check covisibility consistency
            DetectLoop() &&
            // Compute similarity transformation [sR|t]
            // In the stereo/RGBD case s=1
            ComputeSim3())
        {
            // Perform loop fusion and pose graph optimization
            CorrectLoop();
        }*/

        ResetIfRequested();

        if (CheckFinish())
            break;

        // TODO: PAE: wait for new frames?
        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexLoopQueue);
    if (pKF->get_id() != 0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::DetectLoop()
{
    {
        std::unique_lock<std::mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    // If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if (mpCurrentKF->get_id() < mLastLoopKFid + 10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    std::vector<KeyFrame*> const vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    DBoW2::BowVector const& CurrentBowVec = mpCurrentKF->get_BoW();

    float minScore = 1.f;

    for (KeyFrame* pKF : vpConnectedKeyFrames)
    {
        if (pKF->isBad())
            continue;

        float const score = mpORBVocabulary->score(CurrentBowVec, pKF->get_BoW());
        if (score < minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if (vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    std::vector<ConsistentGroup> vCurrentConsistentGroups;
    std::vector<bool> vbConsistentGroup(mvConsistentGroups.size());

    for (size_t i = 0, iend = vpCandidateKFs.size(); i < iend; ++i)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        std::unordered_set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; ++iG)
        {
            std::unordered_set<KeyFrame*> const& sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for (KeyFrame* pCandiateKF : spCandidateGroup)
            {
                if (sPreviousGroup.count(pCandiateKF) != 0)
                {
                    bConsistent = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if (!bConsistent)
                continue;

            int nPreviousConsistency = mvConsistentGroups[iG].second;
            int nCurrentConsistency = nPreviousConsistency + 1;

            if (!vbConsistentGroup[iG])
            {
                vCurrentConsistentGroups.emplace_back(spCandidateGroup, nCurrentConsistency);
                vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
            }

            if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
            {
                mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                bEnoughConsistent = true; //this avoid to insert the same candidate more than once
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if (!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if (mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }

    return true;
}

bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3
    std::size_t const nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75, true);

    std::vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

    std::vector<std::vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    std::vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    std::size_t nCandidates = 0; //candidates with enough matches

    for (std::size_t i = 0; i < nInitialCandidates; ++i)
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase();

        if (pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, vvpMapPointMatches[i]);
        if (nmatches < 20)
        {
            vbDiscarded[i] = true;
            continue;
        }

        Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF, pKF, vvpMapPointMatches[i], mbFixScale);
        pSolver->SetRansacParameters(0.99, 20, 300);
        vpSim3Solvers[i] = pSolver;

        ++nCandidates;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while (nCandidates > 0 && !bMatch)
    {
        for (std::size_t i = 0; i < nInitialCandidates; ++i)
        {
            if (vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // Perform 5 Ransac Iterations
            int nInliers;
            bool bNoMore;
            vector<bool> vbInliers;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                --nCandidates;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if (Scm.empty())
                continue;

            std::vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size());

            for (size_t j = 0, jend = vbInliers.size(); j < jend; ++j)
            {
                if (vbInliers[j])
                    vpMapPointMatches[j] = vvpMapPointMatches[i][j];
            }

            cv::Mat R = pSolver->GetEstimatedRotation();
            cv::Mat t = pSolver->GetEstimatedTranslation();
            float const s = pSolver->GetEstimatedScale();
            matcher.SearchBySim3(mpCurrentKF, pKF, vpMapPointMatches, s, R, t, 7.5);

            g2o::Sim3 gScm(Converter::toMatrix3d(R), Converter::toVector3d(t),s);
            int const nInliers2 = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

            // If optimization is succesful stop ransacs and continue
            if (nInliers2 >= 20)
            {
                bMatch = true;
                mpMatchedKF = pKF;
                g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()), Converter::toVector3d(pKF->GetTranslation()), 1.0);
                mg2oScw = gScm * gSmw;
                mScw = Converter::toCvMat(mg2oScw);
                mvpCurrentMatchedPoints = std::move(vpMapPointMatches);
                break;
            }
        }
    }

    if (!bMatch)
    {
        for (std::size_t i = 0; i < nInitialCandidates; ++i)
             mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    std::vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();

    for (KeyFrame* pKF : vpLoopConnectedKFs)
    {
        std::vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        for (MapPoint* pMP : vpMapPoints)
        {
            if (pMP == nullptr || pMP->isBad())
                continue;

            mvpLoopMapPoints.insert(pMP);
        }
    }

    // Find more matches projecting with the computed Sim3
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints, 10);

    // If enough matches accept Loop
    std::size_t nTotalMatches = 0;

    for (MapPoint* pMP : mvpCurrentMatchedPoints)
    {
        if (pMP != nullptr)
            ++nTotalMatches;
    }

    if (nTotalMatches >= 40)
    {
        for (std::size_t i = 0; i < nInitialCandidates; ++i)
            if(mvpEnoughConsistentCandidates[i] != mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }

    for (std::size_t i = 0; i < nInitialCandidates; ++i)
        mvpEnoughConsistentCandidates[i]->SetErase();

    mpCurrentKF->SetErase();

    return false;
}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // If a Global Bundle Adjustment is running, abort it
    if (isRunningGBA())
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);

        mbStopGBA = true;

        ++mnFullBAIdx;

        if (mpThreadGBA)
        {
            mpThreadGBA->detach();
            delete mpThreadGBA;
        }
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
    mvpCurrentConnectedKFs.push_back(mpCurrentKF);

    KeyFrameAndPose CorrectedSim3;
    KeyFrameAndPose NonCorrectedSim3;

    CorrectedSim3[mpCurrentKF] = mg2oScw;
    cv::Mat Twc = mpCurrentKF->GetPoseInverse();

    {
        // Get Map Mutex
        std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

        for (KeyFrame* pKFi : mvpCurrentConnectedKFs)
        {
            cv::Mat Tiw = pKFi->GetPose();

            if (pKFi != mpCurrentKF)
            {
                cv::Mat Tic = Tiw * Twc;
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                cv::Mat tic = Tic.rowRange(0,3).col(3);
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi] = g2oCorrectedSiw;
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
            cv::Mat tiw = Tiw.rowRange(0,3).col(3);
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi] = g2oSiw;
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for (auto mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; ++mit)
        {
            KeyFrame* pKFi = mit->first;
            g2o::Sim3 g2oCorrectedSiw = mit->second;
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

            std::vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();

            for (MapPoint* pMPi : vpMPsi)
            {
                if (pMPi == nullptr || pMPi->isBad() || pMPi->mnCorrectedByKF == mpCurrentKF->get_id())
                    continue;

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos();
                Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw);
                pMPi->mnCorrectedByKF = mpCurrentKF->get_id();
                pMPi->mnCorrectedReference = pKFi->get_id();
                pMPi->UpdateNormalAndDepth();
            }

            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
            double s = g2oCorrectedSiw.scale();

            eigt *=(1./s); //[R t/s;0 1]

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

            pKFi->SetPose(correctedTiw);

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); ++i)
        {
            MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
            if (pLoopMP == nullptr)
                continue;

            MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
            if (pCurMP != nullptr)
                pCurMP->Replace(pLoopMP);
            else
            {
                mpCurrentKF->AddMapPoint(pLoopMP, i);
                pLoopMP->AddObservation(mpCurrentKF, i);
                pLoopMP->ComputeDistinctiveDescriptors();
            }
        }
    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    SearchAndFuse(CorrectedSim3);

    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    std::unordered_map<KeyFrame*, std::unordered_set<KeyFrame*> > LoopConnections;

    for (KeyFrame* pKFi : mvpCurrentConnectedKFs)
    {
        std::vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

        // Update connections. Detect new links.
        pKFi->UpdateConnections();
        LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();

        auto& set_frames = LoopConnections[pKFi];

        for (KeyFrame* pKFN : vpPreviousNeighbors)
            set_frames.erase(pKFN);

        for (KeyFrame* pCCKF : mvpCurrentConnectedKFs)
            set_frames.erase(pCCKF);
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF,
        NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

    mpMap->InformNewBigChange();

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF);
    mpCurrentKF->AddLoopEdge(mpMatchedKF);

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;

    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this);

    mLastLoopKFid = mpCurrentKF->get_id();
}

void LoopClosing::SearchAndFuse(KeyFrameAndPose const& CorrectedPosesMap)
{
    ORBmatcher matcher(0.8, true);

    for (auto const& pair : CorrectedPosesMap)
    {
        KeyFrame* pKF = pair.first;
        g2o::Sim3 g2oScw = pair.second;

        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        std::unordered_map<MapPoint*, MapPoint*> vpReplacePoints;
        matcher.Fuse(pKF, cvScw, mvpLoopMapPoints, 4, vpReplacePoints);

        // Get Map Mutex
        std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);
        for (auto const& pair : vpReplacePoints)
            pair.second->Replace(pair.first);
    }
}

void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    for (;;)
    {
        {
            unique_lock<mutex> lock2(mMutexReset);

            if (!mbResetRequested)
                break;
        }

        std::this_thread::sleep_for(std::chrono::microseconds(5000));
    }
}

void LoopClosing::ResetIfRequested()
{
    std::unique_lock<std::mutex> lock(mMutexReset);

    if (mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid = 0;
        mbResetRequested = false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment()
{
    cout << "Starting Global Bundle Adjustment" << endl;

    uint64_t const idx = mnFullBAIdx;
    auto pair_GBA = Optimizer::GlobalBundleAdjustemnt(mpMap, 10, &mbStopGBA, false, false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        std::unique_lock<std::mutex> lock(mMutexGBA);

        if (idx != mnFullBAIdx)
            return;

        if (!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;

            // Get Map Mutex
            std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            std::vector<KeyFrame*> const& key_frame_origins = mpMap->get_key_frame_origins();
            std::list<KeyFrame*> lpKFtoCheck(key_frame_origins.begin(), key_frame_origins.end());

            std::unordered_map<KeyFrame*, cv::Mat> map_replaced; // PAE: TODO: think about it

            while (!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                cv::Mat Twc = pKF->GetPoseInverse();

                auto itKF = pair_GBA.first.find(pKF);
                assert(itKF != pair_GBA.first.end());
                cv::Mat const& matKF = itKF->second;

                std::unordered_set<KeyFrame*> const sChilds = pKF->GetChilds();

                for (auto sit = sChilds.cbegin(); sit != sChilds.cend(); ++sit)
                {
                    KeyFrame* pChild = *sit;

                    auto itChild = pair_GBA.first.find(pChild);
                    if (itChild == pair_GBA.first.end()) // TODO: PAE: this in fact should NEVER HAPEN
                        pair_GBA.first.emplace(pChild, pChild->GetPose() * Twc * matKF);

                    lpKFtoCheck.push_back(pChild);
                }

                auto pair_replaced = map_replaced.emplace(pKF, pKF->GetPose());
                assert(pair_replaced.second);
                (void)pair_replaced;

                pKF->SetPose(matKF);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            std::vector<MapPoint*> const vpMPs = mpMap->GetAllMapPoints();

            for (MapPoint* pMP : vpMPs)
            {
                if (pMP->isBad())
                    continue;

                auto itMP = pair_GBA.second.find(pMP);
                if (itMP != pair_GBA.second.end())
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(itMP->second);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    auto itKF = pair_GBA.first.find(pRefKF);
                    if (itKF == pair_GBA.first.end())
                        continue;

                    auto itKFReplaced = map_replaced.find(pRefKF);
                    assert(itKFReplaced != map_replaced.end());

                    // Map to non-corrected camera
                    cv::Mat Rcw = itKFReplaced->second.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = itKFReplaced->second.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
                    cv::Mat twc = Twc.rowRange(0, 3).col(3);

                    pMP->SetWorldPos(Rwc * Xc + twc);
                }
            }

            mpMap->InformNewBigChange();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

} //namespace ORB_SLAM
