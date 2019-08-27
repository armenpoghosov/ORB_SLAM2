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


#include "Tracking.h"

#include <opencv2/imgproc/types_c.h>

#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "MapDrawer.h"
#include "Initializer.h"
#include "System.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "PnPsolver.h"
#include "Viewer.h"

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer,
        MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, string const& strSettingPath, int sensor)
    :
    mState(NO_IMAGES_YET),
    mSensor(sensor),
    mbOnlyTracking(false),
    mbVO(false),
    mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB),
    mpInitializer(nullptr),
    mpSystem(pSys),
    mpViewer(nullptr),
    mpFrameDrawer(pFrameDrawer),
    mpMapDrawer(pMapDrawer),
    mpMap(pMap),
    mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4, 1, CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];

    float const k3 = fSettings["Camera.k3"];
    if (k3 != 0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }

    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if (fps == 0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    
    if (DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    mbRGB = (bool)(int)fSettings["Camera.RGB"];

    if (mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    if (sensor == System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2 * nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if (sensor == System::STEREO || sensor == System::RGBD)
    {
        mThDepth = mbf * (float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if (sensor == System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        mDepthMapFactor = std::fabs(mDepthMapFactor) < 1e-5f ? 1.f : (1.f / mDepthMapFactor);
    }

    m_ofs.open("C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\ORB_SLAM2\\tracking_dump.txt", ofstream::out | ofstream::trunc);
}

Tracking::~Tracking()
{
    m_ofs.close();
}

void Tracking::GrabImageStereo(cv::Mat const& imRectLeft, cv::Mat const& imRectRight, double timestamp)
{
    cv::Mat mImGray = imRectLeft;
    cv::Mat imGrayRight = imRectRight;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
        {
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
        }
        else
        {
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
            cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
        }
    }
    else if (mImGray.channels() == 4)
    {
        if(mbRGB)
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY);
        }
        else
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY);
        }
    }

    mCurrentFrame = Frame(mImGray, imGrayRight, timestamp,
        mpORBextractorLeft, mpORBextractorRight, mpORBVocabulary,
        mK, mDistCoef, mbf, mThDepth);

    Track();
}

void Tracking::GrabImageRGBD(cv::Mat const& imRGB, cv::Mat const& imD, double timestamp)
{
    cv::Mat mImGray = imRGB;
    cv::Mat imDepth = imD;

    if (mImGray.channels() == 3)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGB2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGR2GRAY);
    }
    else if (mImGray.channels() == 4)
    {
        if (mbRGB)
            cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
        else
            cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
    }

    if ((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray, imDepth, timestamp,
        mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    Track();
}

void Tracking::GrabImageMonocular(cv::Mat const& im, double timestamp)
{
    cv::Mat mImGray = im;

    if (mImGray.channels() == 3)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGB2GRAY : CV_BGR2GRAY);
    else if (mImGray.channels() == 4)
        cvtColor(mImGray, mImGray, mbRGB ? CV_RGBA2GRAY : CV_BGRA2GRAY);

    std::unique_ptr<Frame> frame(new Frame(mImGray, timestamp,
        (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) ? mpIniORBextractor : mpORBextractorLeft,
        mpORBVocabulary, mK, mDistCoef, mbf, mThDepth));

    if (m_future.valid())
        m_future.get();

    m_future = std::async(std::launch::async, &Tracking::track_worker, this, std::move(frame));
}

void Tracking::track_worker(std::unique_ptr<Frame> frame)
{
    mCurrentFrame = *frame.get();
    Track();
}

void Tracking::Track()
{
    if (mState == NO_IMAGES_YET)
        mState = NOT_INITIALIZED;

    mLastProcessedState = mState;

    // Get Map Mutex -> Map cannot be changed
    std::unique_lock<std::mutex> lock(mpMap->mMutexMapUpdate);

    if (mState == NOT_INITIALIZED)
    {
        if (mSensor == System::STEREO || mSensor == System::RGBD)
            StereoInitialization();
        else
            MonocularInitialization();

        mpFrameDrawer->Update(this);

        if (mState != OK)
            return;
    }
    else
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if (!mbOnlyTracking)
        {
            // Local Mapping is activated. This is the normal behaviour, unless you explicitly
            // activate the "only tracking" mode.
            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                mLastFrame.commit_replaced_map_points();

                if (mVelocity.empty() || (mCurrentFrame.get_id() < mnLastRelocFrameId + 2) ||
                    !(bOK = TrackWithMotionModel()))
                    bOK = TrackReferenceKeyFrame();
            }
            else
                bOK = Relocalization();
        }
        // Localization Mode: Local Mapping is deactivated
        else if (mState == LOST)
            bOK = Relocalization();
        else if (!mbVO) // In last frame we tracked enough MapPoints in the map
            bOK =  mVelocity.empty() ? TrackReferenceKeyFrame() : TrackWithMotionModel();
        else
        {
            // In last frame we tracked mainly "visual odometry" points.

            // We compute two camera poses, one from motion model and one doing relocalization.
            // If relocalization is sucessfull we choose that solution, otherwise we retain
            // the "visual odometry" solution.

            bool bOKMM = false;
            bool bOKReloc = false;

            cv::Mat TcwMM;

            vector<bool> vbOutMM;
            vector<MapPoint*> vpMPsMM;

            if (!mVelocity.empty())
            {
                bOKMM = TrackWithMotionModel();
                vpMPsMM = mCurrentFrame.mvpMapPoints;
                vbOutMM = mCurrentFrame.mvbOutlier;
                TcwMM = mCurrentFrame.mTcw.clone();
            }

            bOKReloc = Relocalization();

            if (bOKMM && !bOKReloc)
            {
                mCurrentFrame.SetPose(TcwMM);
                mCurrentFrame.mvpMapPoints = vpMPsMM;
                mCurrentFrame.mvbOutlier = vbOutMM;

                if (mbVO)
                {
                    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
                    {
                        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                        if (pMP != nullptr && !mCurrentFrame.mvbOutlier[i])
                            pMP->IncreaseFound();
                    }
                }
            }
            else if (bOKReloc)
            {
                mbVO = false;
            }

            bOK = bOKReloc || bOKMM;
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // 1) If we have an initial estimation of the camera pose and matching. Track the local map.
        // 2) mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
        //      a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
        //      the camera we will use the local map again.
        if (bOK && (!mbOnlyTracking || !mbVO))
            bOK = TrackLocalMap();

        mState = bOK ? OK : LOST;

        // Update drawer
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe
        if (bOK)
        {
            // Update motion model
            if  (!mLastFrame.mTcw.empty())
            {
                cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                mVelocity = mCurrentFrame.mTcw * LastTwc;
            }
            else
                mVelocity = cv::Mat();

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            // Clean VO matches
            for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
            {
                MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

                if (rpMP != nullptr && rpMP->Observations() == 0)
                {
                    mCurrentFrame.mvbOutlier[i] = false;
                    rpMP = nullptr;
                }
            }

            // Delete temporal MapPoints
            for (MapPoint* pMP : mlpTemporalPoints)
                delete pMP;
            mlpTemporalPoints.clear();

            // Check if we need to insert a new keyframe
            if (NeedNewKeyFrame())
                CreateNewKeyFrame();

            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
            {
                MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];
                if (rpMP != nullptr && mCurrentFrame.mvbOutlier[i])
                    rpMP = nullptr;
            }
        }

        // Reset if the camera get lost soon after initialization
        if (mState == LOST && mpMap->KeyFramesInMap() <= 5)
        {
            cout << "Track lost soon after initialisation, reseting..." << endl;
            mpSystem->Reset();
            return;
        }

        if (mCurrentFrame.mpReferenceKF == nullptr)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    }

    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if (!mCurrentFrame.mTcw.empty())
    {
        cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
        mlRelativeFramePoses.push_back(Tcr);
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.get_time_stamp());
    }
    else
    {
        // This can happen if tracking is lost
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
    }

    mlbLost.push_back(mState == LOST);
}


void Tracking::StereoInitialization()
{
    if (mCurrentFrame.get_frame_N() <= 500)
        return;

    // Set Frame pose to the origin
    mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));

    // Create KeyFrame
    KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        float const z = mCurrentFrame.mvDepth[i];
        if (z <= 0.f)
            continue;

        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
    }

    cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

    enqueue_key_frame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.insert(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPointsSet();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // PAE: this is done without any lock here!
    mpMap->add_to_key_frame_origins(pKFini);

    mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

    mState = OK;
}

void Tracking::MonocularInitialization()
{
    enum : int { MIN_KEY_COUNT_FOR_INITIALIZATOIN = 20 };

    std::size_t const size_vKeys = mCurrentFrame.get_key_points().size();

    if (mpInitializer != nullptr)
    {
        // Try to initialize
        if (size_vKeys <= MIN_KEY_COUNT_FOR_INITIALIZATOIN)
        {
            delete mpInitializer;
            mpInitializer = nullptr;
            mvIniMatches.clear();
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9f, true);

        // Check if there are enough correspondences
        if (matcher.SearchForInitialization(mInitialFrame, mCurrentFrame,
                mvbPrevMatched, mvIniMatches, 100) < MIN_KEY_COUNT_FOR_INITIALIZATOIN ||
            // NOTE: PAE: added by me
            mCurrentFrame.get_id() - mInitialFrame.get_id() > 3)
        {
            delete mpInitializer;
            mpInitializer = nullptr;
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        std::vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; ++i)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                    mvIniMatches[i] = -1;
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
    // Set Reference Frame
    else if (size_vKeys > MIN_KEY_COUNT_FOR_INITIALIZATOIN)
    {
        mInitialFrame = Frame(mCurrentFrame);
        mLastFrame = Frame(mCurrentFrame);

        std::size_t const size_vKeysUn = mCurrentFrame.mvKeysUn.size();

        mvbPrevMatched.resize(size_vKeysUn);

        for (size_t i = 0; i < size_vKeysUn; ++i)
            mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

        mpInitializer = new Initializer(mCurrentFrame, 1.f, 200);

        mvIniMatches = std::vector<int>(size_vKeysUn, -1);
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for (size_t i = 0; i < mvIniMatches.size(); ++i)
    {
        int const ini_index = mvIniMatches[i];
        if (ini_index < 0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

        pKFini->AddMapPoint(pMP, i);
        pKFcur->AddMapPoint(pMP, ini_index);

        pMP->AddObservation(pKFini, i);
        pMP->AddObservation(pKFcur, ini_index);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        // Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[ini_index] = pMP;
        mCurrentFrame.mvbOutlier[ini_index] = false;

        // Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20, nullptr, true, true);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    float invMedianDepth = 1.0f / medianDepth;

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3) * invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    std::vector<MapPoint*> const &vpAllMapPoints = pKFini->GetMapPointMatches();
    for (MapPoint* pMP : vpAllMapPoints)
    {
        if (pMP != nullptr)
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }

    enqueue_key_frame(pKFini);
    enqueue_key_frame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.insert(pKFcur);
    mvpLocalKeyFrames.insert(pKFini);
    
    mvpLocalMapPoints = mpMap->GetAllMapPointsSet();

    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    // PAE: this is done without any lock here!
    mpMap->add_to_key_frame_origins(pKFini);

    mState = OK;
}

bool Tracking::TrackReferenceKeyFrame()
{
    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7f, true);

    std::vector<MapPoint*> vpMapPointMatches;
    if (matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches) < 15)
        return false;

    mCurrentFrame.mvpMapPoints = std::move(vpMapPointMatches);
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame, 1);

    // Discard outliers
    std::size_t nmatchesMap = 0;
    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        if (mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvbOutlier[i] = false;
            rpMP->mbTrackInView = false;
            rpMP->mnLastFrameSeen = mCurrentFrame.get_id();
            rpMP = nullptr;
        }
        else if (rpMP->Observations() > 0)
        {
            ++nmatchesMap;
        }
    }

    return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    mLastFrame.SetPose(mlRelativeFramePoses.back() * mLastFrame.mpReferenceKF->GetPose());

    if (mSensor == System::MONOCULAR || mnLastKeyFrameId == mLastFrame.get_id() || !mbOnlyTracking)
        return;

    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    std::vector<std::pair<float, int> > vDepthIdx;
    vDepthIdx.reserve(mLastFrame.get_frame_N());

    for (int i = 0; i < mLastFrame.get_frame_N(); ++i)
    {
        float z = mLastFrame.mvDepth[i];
        if (z > 0)
            vDepthIdx.emplace_back(z, i);
    }

    if (vDepthIdx.empty())
        return;

    sort(vDepthIdx.begin(),vDepthIdx.end());

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for (size_t j = 0; j < vDepthIdx.size(); ++j)
    {
        int i = vDepthIdx[j].second;

        MapPoint*& rpMP = mLastFrame.mvpMapPoints[i];

        if (rpMP == nullptr || rpMP->Observations() < 1)
        {
            cv::Mat x3D = mLastFrame.UnprojectStereo(i);

            rpMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

            mlpTemporalPoints.push_back(rpMP);
        }

        ++nPoints;

        if (vDepthIdx[j].first > mThDepth && nPoints > 100)
            break;
    }
}

bool Tracking::TrackWithMotionModel()
{
    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // Project points seen in previous frame
    float const th = mSensor != System::STEREO ? 15.f : 7.f;

    // If few matches, uses a wider window search
    ORBmatcher matcher(0.9f, true);

    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
    std::size_t nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2.f * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame, 2);

    // Discard outliers
    std::size_t nmatchesMap = 0;
    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        if (mCurrentFrame.mvbOutlier[i])
        {
            mCurrentFrame.mvbOutlier[i] = false;
            rpMP->mbTrackInView = false;
            rpMP->mnLastFrameSeen = mCurrentFrame.get_id();
            rpMP = nullptr;
            --nmatches;
        }
        else if (rpMP->Observations() != 0)
            ++nmatchesMap;
    }

    if (mbOnlyTracking)
    {
        mbVO = nmatchesMap < 10;
        return nmatches > 20;
    }

    return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap();

    SearchLocalPoints();

    if (mLastFrame.get_id() == 600)
    {
        m_ofs << "--------------------------------------------------------------------------" << std::endl;
        m_ofs << "LocalMap points" << std::endl;

        std::vector<MapPoint*> lmp;
        lmp.reserve(mvpLocalMapPoints.size());
        lmp.insert(lmp.end(), mvpLocalMapPoints.begin(), mvpLocalMapPoints.end());
        std::sort(lmp.begin(), lmp.end(),
            [] (MapPoint const* p1, MapPoint const* p2) -> bool
            {
                return p1->get_id() < p2->get_id();
            });

        for (MapPoint* pMP : lmp)
        {
            if (pMP == nullptr)
            {
                m_ofs << "nullptr" << std::endl;
                continue;
            }

            m_ofs << pMP->get_id() << ',' << pMP->isBad() << ',' << pMP->mnLastFrameSeen << std::endl;

            uint32_t const* const ddata = pMP->GetDescriptor().ptr<uint32_t>();
            m_ofs << ddata[0] << '|' << ddata[1] << '|' << ddata[2] << '|' << ddata[3] << '|' <<
                ddata[4] << '|' << ddata[5] << '|' << ddata[6] << '|' << ddata[7] << std::endl;

            m_ofs << pMP->GetWorldPos() << std::endl;
        }
    }

    if (mLastFrame.get_id() == 600)
    {
        m_ofs << "--------------------------------------------------------------------------" << std::endl;
        m_ofs << "Frame Id: " << mLastFrame.get_id() << std::endl;
        m_ofs << "KeyPoints: -----------------------------" << std::endl;

        for (cv::KeyPoint const& kp : mLastFrame.get_key_points())
        {
            m_ofs << kp.pt.x << ',' << kp.pt.y << ',' << kp.angle << ',' <<
                kp.response << ',' << kp.octave << ',' << kp.size << std::endl;
        }

        m_ofs << "MapPoints: -----------------------------" << std::endl;

        for (std::size_t i = 0; i < mLastFrame.get_frame_N(); ++i)
        {
            MapPoint*& rpMP = mLastFrame.mvpMapPoints[i];

            if (rpMP == nullptr)
            {
                m_ofs << "nullptr" << std::endl;
                continue;
            }

            m_ofs << mLastFrame.mvbOutlier[i] << ',' << rpMP->isBad() << ',' << std::endl;

            uint32_t const* const ddata = mLastFrame.get_descriptor(i).ptr<uint32_t>();
            m_ofs << ddata[0] << '|' << ddata[1] << '|' << ddata[2] << '|' << ddata[3] << '|' <<
                ddata[4] << '|' << ddata[5] << '|' << ddata[6] << '|' << ddata[7] << std::endl;

            m_ofs << rpMP->GetWorldPos() << std::endl;
        }

        m_ofs << "--------------------------------------------------------------------------" << std::endl;
        m_ofs << "Frame Id: " << mCurrentFrame.get_id() << std::endl;
        m_ofs << "KeyPoints: -----------------------------" << std::endl;

        for (cv::KeyPoint const& kp : mCurrentFrame.get_key_points())
        {
            m_ofs << kp.pt.x << ',' << kp.pt.y << ',' << kp.angle << ',' <<
                kp.response << ',' << kp.octave << ',' << kp.size << std::endl;
        }

        m_ofs << "--------------------------------------------------" << std::endl;
        for (int i = 0; i < Frame::FRAME_GRID_COLS; ++i)
            for (int j = 0; j < Frame::FRAME_GRID_ROWS; ++j)
            {
                for (std::size_t index : mCurrentFrame.mGrid[i][j])
                {
                    m_ofs << index << '|';
                }
                m_ofs << std::endl;
            }
        m_ofs << std::endl << "--------------------------------------------------" << std::endl;

        m_ofs << "MapPoints: -----------------------------" << std::endl;

        for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
        {
            MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

            if (rpMP == nullptr)
            {
                m_ofs << "nullptr" << std::endl;
                continue;
            }

            m_ofs << mCurrentFrame.mvbOutlier[i] << ',' << rpMP->get_id() << ',' << rpMP->isBad() << ',' << std::endl;

            uint32_t const* const ddata = mCurrentFrame.get_descriptor(i).ptr<uint32_t>();
            m_ofs << ddata[0] << '|' << ddata[1] << '|' << ddata[2] << '|' << ddata[3] << '|' <<
                ddata[4] << '|' << ddata[5] << '|' << ddata[6] << '|' << ddata[7] << std::endl;

            m_ofs << rpMP->GetWorldPos() << std::endl;
        }

        /*
        m_ofs << "--------------------------------------------------------------------------" << std::endl;
        m_ofs << "KeyFrame Id: " << pKF->get_id() << std::endl;
        m_ofs << "MapPoints: -----------------------------" << std::endl;

        std::vector<MapPoint*> const& map_points = pKF->GetMapPointMatches();
        for (MapPoint* pMP : map_points)
        {
            if (pMP == nullptr)
            {
                m_ofs << "nullptr" << std::endl;
                continue;
            }

            m_ofs << pMP->isBad() << std::endl;
            uint32_t const* const ddata = pMP->GetDescriptor().ptr<uint32_t>();
            m_ofs << ddata[0] << '|' << ddata[1] << '|' << ddata[2] << '|' << ddata[3] << '|' <<
                ddata[4] << '|' << ddata[5] << '|' << ddata[6] << '|' << ddata[7] << std::endl;
            m_ofs << pMP->GetWorldPos() << std::endl;
        }*/

        m_ofs.close();
        abort();
    }

    Optimizer::PoseOptimization(&mCurrentFrame, 3);

    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        if (!mCurrentFrame.mvbOutlier[i])
        {
            rpMP->IncreaseFound();

            if (mbOnlyTracking || rpMP->Observations() != 0)
                ++mnMatchesInliers;
        }
        else if (mSensor == System::STEREO)
            rpMP = nullptr;
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    return mnMatchesInliers >= ((mCurrentFrame.get_id() < mnLastRelocFrameId + mMaxFrames) ? 50 : 30);
}

bool Tracking::NeedNewKeyFrame()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mbOnlyTracking)
        return false;

    std::size_t const nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.get_id() < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    std::size_t const nRefMatches = mpReferenceKF->TrackedMapPoints(nKFs <= 2 ? 2 : 3);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = true;

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nTrackedClose = 0;
    int nNonTrackedClose = 0;

    if (mSensor != System::MONOCULAR)
    {
        for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
        {
            if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth)
            {
                if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                    ++nTrackedClose;
                else
                    ++nNonTrackedClose;
            }
        }
    }

    bool const bNeedToInsertClose = nTrackedClose < 100 && nNonTrackedClose > 70;

    // Thresholds
    float const thRefRatio = mSensor == System::MONOCULAR ? 0.9f : (nKFs < 2 ? 0.4f : 0.75f);

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    bool const c1a = mCurrentFrame.get_id() >= mnLastKeyFrameId + mMaxFrames;

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    bool const c1b = mCurrentFrame.get_id() >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle;

    // Condition 1c: tracking is weak
    bool const c1c =  mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    bool const c2 = (mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15;

    return (c1a || c1b || c1c) && c2;
}

void Tracking::CreateNewKeyFrame()
{
    KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    if (mSensor != System::MONOCULAR)
    {
        mCurrentFrame.UpdatePoseMatrices();

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        std::vector<std::pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mCurrentFrame.get_frame_N());

        for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
        {
            float const z = mCurrentFrame.mvDepth[i];
            if (z > 0.f)
                vDepthIdx.emplace_back(z, i);
        }

        if(!vDepthIdx.empty())
        {
            std::sort(vDepthIdx.begin(), vDepthIdx.end());

            int nPoints = 0;

            for (std::size_t j = 0; j < vDepthIdx.size(); ++j)
            {
                int i = vDepthIdx[j].second;

                bool bCreateNew = false;

                MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

                if (rpMP == nullptr)
                    bCreateNew = true;
                else if (rpMP->Observations() < 1)
                {
                    bCreateNew = true;
                    rpMP = nullptr;
                }

                if (bCreateNew)
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);

                    MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    pKF->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    mpMap->AddMapPoint(pNewMP);

                    // PAE: is CurrentFrame mvpMapPoints vector modified above this point?
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }

                ++nPoints;

                if (vDepthIdx[j].first>mThDepth && nPoints>100)
                    break;
            }
        }
    }

    enqueue_key_frame(pKF);

    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // do not search map points already matched
    for (MapPoint*& rpMP : mCurrentFrame.get_map_points())
    {
        if (rpMP == nullptr)
            continue;

        if (rpMP->isBad())
        {
            rpMP = nullptr;
            continue;
        }

        rpMP->IncreaseVisible();
        rpMP->mnLastFrameSeen = mCurrentFrame.get_id();
        rpMP->mbTrackInView = false;
    }

    std::size_t nToMatch = 0;

    // Project points in frame and check its visibility
    for (MapPoint* pMP : mvpLocalMapPoints)
    {
        pMP->mbTrackInView = false;

        if (pMP->mnLastFrameSeen == mCurrentFrame.get_id() || pMP->isBad())
            continue;

        // Project (this fills MapPoint variables for matching)
        if (mCurrentFrame.isInFrustum(pMP, 0.5))
        {
            pMP->IncreaseVisible();
            ++nToMatch;
        }
    }

    if (nToMatch > 0)
    {
        ORBmatcher matcher(0.8f, true);

        // If the camera has been relocalised recently, perform a coarser search
        float const th = (mCurrentFrame.get_id() < mnLastRelocFrameId + 2) ?
            5.f : (mSensor == System::RGBD ? 1.f : 3.f);
        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th, &m_ofs);
    }
}

void Tracking::UpdateLocalMap()
{
    // This is for visualization
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    // Update
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    for (KeyFrame* pKF : mvpLocalKeyFrames)
    {
        std::vector<MapPoint*> const& vpMPs = pKF->GetMapPointMatches();

        for (MapPoint* pMP : vpMPs)
        {
            if (pMP == nullptr || pMP->isBad())
                continue;

            mvpLocalMapPoints.insert(pMP);
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    mvpLocalKeyFrames.clear();

    // Each map point vote for the keyframes in which it has been observed
    std::unordered_map<KeyFrame*, std::size_t> keyframeCounter;

    std::vector<MapPoint*>& frame_map_points = mCurrentFrame.get_map_points();
    for (MapPoint*& rpMP : frame_map_points)
    {
        if (rpMP == nullptr)
            continue;

        if (rpMP->isBad())
        {
            rpMP = nullptr;
            continue;
        }

        std::unordered_map<KeyFrame*, size_t> const& observations = rpMP->GetObservations();
        for (auto const& pair : observations)
            ++keyframeCounter[pair.first];
    }

    if (keyframeCounter.empty())
        return;

    std::size_t max = 0;
    KeyFrame* pKFmax = nullptr;

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points
    struct TPair : std::pair<KeyFrame const*, TPair*> {};

    TPair* head = nullptr;

    std::unordered_map<KeyFrame*, TPair*> map;
    map.reserve(3 * keyframeCounter.size());

    for (auto const& pair : keyframeCounter)
    {
        KeyFrame* pKF = pair.first;

        if (pKF->isBad())
            continue;

        if (pair.second > max ||
            (pair.second == max && pKF->get_id() < pKFmax->get_id()))
        {
            max = pair.second;
            pKFmax = pKF;
        }

        auto const& pair_emplaced = map.emplace(pKF, head);
        assert(pair_emplaced.second);
        head = reinterpret_cast<TPair*>(pair_emplaced.first.operator->());
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (TPair* head_next = nullptr;;)
    {
        std::vector<KeyFrame*> const& neighbour_key_frames = head->first->GetBestCovisibilityKeyFrames(10);
        for (KeyFrame* pNeighKF : neighbour_key_frames)
        {
            if (pNeighKF->isBad())
                continue;

            auto const& pair_emplaced = map.emplace(pNeighKF, head_next);
            if (pair_emplaced.second)
                head_next = reinterpret_cast<TPair*>(pair_emplaced.first.operator->());
        }

        std::unordered_set<KeyFrame*> const& spChilds = head->first->GetChilds();
        for (KeyFrame* pChildKF : spChilds)
        {
            if (pChildKF->isBad())
                continue;

            auto const& pair_emplaced = map.emplace(pChildKF, head_next);
            if (pair_emplaced.second)
                head_next = reinterpret_cast<TPair*>(pair_emplaced.first.operator->());
        }

        KeyFrame* pParent = head->first->GetParent();
        if (pParent != nullptr)
        {
            auto const& pair_emplaced = map.emplace(pParent, head_next);
            if (pair_emplaced.second)
                head_next = reinterpret_cast<TPair*>(pair_emplaced.first.operator->());
        }

        if ((head = head->second) != nullptr)
            continue;

        if (map.size() >= 80 || (head = head_next) == nullptr)
            break;

        head_next = nullptr;
    }

    mvpLocalKeyFrames.reserve(map.size());

    for (auto const& pair : map)
        mvpLocalKeyFrames.insert(pair.first);

    if (pKFmax != nullptr)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    std::vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    if (vpCandidateKFs.empty())
        return false;

    std::size_t const nKFs = vpCandidateKFs.size();

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.75f, true);

    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs);

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nKFs);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nKFs);

    int nCandidates = 0;

    for (std::size_t i = 0; i < nKFs; ++i)
    {
        KeyFrame* pKF = vpCandidateKFs[i];

        if (pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

        if (matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]) < 15)
        {
            vbDiscarded[i] = true;
            continue;
        }

        PnPsolver* pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5f, 5.991f);
        vpPnPsolvers[i] = pSolver;
        ++nCandidates;
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9f, true);

    while (nCandidates > 0 && !bMatch)
    {
        for (std::size_t i = 0; i < nKFs; ++i)
        {
            if(vbDiscarded[i])
                continue;

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i];
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
            if (bNoMore)
            {
                vbDiscarded[i] = true;
                --nCandidates;
            }

            // If a Camera Pose is computed, optimize
            if (!Tcw.empty())
            {
                Tcw.copyTo(mCurrentFrame.mTcw);

                std::unordered_set<MapPoint*> sFound;

                std::size_t const np = vbInliers.size();

                for (std::size_t j = 0; j < np; ++j)
                {
                    MapPoint*& rCFMP = mCurrentFrame.mvpMapPoints[j];

                    if (vbInliers[j])
                        sFound.insert(rCFMP = vvpMapPointMatches[i][j]);
                    else
                        rCFMP = nullptr;
                }

                std::size_t nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.get_frame_N(); ++io)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = nullptr;

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                    if (nadditional + nGood >= 50)
                    {
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if (nGood > 30 && nGood < 50)
                        {
                            sFound.clear();
                            for (int ip = 0; ip < mCurrentFrame.get_frame_N(); ++ip)
                                if (mCurrentFrame.mvpMapPoints[ip])
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]);

                            nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

                            // Final optimization
                            if (nGood + nadditional >= 50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame);

                                for (int io = 0; io < mCurrentFrame.get_frame_N(); ++io)
                                    if (mCurrentFrame.mvbOutlier[io])
                                        mCurrentFrame.mvpMapPoints[io] = nullptr;
                            }
                        }
                    }
                }

                // If the pose is supported by enough inliers stop ransacs and continue
                if (nGood >= 50)
                {
                    bMatch = true;
                    break;
                }
            }
        }
    }

    if (!bMatch)
        return false;

    mnLastRelocFrameId = mCurrentFrame.get_id();

    return true;
}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;

    // TODO: PAE: threading
    if (m_future.valid())
        m_future.get();

    if (mpViewer != nullptr)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mlpRecentAddedMapPoints.clear(); // PAE: threading off
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset();
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear();
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear();

    KeyFrame::reset_id_counter();
    Frame::reset_id_counter();

    mState = NO_IMAGES_YET;

    if (mpInitializer != nullptr)
    {
        delete mpInitializer;
        mpInitializer = nullptr;
    }

    mlRelativeFramePoses.clear();
    mlpReferences.clear();
    mlFrameTimes.clear();
    mlbLost.clear();

    if (mpViewer != nullptr)
        mpViewer->Release();
}

void Tracking::ChangeCalibration(string const& strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

// PAE: functions moved from local mapping

void Tracking::ProcessNewKeyFrame()
{
    // Associate MapPoints to the new keyframe and update normal and descriptor
    std::vector<MapPoint*> const& vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (size_t i = 0, iend = vpMapPointMatches.size(); i < iend; ++i)
    {
        MapPoint* pMP = vpMapPointMatches[i];

        if (pMP == nullptr || pMP->isBad())
            continue;

        if (pMP->AddObservation(mpCurrentKeyFrame, i))
        {
            pMP->UpdateNormalAndDepth();
            pMP->ComputeDistinctiveDescriptors();
        }
        else // this can only happen for new stereo points inserted by the Tracking
        {
            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections();

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame);
}

void Tracking::MapPointCulling()
{
    // Check Recent Added MapPoints

    uint64_t const nCurrentKFid = mpCurrentKeyFrame->get_id();

    std::size_t const cnThObs = mSensor == System::MONOCULAR ? 2 : 3;

    for (auto it = mlpRecentAddedMapPoints.begin(); it != mlpRecentAddedMapPoints.end();)
    {
        MapPoint* pMP = *it;

        if (pMP->isBad())
            it = mlpRecentAddedMapPoints.erase(it);
        else if (pMP->GetFoundRatio() < 0.25f ||
            (nCurrentKFid >= pMP->get_first_key_frame_id() + 2 && pMP->Observations() <= cnThObs))
        {
            pMP->SetBadFlag();
            it = mlpRecentAddedMapPoints.erase(it);
        }
        else if (nCurrentKFid >= pMP->get_first_key_frame_id() + 3)
            it = mlpRecentAddedMapPoints.erase(it);
        else
            ++it;
    }
}

void Tracking::CreateNewMapPoints()
{
    ORBmatcher matcher(0.6f, false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation();
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation();

    cv::Mat Tcw1(3, 4, CV_32F);
    Rcw1.copyTo(Tcw1.colRange(0, 3));
    tcw1.copyTo(Tcw1.col(3));

    cv::Mat Rwc1 = Rcw1.t();
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter();

    float const fx1 = mpCurrentKeyFrame->fx;
    float const fy1 = mpCurrentKeyFrame->fy;

    float const cx1 = mpCurrentKeyFrame->cx;
    float const cy1 = mpCurrentKeyFrame->cy;

    float const invfx1 = mpCurrentKeyFrame->invfx;
    float const invfy1 = mpCurrentKeyFrame->invfy;

    float const ratioFactor = 1.5f * mpCurrentKeyFrame->mfScaleFactor;

    // Retrieve neighbor keyframes in covisibility graph
    std::vector<KeyFrame*> const& vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(mSensor == System::MONOCULAR ? 20 : 10);

    // Search matches with epipolar restriction and triangulate
    for (KeyFrame* pKF2 : vpNeighKFs)
    {
        // Check first that baseline is not too short
        cv::Mat Ow2 = pKF2->GetCameraCenter();

        double const baseline_length = cv::norm(Ow2 - Ow1);

        if ((mSensor != System::MONOCULAR) ?
            (baseline_length < pKF2->mb) :
            (baseline_length / pKF2->ComputeSceneMedianDepth(2) < 0.01))
            continue;

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame, pKF2);

        // Search matches that fullfil epipolar constraint
        std::vector<std::pair<size_t, size_t> > vMatchedIndices;
        matcher.SearchForTriangulation(mpCurrentKeyFrame, pKF2, F12, vMatchedIndices, false);

        cv::Mat Rcw2 = pKF2->GetRotation();
        cv::Mat Rwc2 = Rcw2.t();
        cv::Mat tcw2 = pKF2->GetTranslation();

        cv::Mat Tcw2(3, 4, CV_32F);
        Rcw2.copyTo(Tcw2.colRange(0, 3));
        tcw2.copyTo(Tcw2.col(3));

        float const fx2 = pKF2->fx;
        float const fy2 = pKF2->fy;

        float const cx2 = pKF2->cx;
        float const cy2 = pKF2->cy;

        float const invfx2 = pKF2->invfx;
        float const invfy2 = pKF2->invfy;

        // Triangulate each match
        for (auto const& match_pair : vMatchedIndices)
        {
            size_t const idx1 = match_pair.first;
            size_t const idx2 = match_pair.second;

            cv::KeyPoint const& kp1 = mpCurrentKeyFrame->mvKeysUn[idx1];
            float const kp1_ur = mpCurrentKeyFrame->mvuRight[idx1];
            bool const bStereo1 = kp1_ur >= 0;

            cv::KeyPoint const& kp2 = pKF2->mvKeysUn[idx2];
            float const kp2_ur = pKF2->mvuRight[idx2];
            bool const bStereo2 = kp2_ur >= 0;

            // Check parallax between rays
            cv::Mat xn1 = (cv::Mat_<float>(3, 1) << (kp1.pt.x - cx1) * invfx1, (kp1.pt.y - cy1) * invfy1, 1.f);
            cv::Mat xn2 = (cv::Mat_<float>(3, 1) << (kp2.pt.x - cx2) * invfx2, (kp2.pt.y - cy2) * invfy2, 1.f);

            cv::Mat ray1 = Rwc1 * xn1;
            cv::Mat ray2 = Rwc2 * xn2;
            float const cosParallaxRays = (float)(ray1.dot(ray2) / (cv::norm(ray1) * cv::norm(ray2)));

            float cosParallaxStereo = cosParallaxRays + 1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if (bStereo1)
                cosParallaxStereo1 = std::cos(2 * std::atan2(mpCurrentKeyFrame->mb / 2, mpCurrentKeyFrame->mvDepth[idx1]));
            else if (bStereo2)
                cosParallaxStereo2 = std::cos(2 * std::atan2(pKF2->mb / 2, pKF2->mvDepth[idx2]));

            cosParallaxStereo = (std::min)(cosParallaxStereo1, cosParallaxStereo2);

            cv::Mat x3D;
            if (cosParallaxRays > 0.f && cosParallaxRays < cosParallaxStereo &&
                (bStereo1 || bStereo2 || cosParallaxRays < 0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4, 4, CV_32F);
                A.row(0) = xn1.at<float>(0) * Tcw1.row(2) - Tcw1.row(0);
                A.row(1) = xn1.at<float>(1) * Tcw1.row(2) - Tcw1.row(1);
                A.row(2) = xn2.at<float>(0) * Tcw2.row(2) - Tcw2.row(0);
                A.row(3) = xn2.at<float>(1) * Tcw2.row(2) - Tcw2.row(1);

                cv::Mat w;
                cv::Mat u;
                cv::Mat vt;
                cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if (x3D.at<float>(3) == 0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
            }
            else if (bStereo1 && cosParallaxStereo1 < cosParallaxStereo2)
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);
            else if (bStereo2 && cosParallaxStereo2 < cosParallaxStereo1)
                x3D = pKF2->UnprojectStereo(idx2);
            else
                continue; // No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            // Check triangulation in front of cameras
            float const z1 = Rcw1.row(2).dot(x3Dt) + tcw1.at<float>(2);
            if (z1 <= 0.f)
                continue;

            float const z2 = Rcw2.row(2).dot(x3Dt) + tcw2.at<float>(2);
            if (z2 <= 0.f)
                continue;

            //Check reprojection error in first keyframe
            float const sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            float const x1 = Rcw1.row(0).dot(x3Dt) + tcw1.at<float>(0);
            float const y1 = Rcw1.row(1).dot(x3Dt) + tcw1.at<float>(1);
            float const invz1 = 1.f / z1;

            if (!bStereo1)
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float v1 = fy1 * y1 * invz1 + cy1;

                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;

                if ((errX1 * errX1 + errY1 * errY1) > 5.991 * sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1 * x1 * invz1 + cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf * invz1;
                float v1 = fy1 * y1 * invz1 + cy1;

                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;

                if ((errX1 * errX1 + errY1 * errY1 + errX1_r * errX1_r) > 7.8f * sigmaSquare1)
                    continue;
            }

            // Check reprojection error in second keyframe
            float const sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            float const x2 = Rcw2.row(0).dot(x3Dt) + tcw2.at<float>(0);
            float const y2 = Rcw2.row(1).dot(x3Dt) + tcw2.at<float>(1);
            float const invz2 = 1.f / z2;

            if (!bStereo2)
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float v2 = fy2 * y2 * invz2 + cy2;

                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;

                if ((errX2 * errX2 + errY2 * errY2) > 5.991 * sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2 * x2 * invz2 + cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf * invz2;
                float v2 = fy2 * y2 * invz2 + cy2;

                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;

                if ((errX2 * errX2 + errY2 * errY2 + errX2_r * errX2_r) > 7.8f * sigmaSquare2)
                    continue;
            }

            // Check scale consistency
            cv::Mat normal1 = x3D - Ow1;
            double const dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D - Ow2;
            double const dist2 = cv::norm(normal2);

            if (dist1 == 0. || dist2 == 0.)
                continue;

            float const ratioDist = (float)(dist2 / dist1);
            float const ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave] / pKF2->mvScaleFactors[kp2.octave];

            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D, mpCurrentKeyFrame, mpMap);

            pMP->AddObservation(mpCurrentKeyFrame, idx1);
            pMP->AddObservation(pKF2, idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP, idx1);
            pKF2->AddMapPoint(pMP, idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);

            mlpRecentAddedMapPoints.push_back(pMP);
        }
    }
}

void Tracking::SearchInNeighbors()
{
    std::vector<KeyFrame*> const& vpNeighKFs =
        mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(mSensor == System::MONOCULAR ? 20 : 10);

    std::unordered_set<KeyFrame*> target_key_frames;

    for (KeyFrame* pKFCV : vpNeighKFs)
    {
        if (pKFCV->isBad())
            continue;

        auto const& pair = target_key_frames.emplace(pKFCV);
        if (!pair.second)
            continue;

        std::vector<KeyFrame*> const& neighbours2 = pKFCV->GetBestCovisibilityKeyFrames(5);
        for (KeyFrame* pKFCV2 : neighbours2)
        {
            if (pKFCV2 == mpCurrentKeyFrame || pKFCV2->isBad())
                continue;

            target_key_frames.emplace(pKFCV);
        }
    }

    // TODO: PAE: removing non-determinizm introduced my unordered set here
    std::vector<KeyFrame*> target_key_frames_vector;
    target_key_frames_vector.reserve(target_key_frames.size());
    target_key_frames_vector.insert(target_key_frames_vector.end(), target_key_frames.begin(), target_key_frames.end());
    std::sort(target_key_frames_vector.begin(), target_key_frames_vector.end(),
        [] (KeyFrame const* p1, KeyFrame const* p2)->bool
        {
            return p1->get_id() < p2->get_id();
        });

    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher(0.6f, true);

    std::vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    // TODO: PAE: think about removing the vector in the call above and turning it to a set
    std::unordered_set<MapPoint*> fuse_candidates;
    fuse_candidates.insert(vpMapPointMatches.begin(), vpMapPointMatches.end());
    fuse_candidates.erase(nullptr);

    for (KeyFrame* pTKF : target_key_frames_vector)
        matcher.Fuse(pTKF, fuse_candidates, 0.3f);

    // Search matches by projection from target KFs in current KF
    fuse_candidates.clear();
    fuse_candidates.reserve(target_key_frames.size() * vpMapPointMatches.size());

    for (KeyFrame* pTKF : target_key_frames_vector)
    {
        std::vector<MapPoint*> const& vpMapPointsKFi = pTKF->GetMapPointMatches();

        for (MapPoint* pMP : vpMapPointsKFi)
        {
            if (pMP == nullptr || pMP->isBad())
                continue;

            fuse_candidates.insert(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame, fuse_candidates, 0.3f);

    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();

    for (MapPoint* pMP : vpMapPointMatches)
    {
        if (pMP == nullptr || pMP->isBad())
            continue;

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat Tracking::ComputeF12(KeyFrame* pKF1, KeyFrame* pKF2)
{
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    cv::Mat R12 = R1w * R2w.t();
    cv::Mat t12 = -R1w * R2w.t() * t2w + t1w;

    cv::Mat t12x = SkewSymmetricMatrix(t12);

    cv::Mat const& K1 = pKF1->mK;
    cv::Mat const& K2 = pKF2->mK;

    return K1.t().inv() * t12x * R12 * K2.inv();
}

void Tracking::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    std::vector<KeyFrame*> const& vpLocalKFs = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    int const thObs = 3;

    for (KeyFrame* pKF : vpLocalKFs)
    {
        if (pKF->get_id() == 0)
            continue;

        std::size_t nMPs = 0;
        std::size_t nRedundantObservations = 0;

        std::vector<MapPoint*> const& vpMapPoints = pKF->GetMapPointMatches();

        for (size_t i = 0, iend = vpMapPoints.size(); i < iend; ++i)
        {
            MapPoint* pMP = vpMapPoints[i];

            if (pMP == nullptr || pMP->isBad() ||
                (mSensor != System::MONOCULAR && (pKF->mvDepth[i] > pKF->mThDepth || pKF->mvDepth[i] < 0)))
                continue;

            ++nMPs;

            if (pMP->Observations() <= thObs)
                continue;

            int const octave = pKF->mvKeysUn[i].octave;

            std::unordered_map<KeyFrame*, size_t> const& observations = pMP->GetObservations();

            int nObs = 0;

            for (auto const& pair : observations)
            {
                if (pair.first == pKF || pair.first->isBad())
                    continue;

                if (pair.first->mvKeysUn[pair.second].octave <= octave + 1 && ++nObs >= thObs)
                {
                    ++nRedundantObservations;
                    break;
                }
            }
        }

        if (nRedundantObservations > 0.9 * nMPs)
            pKF->SetBadFlag();
    }
}

cv::Mat Tracking::SkewSymmetricMatrix(cv::Mat const& v)
{
    return (cv::Mat_<float>(3, 3) <<
        0, -v.at<float>(2), v.at<float>(1),
        v.at<float>(2), 0, -v.at<float>(0),
        -v.at<float>(1), v.at<float>(0), 0);
}

void Tracking::enqueue_key_frame(KeyFrame* pKF)
{
    mpCurrentKeyFrame = pKF;

    // BoW conversion and insertion in Map
    ProcessNewKeyFrame();

    // Check recent MapPoints
    MapPointCulling();

    // Triangulate new MapPoints
    CreateNewMapPoints();

    // Find more matches in neighbor keyframes and fuse point duplications
    SearchInNeighbors();

    // Local BA
    if (mpMap->KeyFramesInMap() > 2)
        Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame, nullptr, mpMap);

    // Check redundant local Keyframes
    KeyFrameCulling();

    // PAE: no loop closing for now
    // mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
}

} //namespace ORB_SLAM
