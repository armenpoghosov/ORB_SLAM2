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
#include "Initializer.h"
#include "System.h"
#include "Optimizer.h"
#include "ORBmatcher.h"
#include "PnPsolver.h"

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

    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

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

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

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
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode.
            if (mState == OK)
            {
                // Local Mapping might have changed some MapPoints tracked in last frame
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.get_id() < mnLastRelocFrameId + 2)
                    bOK = TrackReferenceKeyFrame();
                else if (!(bOK = TrackWithMotionModel()))
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
            for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
            {
                MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

                if (rpMP != nullptr && rpMP->Observations() < 1)
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
            for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
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

    mpLocalMapper->enqueue_key_frame(pKFini);

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
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
        int nmatches = matcher.SearchForInitialization(mInitialFrame,
            mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

        // Check if there are enough correspondences
        if (nmatches < MIN_KEY_COUNT_FOR_INITIALIZATOIN ||
            // NOTE: PAE: added by me
            mCurrentFrame.get_id() - mInitialFrame.get_id() > 3)
        {
            delete mpInitializer;
            mpInitializer = nullptr;
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if (mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for (size_t i = 0, iend = mvIniMatches.size(); i < iend; ++i)
            {
                if (mvIniMatches[i] >= 0 && !vbTriangulated[i])
                {
                    mvIniMatches[i] = -1;
                    --nmatches;
                }
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

    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

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

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[ini_index] = pMP;
        mCurrentFrame.mvbOutlier[ini_index] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

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
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    std::vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for (MapPoint* pMP : vpAllMapPoints)
    {
        if (pMP != nullptr)
            pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }

    mpLocalMapper->enqueue_key_frame(pKFini);
    mpLocalMapper->enqueue_key_frame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    // PAE: this is done without any lock here!
    mpMap->add_to_key_frame_origins(pKFini);

    mState = OK;
}

void Tracking::CheckReplacedInLastFrame()
{
    for (int i = 0; i < mLastFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mLastFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        MapPoint* pRep = rpMP->GetReplaced();
        if (pRep != nullptr)
            rpMP = pRep;
    }
}

bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7f, true);

    std::vector<MapPoint*> vpMapPointMatches;
    if (matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches) < 15)
        return false;

    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    mCurrentFrame.SetPose(mLastFrame.mTcw);

    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
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
    KeyFrame* pRef = mLastFrame.mpReferenceKF;
    cv::Mat Tlr = mlRelativeFramePoses.back();
    mLastFrame.SetPose(Tlr * pRef->GetPose());

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

    if(vDepthIdx.empty())
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
    ORBmatcher matcher(0.9f, true);

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame();

    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

    // Project points seen in previous frame
    float th = mSensor != System::STEREO ? 15.f : 7.f;

    // If few matches, uses a wider window search
    std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
    int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
    if (nmatches < 20)
    {
        std::fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), nullptr);
        nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, 2.f * th, mSensor == System::MONOCULAR);
    }

    if (nmatches < 20)
        return false;

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame);

    // Discard outliers
    int nmatchesMap = 0;
    for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
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
        else if (rpMP->Observations() > 0)
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

    // Optimize Pose
    Optimizer::PoseOptimization(&mCurrentFrame);
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for (int i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        if (!mCurrentFrame.mvbOutlier[i])
        {
            rpMP->IncreaseFound();

            if (mbOnlyTracking || rpMP->Observations() > 0)
                ++mnMatchesInliers;
        }
        else if (mSensor == System::STEREO)
            rpMP = nullptr;
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if (mCurrentFrame.get_id() < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 50)
        return false;

    return mnMatchesInliers >= 30;
}

bool Tracking::NeedNewKeyFrame()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if (mbOnlyTracking || mpLocalMapper->is_paused() || mpLocalMapper->is_pause_requested())
        return false;

    std::size_t const nKFs = mpMap->KeyFramesInMap();

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    if (mCurrentFrame.get_id() < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    // Tracked MapPoints in the reference keyframe
    int nMinObs = nKFs <= 2 ? 2 : 3;

    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

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
    float thRefRatio = mSensor == System::MONOCULAR ? 0.9f : (nKFs < 2 ? 0.4f : 0.75f);

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    bool const c1a = mCurrentFrame.get_id() >= mnLastKeyFrameId + mMaxFrames;

    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    bool const c1b = mCurrentFrame.get_id() >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle;

    // Condition 1c: tracking is weak
    bool const c1c =  mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);

    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    bool const c2 = (mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) && mnMatchesInliers > 15;

    if ((c1a || c1b || c1c) && c2)
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA <- PAE: the comment is outdated
        return bLocalMappingIdle || (mSensor != System::MONOCULAR && mpLocalMapper->queued_key_frames() < 3);
    }

    return false;
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

    mpLocalMapper->enqueue_key_frame(pKF);

    mnLastKeyFrameId = mCurrentFrame.get_id();
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // do not search map points already matched
    for (MapPoint*& rpMP : mCurrentFrame.mvpMapPoints)
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

        int th = mSensor == System::RGBD ? 1 : 3;

        // If the camera has been relocalised recently, perform a coarser search
        if (mCurrentFrame.get_id() < mnLastRelocFrameId + 2)
            th = 5;

        matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, (float)th);
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
        std::vector<MapPoint*> const vpMPs = pKF->GetMapPointMatches();

        for (MapPoint* pMP : vpMPs)
        {
            if (pMP == nullptr || pMP->isBad() || pMP->mnTrackReferenceForFrame == mCurrentFrame.get_id())
                continue;

            mvpLocalMapPoints.push_back(pMP);
            pMP->mnTrackReferenceForFrame = mCurrentFrame.get_id();
        }
    }
}

void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    std::unordered_map<KeyFrame*, std::size_t> keyframeCounter;

    for (std::size_t i = 0; i < mCurrentFrame.get_frame_N(); ++i)
    {
        MapPoint*& rpMP = mCurrentFrame.mvpMapPoints[i];

        if (rpMP == nullptr)
            continue;

        if (rpMP->isBad())
        {
            rpMP = nullptr;
            continue;
        }

        std::unordered_map<KeyFrame*, size_t> const observations = rpMP->GetObservations();

        for (auto const& pair : observations)
        {
            std::size_t& count = keyframeCounter[pair.first];
            ++count;
        }
    }

    if (keyframeCounter.empty())
        return;

    std::size_t max = 0;
    KeyFrame* pKFmax = nullptr;

    mvpLocalKeyFrames.clear();
    mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    // Also check which keyframe shares most points
    for (auto const& kf_count_pair : keyframeCounter)
    {
        KeyFrame* pKF = kf_count_pair.first;

        if (pKF->isBad())
            continue;

        if (kf_count_pair.second > max)
        {
            max = kf_count_pair.second;
            pKFmax = pKF;
        }

        mvpLocalKeyFrames.push_back(kf_count_pair.first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.get_id();
    }

    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for (std::size_t index = 0, size = mvpLocalKeyFrames.size(); size <= 80 && index < size; ++index)
    {
        KeyFrame* pKF = mvpLocalKeyFrames[index];

        std::vector<KeyFrame*> const neighbour_key_frames = pKF->GetBestCovisibilityKeyFrames(10);

        for (KeyFrame* pNeighKF : neighbour_key_frames)
        {
            if (!pNeighKF->isBad() && pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.get_id())
            {
                pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.get_id();
                mvpLocalKeyFrames.push_back(pNeighKF);
                ++size;
                break;
            }
        }

        std::unordered_set<KeyFrame*> const spChilds = pKF->GetChilds();
        for (KeyFrame* pChildKF : spChilds)
        {
            if (!pChildKF->isBad() && pChildKF->mnTrackReferenceForFrame != mCurrentFrame.get_id())
            {
                pChildKF->mnTrackReferenceForFrame = mCurrentFrame.get_id();
                mvpLocalKeyFrames.push_back(pChildKF);
                ++size;
                break;
            }
        }

        KeyFrame* pParent = pKF->GetParent();

        if (pParent != nullptr && pParent->mnTrackReferenceForFrame != mCurrentFrame.get_id())
        {
            pParent->mnTrackReferenceForFrame = mCurrentFrame.get_id();
            mvpLocalKeyFrames.push_back(pParent);
            ++size;
            break;
        }
    }

    if (pKFmax != nullptr)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW();

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

        int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
        if (nmatches < 15)
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

                int nGood = Optimizer::PoseOptimization(&mCurrentFrame);
                if (nGood < 10)
                    continue;

                for (int io = 0; io < mCurrentFrame.get_frame_N(); ++io)
                    if (mCurrentFrame.mvbOutlier[io])
                        mCurrentFrame.mvpMapPoints[io] = nullptr;

                // If few inliers, search by projection in a coarse window and optimize again
                if (nGood < 50)
                {
                    int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);
                    if (nadditional + nGood>=50)
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

    if (mpViewer != nullptr)
    {
        mpViewer->RequestStop();
        while (!mpViewer->isStopped())
            std::this_thread::sleep_for(std::chrono::microseconds(3000));
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset();
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



} //namespace ORB_SLAM
