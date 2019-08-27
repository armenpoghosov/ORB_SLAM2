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

#include "Map.h"

#include "MapPoint.h"
#include "KeyFrame.h"

#include <algorithm>

namespace ORB_SLAM2
{

bool Map::AddKeyFrame(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(mMutexMap);

    auto const& pair = mspKeyFrames.insert(pKF);

    if (pKF->get_id() > mnMaxKFid)
        mnMaxKFid = pKF->get_id();

    return pair.second;
}

std::vector<KeyFrame*> Map::GetAllKeyFrames() const
{
    // --------------------------------------------------------------------------
    // TODO: PAE: we need determinizm for now
    std::vector<KeyFrame*> kfs;
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        kfs.reserve(mspKeyFrames.size());
        kfs.insert(kfs.end(), mspKeyFrames.begin(), mspKeyFrames.end());
    }

    std::sort(kfs.begin(), kfs.end(),
        [] (KeyFrame const* p1, KeyFrame const* p2)->bool
        {
            return p1->get_id() < p2->get_id();
        });
    // --------------------------------------------------------------------------

    return kfs;
}

std::vector<MapPoint*> Map::GetAllMapPoints() const
{
    // --------------------------------------------------------------------------
    // TODO: PAE: we need determinizm for now
    std::vector<MapPoint*> mps;
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mps.reserve(mspMapPoints.size());
        mps.insert(mps.end(), mspMapPoints.begin(), mspMapPoints.end());
    }

    std::sort(mps.begin(), mps.end(),
        [] (MapPoint const* p1, MapPoint const* p2)->bool
        {
            return p1->get_id() < p2->get_id();
        });
    // --------------------------------------------------------------------------

    return mps;
}

void Map::clear()
{
    for (MapPoint* pMP : mspMapPoints)
        delete pMP;

    for (KeyFrame* pKF : mspKeyFrames)
        delete pKF;

    mspMapPoints.clear();
    mspKeyFrames.clear();

    mnMaxKFid = 0;

    mvpReferenceMapPoints.clear();
    mvpKeyFrameOrigins.clear();
}

} //namespace ORB_SLAM
