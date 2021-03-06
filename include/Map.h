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

#ifndef MAP_H
#define MAP_H

#include <atomic>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace ORB_SLAM2
{

class MapPoint;
class KeyFrame;

class Map
{
public:

    Map()
        :
        mnMaxKFid(0),
        mnBigChangeIdx(0)
    {}

    bool AddKeyFrame(KeyFrame *pKF);

    bool AddMapPoint(MapPoint *pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspMapPoints.insert(pMP).second;
    }

    bool EraseMapPoint(MapPoint *pMP)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        // TODO: This only erases the pointer.
        return mspMapPoints.erase(pMP) != 0;
    }

    bool EraseKeyFrame(KeyFrame *pKF)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        // TODO: This only erases the pointer.
        return mspKeyFrames.erase(pKF) != 0;
    }

    // TODO: PAE: what does this thing do? ... it does not validate anything right?
    void SetReferenceMapPoints(std::unordered_set<MapPoint*> const& vpMPs)
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void InformNewBigChange()
        { ++mnBigChangeIdx; }

    int GetLastBigChangeIdx() const
        { return mnBigChangeIdx; }

    std::vector<KeyFrame*> GetAllKeyFrames() const;
    std::vector<MapPoint*> GetAllMapPoints() const;

    std::unordered_set<MapPoint*> GetAllMapPointsSet() const
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspMapPoints;
    }

    std::unordered_set<MapPoint*> GetReferenceMapPoints() const
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    std::size_t MapPointsInMap() const
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    std::size_t KeyFramesInMap() const
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    uint64_t GetMaxKFid() const
    {
        std::unique_lock<std::mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void clear();

    std::vector<KeyFrame*> const& get_key_frame_origins() const
        { return mvpKeyFrameOrigins; }

    void add_to_key_frame_origins(KeyFrame* pKF)
        { mvpKeyFrameOrigins.push_back(pKF); }

    std::mutex                      mMutexMapUpdate;

protected:

    std::unordered_set<MapPoint*>   mspMapPoints;
    std::unordered_set<KeyFrame*>   mspKeyFrames;
    std::vector<KeyFrame*>          mvpKeyFrameOrigins;
    std::unordered_set<MapPoint*>   mvpReferenceMapPoints;
    uint64_t                        mnMaxKFid;
    // Index related to a big change in the map (loop closure, global BA)
    std::atomic<int>                mnBigChangeIdx;
    std::mutex mutable              mMutexMap;
};

} //namespace ORB_SLAM

#endif // MAP_H
