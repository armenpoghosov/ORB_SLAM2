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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include <mutex>
#include <unordered_set>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase(ORBVocabulary const& voc)
    :
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

void KeyFrameDatabase::add(KeyFrame *pKF)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    for (auto const& pair : pKF->get_BoW())
        mvInvertedFile[pair.first].push_back(pKF);
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    std::unique_lock<std::mutex> lock(m_mutex);

    // Erase elements in the Inverse File for the entry
    for (auto const& pair : pKF->get_BoW())
    {
        // List of keyframes that share the word
        std::list<KeyFrame*>& lKFs = mvInvertedFile[pair.first];
        auto it = std::find(lKFs.begin(), lKFs.end(), pKF);
        assert(pKF != lKFs.end());
        lKFs.erase(it);
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, double dblMinScore) const
{
    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    struct TEntry
    {
        std::size_t                         m_count = 0;
        double                              m_score = 0.;
        std::pair<KeyFrame* const, TEntry>* m_next = nullptr;
    };

    std::vector<KeyFrame*> vpLoopCandidates;

    std::unordered_set<KeyFrame*> const& spConnectedKeyFrames = pKF->GetConnectedKeyFrames();

    std::size_t maxCommonWords = 0;
    std::unordered_map<KeyFrame*, TEntry> map;
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        for (auto const& pair : pKF->get_BoW())
        {
            for (KeyFrame* pWIdKF : mvInvertedFile[pair.first])
            {
                if (spConnectedKeyFrames.find(pWIdKF) != spConnectedKeyFrames.end())
                    continue;

                std::size_t& count = map[pWIdKF].m_count;

                if (++count > maxCommonWords)
                    maxCommonWords = count;
            }
        }
    }

    if (map.empty())
        return vpLoopCandidates;

    std::size_t const minCommonWordsRetain = (std::size_t)(maxCommonWords * 0.8);

    // Remove all frames that have < minCommonWordsRetain. Compute similarity score.
    // Link the matches whose score is higher than minScore.
    std::pair<KeyFrame* const, TEntry>* head_list = nullptr;
    for (auto it = map.begin(), itEnd = map.end(); it != itEnd;)
    {
        if (it->second.m_count <= minCommonWordsRetain)
        {
            it = map.erase(it);
            continue;
        }

        it->second.m_score = mpVoc->score(pKF->get_BoW(), it->first->get_BoW());

        if (it->second.m_score >= dblMinScore)
        {
            it->second.m_next = head_list;
            head_list = &(*it);
        }

        ++it;
    }

    if (head_list == nullptr)
        return vpLoopCandidates;

    double dblBestAccumulatedScore = dblMinScore;

    // Lets now accumulate score by covisibility
    std::unordered_map<KeyFrame*, double> map_best_scores;

    while (head_list != nullptr)
    {
        KeyFrame* pKFBestScore = head_list->first;
        double dblBestScore = head_list->second.m_score;

        double dblAccumulatedScore = dblBestScore;
        std::vector<KeyFrame*> const& neighbours = pKFBestScore->GetBestCovisibilityKeyFrames(10);

        for (KeyFrame* pKFN : neighbours)
        {
            auto it = map.find(pKFN);
            if (it == map.end())
                continue;

            dblAccumulatedScore += it->second.m_score;

            if (it->second.m_score > dblBestScore)
            {
                pKFBestScore = pKFN;
                dblBestScore = it->second.m_score;
            }
        }

        // TODO: PAE: I honestly don't fully undersand why it makes sence to examine
        // convisibility frames here this way ... I mean the entire algorithm ...
        auto const& pair = map_best_scores.emplace(pKFBestScore, dblBestScore);
        if (!pair.second && pair.first->second < dblBestScore)
            pair.first->second = dblBestScore;

        if (dblAccumulatedScore > dblBestAccumulatedScore)
            dblBestAccumulatedScore = dblAccumulatedScore;

        head_list = head_list->second.m_next;
    }

    // Return all those keyframes with a score higher than 0.75 * dblBestAccumulatedScore
    double const dblMinScoreToRetain = 0.75 * dblBestAccumulatedScore;

    vpLoopCandidates.reserve(map_best_scores.size());

    for (auto const& pair : map_best_scores)
    {
        if (pair.second >= dblMinScoreToRetain)
            vpLoopCandidates.push_back(pair.first);
    }

    return vpLoopCandidates;
}

std::vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame* F) const
{
    struct TEntry
    {
        std::size_t m_count = 0;
        double      m_score = 0.;
    };

    std::vector<KeyFrame*> vpRelocCandidates;

    std::size_t maxCommonWords = 0;
    std::unordered_map<KeyFrame*, TEntry> map;
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        for (auto const& pair : F->mBowVec)
        {
            for (KeyFrame* pWIdKF : mvInvertedFile[pair.first])
            {
                std::size_t& count = map[pWIdKF].m_count;
                if (++count > maxCommonWords)
                    maxCommonWords = count;
            }
        }
    }

    if (map.empty())
        return vpRelocCandidates;

    std::size_t const minCommonWordsRetain = (std::size_t)(maxCommonWords * 0.8);

    for (auto it = map.begin(), itEnd = map.end(); it != itEnd;)
    {
        if (it->second.m_count <= minCommonWordsRetain)
        {
            it = map.erase(it);
            continue;
        }

        it->second.m_score = mpVoc->score(F->mBowVec, it->first->get_BoW());

        ++it;
    }

    if (map.empty())
        return vpRelocCandidates;

    double dblBestAccumulatedScore = 0.;
    std::unordered_map<KeyFrame*, double> map_best_scores;

    for (auto const& pair : map)
    {
        KeyFrame* pKFBestScore = pair.first;
        double dblBestScore = pair.second.m_score;

        double dblAccumulatedScore = dblBestScore;
        std::vector<KeyFrame*> const& neighbours = pKFBestScore->GetBestCovisibilityKeyFrames(10);

        for (KeyFrame* pKFN : neighbours)
        {
            auto it = map.find(pKFN);
            if (it == map.end())
                continue;

            dblAccumulatedScore += it->second.m_score;

            if (it->second.m_score > dblBestScore)
            {
                pKFBestScore = pKFN;
                dblBestScore = it->second.m_score;
            }
        }

        // TODO: PAE: I honestly don't fully undersand why it makes sence to examine
        // convisibility frames here this way ... I mean the entire algorithm ...
        auto const& pair = map_best_scores.emplace(pKFBestScore, dblBestScore);
        if (!pair.second && pair.first->second < dblBestScore)
            pair.first->second = dblBestScore;

        if (dblAccumulatedScore > dblBestAccumulatedScore)
            dblBestAccumulatedScore = dblAccumulatedScore;
    }

    double const dblMinScoreToRetain = 0.75 * dblBestAccumulatedScore;

    vpRelocCandidates.reserve(map_best_scores.size());

    for (auto const& pair : map_best_scores)
    {
        if (pair.second >= dblMinScoreToRetain)
            vpRelocCandidates.push_back(pair.first);
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
