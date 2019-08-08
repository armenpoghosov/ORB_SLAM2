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

std::vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
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

        if ((it->second.m_score = mpVoc->score(pKF->get_BoW(), it->first->get_BoW())) >= minScore)
        {
            it->second.m_next = head_list;
            head_list = &(*it);
        }

        ++it;
    }

    if (head_list == nullptr)
        return vpLoopCandidates;

    double dblBestAccumulatedScore = minScore;

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

vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame* F)
{
    std::list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current frame
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        for (auto const& pair_work_id_value : F->mBowVec)
        {
            std::list<KeyFrame*> const& lKFs = mvInvertedFile[pair_work_id_value.first];

            for (KeyFrame* pKFi : lKFs)
            {
                if (pKFi->mnRelocQuery != F->get_id())
                {
                    pKFi->mnRelocWords = 0;
                    pKFi->mnRelocQuery = F->get_id();
                    lKFsSharingWords.push_back(pKFi);
                }

                ++pKFi->mnRelocWords;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords = 0;
    for (KeyFrame* pKFi : lKFsSharingWords)
    {
        if (pKFi->mnRelocWords > maxCommonWords)
            maxCommonWords = pKFi->mnRelocWords;
    }

    int minCommonWords = (int)(maxCommonWords * 0.8f);

    std::list<pair<float, KeyFrame*> > lScoreAndMatch;

    // Compute similarity score.
    for (KeyFrame* pKFi : lKFsSharingWords)
    {
        if (pKFi->mnRelocWords > minCommonWords)
        {
            float const si = (float)mpVoc->score(F->mBowVec, pKFi->mBowVec);
            pKFi->mRelocScore = si;
            lScoreAndMatch.emplace_back(si, pKFi);
        }
    }

    if (lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    float bestAccScore = 0;
    std::list<pair<float, KeyFrame*> > lAccScoreAndMatch;

    // Lets now accumulate score by covisibility
    for (auto const& pair_score_key_frame : lScoreAndMatch)
    {
        KeyFrame* pKFi = pair_score_key_frame.second;

        KeyFrame* pBestKF = pKFi;
        float bestScore = pair_score_key_frame.first;

        std::vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);
        float accScore = bestScore;

        for (KeyFrame* pKF2 : vpNeighs)
        {
            if (pKF2->mnRelocQuery != F->get_id())
                continue;

            accScore += pKF2->mRelocScore;

            if (pKF2->mRelocScore > bestScore)
            {
                pBestKF = pKF2;
                bestScore = pKF2->mRelocScore;
            }
        }

        lAccScoreAndMatch.emplace_back(accScore, pBestKF);
        
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75 * bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    std::unordered_set<KeyFrame*> spAlreadyAddedKF;

    std::vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());

    for (auto const& pair : lAccScoreAndMatch)
    {
        if (pair.first <= minScoreToRetain)
            continue;

        auto const& pair_set = spAlreadyAddedKF.insert(pair.second);
        if (pair_set.second)
            vpRelocCandidates.push_back(pair.second);
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM
