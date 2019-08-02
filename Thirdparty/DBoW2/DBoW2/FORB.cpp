/**
 * File: FORB.cpp
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 * Distance function has been modified 
 *
 */

 
#include <vector>
#include <string>
#include <sstream>
#include <cstdint>

#include "FORB.h"

using namespace std;

namespace DBoW2
{

// --------------------------------------------------------------------------

void FORB::meanValue(std::vector<FORB::pDescriptor> const& descriptors, FORB::TDescriptor& mean)
{
    if (descriptors.empty())
    {
        mean.release();
        return;
    }
  
    if (descriptors.size() == 1)
    {
        mean = descriptors[0]->clone();
        return;
    }

    std::size_t sum[FORB::L * 8] = {};

    for (cv::Mat const* d : descriptors)
    {
        uint8_t const* p = d->ptr<uint8_t>();

        assert(d->cols == FORB::L);

        for (int j = 0; j < FORB::L; ++j)
        {
            uint8_t const byte = p[j];
            sum[j * 8 + 0] += ((byte & (1 << 7)) != 0);
            sum[j * 8 + 1] += ((byte & (1 << 6)) != 0);
            sum[j * 8 + 2] += ((byte & (1 << 5)) != 0);
            sum[j * 8 + 3] += ((byte & (1 << 4)) != 0);
            sum[j * 8 + 4] += ((byte & (1 << 3)) != 0);
            sum[j * 8 + 5] += ((byte & (1 << 2)) != 0);
            sum[j * 8 + 6] += ((byte & (1 << 1)) != 0);
            sum[j * 8 + 7] += ((byte & (1 << 0)) != 0);
        }
    }

    mean = cv::Mat::zeros(1, FORB::L, CV_8U);
    uint8_t* p = mean.ptr<uint8_t>();

    std::size_t const descriptors_size = descriptors.size();
    std::size_t N2 = (descriptors_size >> 1) + (descriptors_size & 1);

    for (int i = 0; i < FORB::L * 8; ++i)
    {
        if (sum[i] >= N2)
            *p |= 1 << (7 - (i % 8));
      
        if (i % 8 == 7)
            ++p;
    }
}

// --------------------------------------------------------------------------

int FORB::distance(FORB::TDescriptor const& a, FORB::TDescriptor const& b)
{
    // Bit set count operation from
    // http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel

    uint32_t const *pa = a.ptr<uint32_t>();
    uint32_t const *pb = b.ptr<uint32_t>();

    int dist = 0;

    for (int i = 0; i < 8; ++i)
    {
        uint32_t v = pa[i] ^ pb[i];

        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (int)((((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24);
    }

    return dist;
}

// --------------------------------------------------------------------------
  
std::string FORB::toString(const FORB::TDescriptor &a)
{
    ostringstream ss;

    for (uint8_t const* p = a.ptr<uint8_t>(), * pend = p + a.cols; p < pend; ++p)
        ss << (int)*p << " ";
  
    return ss.str();
}

// --------------------------------------------------------------------------
  
void FORB::fromString(FORB::TDescriptor& a, istringstream& ss)
{
    a.create(1, FORB::L, CV_8U);

    for (uint8_t *p = a.ptr<uint8_t>(), *pend = p + FORB::L; p < pend; ++p)
    {
        int n;
        ss >> n;
    
        if (!ss.fail()) 
            *p = (uint8_t)n;
    }
}

// --------------------------------------------------------------------------

void FORB::toMat32F(std::vector<TDescriptor> const& descriptors, cv::Mat& mat)
{
    if (descriptors.empty())
    {
        mat.release();
        return;
    }

    std::size_t const N = descriptors.size();

    mat.create((int)N, FORB::L * 8, CV_32F);
    float* p = mat.ptr<float>();

    for (size_t i = 0; i < N; ++i)
    {
        int const C = descriptors[i].cols;

        uint8_t const* desc = descriptors[i].ptr<uint8_t>();

        for (int j = 0; j < C; ++j, p += 8)
        {
            p[0] = (desc[j] & (1 << 7) ? 1.f : 0.f);
            p[1] = (desc[j] & (1 << 6) ? 1.f : 0.f);
            p[2] = (desc[j] & (1 << 5) ? 1.f : 0.f);
            p[3] = (desc[j] & (1 << 4) ? 1.f : 0.f);
            p[4] = (desc[j] & (1 << 3) ? 1.f : 0.f);
            p[5] = (desc[j] & (1 << 2) ? 1.f : 0.f);
            p[6] = (desc[j] & (1 << 1) ? 1.f : 0.f);
            p[7] = (desc[j] & (1 << 0) ? 1.f : 0.f);
        }
    } 
}

// --------------------------------------------------------------------------

void FORB::toMat8U(std::vector<TDescriptor> const& descriptors,  cv::Mat& mat)
{
    mat.create((int)descriptors.size(), 32, CV_8U);

    uint8_t* p = mat.ptr<uint8_t>();

    for (TDescriptor const& desc : descriptors)
    {
        uint8_t const* d = desc.ptr<uint8_t>();
        std::copy(d, d + 32, p);
        p += 32;
    }
}

// --------------------------------------------------------------------------

} // namespace DBoW2


