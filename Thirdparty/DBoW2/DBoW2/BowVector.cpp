/**
 * File: BowVector.cpp
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: bag of words vector
 * License: see the LICENSE.txt file
 *
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

#include "BowVector.h"

namespace DBoW2
{

// --------------------------------------------------------------------------

void BowVector::normalize(LNorm norm_type)
{
    double norm = 0.; 

    BowVector::iterator it;

    if (norm_type == DBoW2::L1)
    {
        for (auto const& pair : *this)
            norm += std::fabs(pair.second);
    }
    else
    {
        for (auto const& pair : *this)
            norm += pair.second * pair.second;

        norm = std::sqrt(norm);
    }

    if (norm > 0.)
    {
        for (auto& pair : *this)
            pair.second /= norm;
    }
}

// --------------------------------------------------------------------------

std::ostream& operator<< (std::ostream &out, const BowVector &v)
{
    for (auto it = v.cbegin(), itEnd = v.cend();;)
    {
        out << "<" << it->first << ", " << it->second << ">";

        if (++it == itEnd)
            return out;

        out << ", ";
    }
}

// --------------------------------------------------------------------------

void BowVector::saveM(std::string const& filename, size_t W) const
{
    std::fstream f(filename.c_str(), std::ios::out);

    WordId last = 0;

    for (auto it = cbegin(), itEnd = cend(); it != itEnd; ++it)
    {
        for (; last < it->first; ++last)
            f << "0 ";

        f << it->second << " ";

        last = it->first + 1;
    }
  
    for (; last < (WordId)W; ++last)
        f << "0 ";

    f.close();
}

// --------------------------------------------------------------------------

} // namespace DBoW2

