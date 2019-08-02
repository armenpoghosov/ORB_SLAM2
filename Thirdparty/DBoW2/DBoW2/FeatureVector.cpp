/**
 * File: FeatureVector.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: feature vector
 * License: see the LICENSE.txt file
 *
 */

#include "FeatureVector.h"

#include <map>
#include <vector>
#include <iostream>

namespace DBoW2
{

// -------------------------------------------------------------------------------------------------

std::ostream& operator << (std::ostream &out, FeatureVector const& v)
{
    if (!v.empty())
    {
        FeatureVector::const_iterator vit = v.begin();

        std::vector<uint32_t> const* f = &vit->second;

        out << "<" << vit->first << ": [";

        if (!f->empty())
            out << (*f)[0];

        for (std::size_t i = 1; i < f->size(); ++i)
            out << ", " << (*f)[i];

        out << "]>";
    
        for (++vit; vit != v.end(); ++vit)
        {
            f = &vit->second;

            out << ", <" << vit->first << ": [";

            if (!f->empty())
                out << (*f)[0];

            for (std::size_t i = 1; i < f->size(); ++i)
                out << ", " << (*f)[i];

            out << "]>";
        }
    }
  
    return out;
}

// -------------------------------------------------------------------------------------------------

} // namespace DBoW2
