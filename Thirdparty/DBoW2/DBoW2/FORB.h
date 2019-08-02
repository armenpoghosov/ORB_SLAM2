/**
 * File: FORB.h
 * Date: June 2012
 * Author: Dorian Galvez-Lopez
 * Description: functions for ORB descriptors
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_F_ORB__
#define __D_T_F_ORB__

#include <opencv2/core/core.hpp>

#include <vector>
#include <string>

#include "FClass.h"

namespace DBoW2
{

/// Functions to manipulate ORB descriptors
class FORB: protected FClass
{
public:

    /// Descriptor type
    typedef cv::Mat TDescriptor; // CV_8U
    /// Pointer to a single descriptor
    typedef const TDescriptor *pDescriptor;

    /// Descriptor length (in bytes)
    enum : int { L = 32 };

    /**
    * Calculates the mean value of a set of descriptors
    * @param descriptors
    * @param mean mean descriptor
    */
    static void meanValue(std::vector<pDescriptor> const& descriptors, TDescriptor& mean);

    /**
    * Calculates the distance between two descriptors
    * @param a
    * @param b
    * @return distance
    */
    static int distance(TDescriptor const& a, TDescriptor const& b);

    /**
    * Returns a string version of the descriptor
    * @param a descriptor
    * @return string version
    */
    static std::string toString(TDescriptor const& a);

    /**
    * Returns a descriptor from a string
    * @param a descriptor
    * @param s string version
    */
    static void fromString(TDescriptor& a, std::istringstream& s);

    /**
    * Returns a mat with the descriptors in float format
    * @param descriptors
    * @param mat (out) NxL 32F matrix
    */
    static void toMat32F(std::vector<TDescriptor> const& descriptors, cv::Mat& mat);
    static void toMat8U(std::vector<TDescriptor> const& descriptors, cv::Mat& mat);
};

} // namespace DBoW2

#endif

