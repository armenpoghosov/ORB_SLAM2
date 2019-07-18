/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>

#include "ORBextractor.h"

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

static int init_umax(int (&u_max)[ORBextractor::HALF_PATCH_SIZE + 1])
{
    // This is for orientation
    // pre-compute the end of a row in a circular patch

    int const vmin = cvCeil(ORBextractor::HALF_PATCH_SIZE * std::sqrt(2.f) / 2);
    int const vmax = cvFloor(ORBextractor::HALF_PATCH_SIZE * std::sqrt(2.f) / 2 + 1);

    double const hp2 = ORBextractor::HALF_PATCH_SIZE * ORBextractor::HALF_PATCH_SIZE;
    for (int v = 0; v <= vmax; ++v)
        u_max[v] = cvRound(std::sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (int v = ORBextractor::HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (u_max[v0] == u_max[v0 + 1])
            ++v0;

        u_max[v] = v0++;
    }

    return 0;
}

static float IC_Angle(Mat const& image, Point2f const& pt)
{
    static int u_max[ORBextractor::HALF_PATCH_SIZE + 1];
    static int const init = init_umax(u_max);

    int m_01 = 0;
    int m_10 = 0;

    uchar const* center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -ORBextractor::HALF_PATCH_SIZE; u <= ORBextractor::HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= ORBextractor::HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];

        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step];
            int val_minus = center[u - v * step];

            v_sum += val_plus - val_minus;
            m_10 += u * (val_plus + val_minus);
        }

        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

static void computeOrientation(Mat const& image, std::vector<KeyPoint>& keypoints)
{
    for (KeyPoint& keypoint : keypoints)
        keypoint.angle = IC_Angle(image, keypoint.pt);
}

static void computeOrbDescriptor(KeyPoint const& kpt, Mat const& img, uchar* desc)
{
    static Point const pattern_array[512] =
    {
        Point(8,-3), Point(9,5),// mean (0), correlation (0)
        Point(4,2), Point(7,-12),// mean (1.12461e-05), correlation (0.0437584)
        Point(-11,9), Point(-8,2),// mean (3.37382e-05), correlation (0.0617409)
        Point(7,-12), Point(12,-13),// mean (5.62303e-05), correlation (0.0636977)
        Point(2,-13), Point(2,12),// mean (0.000134953), correlation (0.085099)
        Point(1,-7), Point(1,6),// mean (0.000528565), correlation (0.0857175)
        Point(-2,-10), Point(-2,-4),// mean (0.0188821), correlation (0.0985774)
        Point(-13,-13), Point(-11,-8),// mean (0.0363135), correlation (0.0899616)
        Point(-13,-3), Point(-12,-9),// mean (0.121806), correlation (0.099849)
        Point(10,4), Point(11,9),// mean (0.122065), correlation (0.093285)
        Point(-13,-8), Point(-8,-9),// mean (0.162787), correlation (0.0942748)
        Point(-11,7), Point(-9,12),// mean (0.21561), correlation (0.0974438)
        Point(7,7), Point(12,6),// mean (0.160583), correlation (0.130064)
        Point(-4,-5), Point(-3,0),// mean (0.228171), correlation (0.132998)
        Point(-13,2), Point(-12,-3),// mean (0.00997526), correlation (0.145926)
        Point(-9,0), Point(-7,5),// mean (0.198234), correlation (0.143636)
        Point(12,-6), Point(12,-1),// mean (0.0676226), correlation (0.16689)
        Point(-3,6), Point(-2,12),// mean (0.166847), correlation (0.171682)
        Point(-6,-13), Point(-4,-8),// mean (0.101215), correlation (0.179716)
        Point(11,-13), Point(12,-8),// mean (0.200641), correlation (0.192279)
        Point(4,7), Point(5,1),// mean (0.205106), correlation (0.186848)
        Point(5,-3), Point(10,-3),// mean (0.234908), correlation (0.192319)
        Point(3,-7), Point(6,12),// mean (0.0709964), correlation (0.210872)
        Point(-8,-7), Point(-6,-2),// mean (0.0939834), correlation (0.212589)
        Point(-2,11), Point(-1,-10),// mean (0.127778), correlation (0.20866)
        Point(-13,12), Point(-8,10),// mean (0.14783), correlation (0.206356)
        Point(-7,3), Point(-5,-3),// mean (0.182141), correlation (0.198942)
        Point(-4,2), Point(-3,7),// mean (0.188237), correlation (0.21384)
        Point(-10,-12), Point(-6,11),// mean (0.14865), correlation (0.23571)
        Point(5,-12), Point(6,-7),// mean (0.222312), correlation (0.23324)
        Point(5,-6), Point(7,-1),// mean (0.229082), correlation (0.23389)
        Point(1,0), Point(4,-5),// mean (0.241577), correlation (0.215286)
        Point(9,11), Point(11,-13),// mean (0.00338507), correlation (0.251373)
        Point(4,7), Point(4,12),// mean (0.131005), correlation (0.257622)
        Point(2,-1), Point(4,4),// mean (0.152755), correlation (0.255205)
        Point(-4,-12), Point(-2,7),// mean (0.182771), correlation (0.244867)
        Point(-8,-5), Point(-7,-10),// mean (0.186898), correlation (0.23901)
        Point(4,11), Point(9,12),// mean (0.226226), correlation (0.258255)
        Point(0,-8), Point(1,-13),// mean (0.0897886), correlation (0.274827)
        Point(-13,-2), Point(-8,2),// mean (0.148774), correlation (0.28065)
        Point(-3,-2), Point(-2,3),// mean (0.153048), correlation (0.283063)
        Point(-6,9), Point(-4,-9),// mean (0.169523), correlation (0.278248)
        Point(8,12), Point(10,7),// mean (0.225337), correlation (0.282851)
        Point(0,9), Point(1,3),// mean (0.226687), correlation (0.278734)
        Point(7,-5), Point(11,-10),// mean (0.00693882), correlation (0.305161)
        Point(-13,-6), Point(-11,0),// mean (0.0227283), correlation (0.300181)
        Point(10,7), Point(12,1),// mean (0.125517), correlation (0.31089)
        Point(-6,-3), Point(-6,12),// mean (0.131748), correlation (0.312779)
        Point(10,-9), Point(12,-4),// mean (0.144827), correlation (0.292797)
        Point(-13,8), Point(-8,-12),// mean (0.149202), correlation (0.308918)
        Point(-13,0), Point(-8,-4),// mean (0.160909), correlation (0.310013)
        Point(3,3), Point(7,8),// mean (0.177755), correlation (0.309394)
        Point(5,7), Point(10,-7),// mean (0.212337), correlation (0.310315)
        Point(-1,7), Point(1,-12),// mean (0.214429), correlation (0.311933)
        Point(3,-10), Point(5,6),// mean (0.235807), correlation (0.313104)
        Point(2,-4), Point(3,-10),// mean (0.00494827), correlation (0.344948)
        Point(-13,0), Point(-13,5),// mean (0.0549145), correlation (0.344675)
        Point(-13,-7), Point(-12,12),// mean (0.103385), correlation (0.342715)
        Point(-13,3), Point(-11,8),// mean (0.134222), correlation (0.322922)
        Point(-7,12), Point(-4,7),// mean (0.153284), correlation (0.337061)
        Point(6,-10), Point(12,8),// mean (0.154881), correlation (0.329257)
        Point(-9,-1), Point(-7,-6),// mean (0.200967), correlation (0.33312)
        Point(-2,-5), Point(0,12),// mean (0.201518), correlation (0.340635)
        Point(-12,5), Point(-7,5),// mean (0.207805), correlation (0.335631)
        Point(3,-10), Point(8,-13),// mean (0.224438), correlation (0.34504)
        Point(-7,-7), Point(-4,5),// mean (0.239361), correlation (0.338053)
        Point(-3,-2), Point(-1,-7),// mean (0.240744), correlation (0.344322)
        Point(2,9), Point(5,-11),// mean (0.242949), correlation (0.34145)
        Point(-11,-13), Point(-5,-13),// mean (0.244028), correlation (0.336861)
        Point(-1,6), Point(0,-1),// mean (0.247571), correlation (0.343684)
        Point(5,-3), Point(5,2),// mean (0.000697256), correlation (0.357265)
        Point(-4,-13), Point(-4,12),// mean (0.00213675), correlation (0.373827)
        Point(-9,-6), Point(-9,6),// mean (0.0126856), correlation (0.373938)
        Point(-12,-10), Point(-8,-4),// mean (0.0152497), correlation (0.364237)
        Point(10,2), Point(12,-3),// mean (0.0299933), correlation (0.345292)
        Point(7,12), Point(12,12),// mean (0.0307242), correlation (0.366299)
        Point(-7,-13), Point(-6,5),// mean (0.0534975), correlation (0.368357)
        Point(-4,9), Point(-3,4),// mean (0.099865), correlation (0.372276)
        Point(7,-1), Point(12,2),// mean (0.117083), correlation (0.364529)
        Point(-7,6), Point(-5,1),// mean (0.126125), correlation (0.369606)
        Point(-13,11), Point(-12,5),// mean (0.130364), correlation (0.358502)
        Point(-3,7), Point(-2,-6),// mean (0.131691), correlation (0.375531)
        Point(7,-8), Point(12,-7),// mean (0.160166), correlation (0.379508)
        Point(-13,-7), Point(-11,-12),// mean (0.167848), correlation (0.353343)
        Point(1,-3), Point(12,12),// mean (0.183378), correlation (0.371916)
        Point(2,-6), Point(3,0),// mean (0.228711), correlation (0.371761)
        Point(-4,3), Point(-2,-13),// mean (0.247211), correlation (0.364063)
        Point(-1,-13), Point(1,9),// mean (0.249325), correlation (0.378139)
        Point(7,1), Point(8,-6),// mean (0.000652272), correlation (0.411682)
        Point(1,-1), Point(3,12),// mean (0.00248538), correlation (0.392988)
        Point(9,1), Point(12,6),// mean (0.0206815), correlation (0.386106)
        Point(-1,-9), Point(-1,3),// mean (0.0364485), correlation (0.410752)
        Point(-13,-13), Point(-10,5),// mean (0.0376068), correlation (0.398374)
        Point(7,7), Point(10,12),// mean (0.0424202), correlation (0.405663)
        Point(12,-5), Point(12,9),// mean (0.0942645), correlation (0.410422)
        Point(6,3), Point(7,11),// mean (0.1074), correlation (0.413224)
        Point(5,-13), Point(6,10),// mean (0.109256), correlation (0.408646)
        Point(2,-12), Point(2,3),// mean (0.131691), correlation (0.416076)
        Point(3,8), Point(4,-6),// mean (0.165081), correlation (0.417569)
        Point(2,6), Point(12,-13),// mean (0.171874), correlation (0.408471)
        Point(9,-12), Point(10,3),// mean (0.175146), correlation (0.41296)
        Point(-8,4), Point(-7,9),// mean (0.183682), correlation (0.402956)
        Point(-11,12), Point(-4,-6),// mean (0.184672), correlation (0.416125)
        Point(1,12), Point(2,-8),// mean (0.191487), correlation (0.386696)
        Point(6,-9), Point(7,-4),// mean (0.192668), correlation (0.394771)
        Point(2,3), Point(3,-2),// mean (0.200157), correlation (0.408303)
        Point(6,3), Point(11,0),// mean (0.204588), correlation (0.411762)
        Point(3,-3), Point(8,-8),// mean (0.205904), correlation (0.416294)
        Point(7,8), Point(9,3),// mean (0.213237), correlation (0.409306)
        Point(-11,-5), Point(-6,-4),// mean (0.243444), correlation (0.395069)
        Point(-10,11), Point(-5,10),// mean (0.247672), correlation (0.413392)
        Point(-5,-8), Point(-3,12),// mean (0.24774), correlation (0.411416)
        Point(-10,5), Point(-9,0),// mean (0.00213675), correlation (0.454003)
        Point(8,-1), Point(12,-6),// mean (0.0293635), correlation (0.455368)
        Point(4,-6), Point(6,-11),// mean (0.0404971), correlation (0.457393)
        Point(-10,12), Point(-8,7),// mean (0.0481107), correlation (0.448364)
        Point(4,-2), Point(6,7),// mean (0.050641), correlation (0.455019)
        Point(-2,0), Point(-2,12),// mean (0.0525978), correlation (0.44338)
        Point(-5,-8), Point(-5,2),// mean (0.0629667), correlation (0.457096)
        Point(7,-6), Point(10,12),// mean (0.0653846), correlation (0.445623)
        Point(-9,-13), Point(-8,-8),// mean (0.0858749), correlation (0.449789)
        Point(-5,-13), Point(-5,-2),// mean (0.122402), correlation (0.450201)
        Point(8,-8), Point(9,-13),// mean (0.125416), correlation (0.453224)
        Point(-9,-11), Point(-9,0),// mean (0.130128), correlation (0.458724)
        Point(1,-8), Point(1,-2),// mean (0.132467), correlation (0.440133)
        Point(7,-4), Point(9,1),// mean (0.132692), correlation (0.454)
        Point(-2,1), Point(-1,-4),// mean (0.135695), correlation (0.455739)
        Point(11,-6), Point(12,-11),// mean (0.142904), correlation (0.446114)
        Point(-12,-9), Point(-6,4),// mean (0.146165), correlation (0.451473)
        Point(3,7), Point(7,12),// mean (0.147627), correlation (0.456643)
        Point(5,5), Point(10,8),// mean (0.152901), correlation (0.455036)
        Point(0,-4), Point(2,8),// mean (0.167083), correlation (0.459315)
        Point(-9,12), Point(-5,-13),// mean (0.173234), correlation (0.454706)
        Point(0,7), Point(2,12),// mean (0.18312), correlation (0.433855)
        Point(-1,2), Point(1,7),// mean (0.185504), correlation (0.443838)
        Point(5,11), Point(7,-9),// mean (0.185706), correlation (0.451123)
        Point(3,5), Point(6,-8),// mean (0.188968), correlation (0.455808)
        Point(-13,-4), Point(-8,9),// mean (0.191667), correlation (0.459128)
        Point(-5,9), Point(-3,-3),// mean (0.193196), correlation (0.458364)
        Point(-4,-7), Point(-3,-12),// mean (0.196536), correlation (0.455782)
        Point(6,5), Point(8,0),// mean (0.1972), correlation (0.450481)
        Point(-7,6), Point(-6,12),// mean (0.199438), correlation (0.458156)
        Point(-13,6), Point(-5,-2),// mean (0.211224), correlation (0.449548)
        Point(1,-10), Point(3,10),// mean (0.211718), correlation (0.440606)
        Point(4,1), Point(8,-4),// mean (0.213034), correlation (0.443177)
        Point(-2,-2), Point(2,-13),// mean (0.234334), correlation (0.455304)
        Point(2,-12), Point(12,12),// mean (0.235684), correlation (0.443436)
        Point(-2,-13), Point(0,-6),// mean (0.237674), correlation (0.452525)
        Point(4,1), Point(9,3),// mean (0.23962), correlation (0.444824)
        Point(-6,-10), Point(-3,-5),// mean (0.248459), correlation (0.439621)
        Point(-3,-13), Point(-1,1),// mean (0.249505), correlation (0.456666)
        Point(7,5), Point(12,-11),// mean (0.00119208), correlation (0.495466)
        Point(4,-2), Point(5,-7),// mean (0.00372245), correlation (0.484214)
        Point(-13,9), Point(-9,-5),// mean (0.00741116), correlation (0.499854)
        Point(7,1), Point(8,6),// mean (0.0208952), correlation (0.499773)
        Point(7,-8), Point(7,6),// mean (0.0220085), correlation (0.501609)
        Point(-7,-4), Point(-7,1),// mean (0.0233806), correlation (0.496568)
        Point(-8,11), Point(-7,-8),// mean (0.0236505), correlation (0.489719)
        Point(-13,6), Point(-12,-8),// mean (0.0268781), correlation (0.503487)
        Point(2,4), Point(3,9),// mean (0.0323324), correlation (0.501938)
        Point(10,-5), Point(12,3),// mean (0.0399235), correlation (0.494029)
        Point(-6,-5), Point(-6,7),// mean (0.0420153), correlation (0.486579)
        Point(8,-3), Point(9,-8),// mean (0.0548021), correlation (0.484237)
        Point(2,-12), Point(2,8),// mean (0.0616622), correlation (0.496642)
        Point(-11,-2), Point(-10,3),// mean (0.0627755), correlation (0.498563)
        Point(-12,-13), Point(-7,-9),// mean (0.0829622), correlation (0.495491)
        Point(-11,0), Point(-10,-5),// mean (0.0843342), correlation (0.487146)
        Point(5,-3), Point(11,8),// mean (0.0929937), correlation (0.502315)
        Point(-2,-13), Point(-1,12),// mean (0.113327), correlation (0.48941)
        Point(-1,-8), Point(0,9),// mean (0.132119), correlation (0.467268)
        Point(-13,-11), Point(-12,-5),// mean (0.136269), correlation (0.498771)
        Point(-10,-2), Point(-10,11),// mean (0.142173), correlation (0.498714)
        Point(-3,9), Point(-2,-13),// mean (0.144141), correlation (0.491973)
        Point(2,-3), Point(3,2),// mean (0.14892), correlation (0.500782)
        Point(-9,-13), Point(-4,0),// mean (0.150371), correlation (0.498211)
        Point(-4,6), Point(-3,-10),// mean (0.152159), correlation (0.495547)
        Point(-4,12), Point(-2,-7),// mean (0.156152), correlation (0.496925)
        Point(-6,-11), Point(-4,9),// mean (0.15749), correlation (0.499222)
        Point(6,-3), Point(6,11),// mean (0.159211), correlation (0.503821)
        Point(-13,11), Point(-5,5),// mean (0.162427), correlation (0.501907)
        Point(11,11), Point(12,6),// mean (0.16652), correlation (0.497632)
        Point(7,-5), Point(12,-2),// mean (0.169141), correlation (0.484474)
        Point(-1,12), Point(0,7),// mean (0.169456), correlation (0.495339)
        Point(-4,-8), Point(-3,-2),// mean (0.171457), correlation (0.487251)
        Point(-7,1), Point(-6,7),// mean (0.175), correlation (0.500024)
        Point(-13,-12), Point(-8,-13),// mean (0.175866), correlation (0.497523)
        Point(-7,-2), Point(-6,-8),// mean (0.178273), correlation (0.501854)
        Point(-8,5), Point(-6,-9),// mean (0.181107), correlation (0.494888)
        Point(-5,-1), Point(-4,5),// mean (0.190227), correlation (0.482557)
        Point(-13,7), Point(-8,10),// mean (0.196739), correlation (0.496503)
        Point(1,5), Point(5,-13),// mean (0.19973), correlation (0.499759)
        Point(1,0), Point(10,-13),// mean (0.204465), correlation (0.49873)
        Point(9,12), Point(10,-1),// mean (0.209334), correlation (0.49063)
        Point(5,-8), Point(10,-9),// mean (0.211134), correlation (0.503011)
        Point(-1,11), Point(1,-13),// mean (0.212), correlation (0.499414)
        Point(-9,-3), Point(-6,2),// mean (0.212168), correlation (0.480739)
        Point(-1,-10), Point(1,12),// mean (0.212731), correlation (0.502523)
        Point(-13,1), Point(-8,-10),// mean (0.21327), correlation (0.489786)
        Point(8,-11), Point(10,-6),// mean (0.214159), correlation (0.488246)
        Point(2,-13), Point(3,-6),// mean (0.216993), correlation (0.50287)
        Point(7,-13), Point(12,-9),// mean (0.223639), correlation (0.470502)
        Point(-10,-10), Point(-5,-7),// mean (0.224089), correlation (0.500852)
        Point(-10,-8), Point(-8,-13),// mean (0.228666), correlation (0.502629)
        Point(4,-6), Point(8,5),// mean (0.22906), correlation (0.498305)
        Point(3,12), Point(8,-13),// mean (0.233378), correlation (0.503825)
        Point(-4,2), Point(-3,-3),// mean (0.234323), correlation (0.476692)
        Point(5,-13), Point(10,-12),// mean (0.236392), correlation (0.475462)
        Point(4,-13), Point(5,-1),// mean (0.236842), correlation (0.504132)
        Point(-9,9), Point(-4,3),// mean (0.236977), correlation (0.497739)
        Point(0,3), Point(3,-9),// mean (0.24314), correlation (0.499398)
        Point(-12,1), Point(-6,1),// mean (0.243297), correlation (0.489447)
        Point(3,2), Point(4,-8),// mean (0.00155196), correlation (0.553496)
        Point(-10,-10), Point(-10,9),// mean (0.00239541), correlation (0.54297)
        Point(8,-13), Point(12,12),// mean (0.0034413), correlation (0.544361)
        Point(-8,-12), Point(-6,-5),// mean (0.003565), correlation (0.551225)
        Point(2,2), Point(3,7),// mean (0.00835583), correlation (0.55285)
        Point(10,6), Point(11,-8),// mean (0.00885065), correlation (0.540913)
        Point(6,8), Point(8,-12),// mean (0.0101552), correlation (0.551085)
        Point(-7,10), Point(-6,5),// mean (0.0102227), correlation (0.533635)
        Point(-3,-9), Point(-3,9),// mean (0.0110211), correlation (0.543121)
        Point(-1,-13), Point(-1,5),// mean (0.0113473), correlation (0.550173)
        Point(-3,-7), Point(-3,4),// mean (0.0140913), correlation (0.554774)
        Point(-8,-2), Point(-8,3),// mean (0.017049), correlation (0.55461)
        Point(4,2), Point(12,12),// mean (0.01778), correlation (0.546921)
        Point(2,-5), Point(3,11),// mean (0.0224022), correlation (0.549667)
        Point(6,-9), Point(11,-13),// mean (0.029161), correlation (0.546295)
        Point(3,-1), Point(7,12),// mean (0.0303081), correlation (0.548599)
        Point(11,-1), Point(12,4),// mean (0.0355151), correlation (0.523943)
        Point(-3,0), Point(-3,6),// mean (0.0417904), correlation (0.543395)
        Point(4,-11), Point(4,12),// mean (0.0487292), correlation (0.542818)
        Point(2,-4), Point(2,1),// mean (0.0575124), correlation (0.554888)
        Point(-10,-6), Point(-8,1),// mean (0.0594242), correlation (0.544026)
        Point(-13,7), Point(-11,1),// mean (0.0597391), correlation (0.550524)
        Point(-13,12), Point(-11,-13),// mean (0.0608974), correlation (0.55383)
        Point(6,0), Point(11,-13),// mean (0.065126), correlation (0.552006)
        Point(0,-1), Point(1,4),// mean (0.074224), correlation (0.546372)
        Point(-13,3), Point(-9,-2),// mean (0.0808592), correlation (0.554875)
        Point(-9,8), Point(-6,-3),// mean (0.0883378), correlation (0.551178)
        Point(-13,-6), Point(-8,-2),// mean (0.0901035), correlation (0.548446)
        Point(5,-9), Point(8,10),// mean (0.0949843), correlation (0.554694)
        Point(2,7), Point(3,-9),// mean (0.0994152), correlation (0.550979)
        Point(-1,-6), Point(-1,-1),// mean (0.10045), correlation (0.552714)
        Point(9,5), Point(11,-2),// mean (0.100686), correlation (0.552594)
        Point(11,-3), Point(12,-8),// mean (0.101091), correlation (0.532394)
        Point(3,0), Point(3,5),// mean (0.101147), correlation (0.525576)
        Point(-1,4), Point(0,10),// mean (0.105263), correlation (0.531498)
        Point(3,-6), Point(4,5),// mean (0.110785), correlation (0.540491)
        Point(-13,0), Point(-10,5),// mean (0.112798), correlation (0.536582)
        Point(5,8), Point(12,11),// mean (0.114181), correlation (0.555793)
        Point(8,9), Point(9,-6),// mean (0.117431), correlation (0.553763)
        Point(7,-4), Point(8,-12),// mean (0.118522), correlation (0.553452)
        Point(-10,4), Point(-10,9),// mean (0.12094), correlation (0.554785)
        Point(7,3), Point(12,4),// mean (0.122582), correlation (0.555825)
        Point(9,-7), Point(10,-2),// mean (0.124978), correlation (0.549846)
        Point(7,0), Point(12,-2),// mean (0.127002), correlation (0.537452)
        Point(-1,-6), Point(0,-11)// mean (0.127148), correlation (0.547401)
    };

    float const angle = kpt.angle * (float)(CV_PI / 180.);

    float const a = std::cos(angle);
    float const b = std::sin(angle);

    uchar const* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    int const step = (int)img.step;

#define GET_VALUE(idx) \
    center[cvRound(pattern[idx].x * b + pattern[idx].y * a) * step + \
           cvRound(pattern[idx].x * a - pattern[idx].y * b)]

    Point const* pattern = pattern_array;

    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0;
        int t1;
        int val;

        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;

        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (int)(t0 < t1) << 1;

        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (int)(t0 < t1) << 2;

        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (int)(t0 < t1) << 3;

        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (int)(t0 < t1) << 4;

        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (int)(t0 < t1) << 5;

        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (int)(t0 < t1) << 6;

        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (int)(t0 < t1) << 7;

        desc[i] = (uchar)val;
    }

#undef GET_VALUE
}

static void computeDescriptors(Mat const& image, vector<KeyPoint>& keypoints, Mat& descriptors)
{
    int count_keypoints = (int)keypoints.size();

    descriptors = Mat::zeros(count_keypoints, 32, CV_8UC1);

    for (int i = 0; i < count_keypoints; ++i)
        computeOrbDescriptor(keypoints[i], image, descriptors.ptr(i));
}


ORBextractor::ORBextractor(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    :
    nfeatures(_nfeatures),
    scaleFactor(_scaleFactor),
    nlevels(_nlevels),
    iniThFAST(_iniThFAST),
    minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);

    mvScaleFactor[0] = 1.0f;
    mvLevelSigma2[0] = 1.0f;

    for (int i = 1; i < nlevels; ++i)
    {
        mvScaleFactor[i] = (float)(mvScaleFactor[i - 1] * scaleFactor); // PAE: what the hack
        mvLevelSigma2[i] = mvScaleFactor[i] * mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);

    for (int i = 0; i < nlevels; ++i)
    {
        mvInvScaleFactor[i] = 1.f / mvScaleFactor[i];
        mvInvLevelSigma2[i] = 1.f / mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    mnFeaturesPerLevel.resize(nlevels);

    float factor = (float)(1.0f / scaleFactor); // PAE: what the hack!
    float nDesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels - 1; ++level)
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }

    mnFeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
}

void ExtractorNode::DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4)
{
    int const halfX = ceil((float)(UR.x - UL.x) / 2);
    int const halfY = ceil((float)(BR.y - UL.y) / 2);

    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x + halfX, UL.y);
    n1.BL = cv::Point2i(UL.x, UL.y + halfY);
    n1.BR = cv::Point2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (cv::KeyPoint const& kp : vKeys)
    {
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else
        {
            if (kp.pt.y < n1.BR.y)
                n2.vKeys.push_back(kp);
            else
                n4.vKeys.push_back(kp);
        }
    }
}

vector<cv::KeyPoint> ORBextractor::DistributeOctTree(
    vector<cv::KeyPoint> const& vToDistributeKeys,
    int minX, int maxX, int minY, int maxY, int N)
{
    // compute how many initial nodes   
    int const nIni = round((float)(maxX - minX) / (maxY - minY));

    float const hX = (float)(maxX - minX) / nIni;

    list<ExtractorNode> lNodes;
    {
        std::vector<ExtractorNode*> vpIniNodes;
        vpIniNodes.resize(nIni);

        for (int i = 0; i < nIni; ++i)
        {
            ExtractorNode ni;

            ni.UL = cv::Point2i(hX * i, 0);
            ni.UR = cv::Point2i(hX * i + 1, 0);

            ni.BL = cv::Point2i(ni.UL.x, maxY - minY);
            ni.BR = cv::Point2i(ni.UR.x, maxY - minY);

            ni.vKeys.reserve(vToDistributeKeys.size());

            lNodes.push_back(ni);
            vpIniNodes[i] = &lNodes.back();
        }

        // Associate points to childs
        for (size_t i = 0; i < vToDistributeKeys.size(); ++i)
        {
            cv::KeyPoint const& kp = vToDistributeKeys[i];
            vpIniNodes[kp.pt.x / hX]->vKeys.push_back(kp);
        }
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    while (lit != lNodes.end())
    {
        if (lit->vKeys.empty())
            lit = lNodes.erase(lit);
        else
            ++lit;
    }

    std::vector<std::pair<std::size_t, ExtractorNode*> > vSizeAndPointerToNode;
    vSizeAndPointerToNode.reserve(lNodes.size() * 4);

    for (;;)
    {
        std::size_t prevSize = lNodes.size();

        std::size_t nToExpand = 0;

        vSizeAndPointerToNode.clear();

        for (lit = lNodes.begin(); lit != lNodes.end(); )
        {
            // PAE: here we had dummy bNoMore flag ... the article says though that the number
            // of corners retained should be 5 not 1
            if (lit->vKeys.size() == 1)
            {
                ++lit; // If node only contains one point do not subdivide and continue
                continue;
            }

            // If more than one point, subdivide
            ExtractorNode childs[4];
            lit->DivideNode(childs[0], childs[1], childs[2], childs[3]);

            // Add childs if they contain points
            for (int index = 0; index < 4; ++index)
            {
                ExtractorNode& child_node = childs[index];

                std::size_t const keys_count = child_node.vKeys.size();

                if (keys_count == 0)
                    continue;

                lNodes.push_front(child_node);

                if (keys_count > 1)
                {
                    ++nToExpand;

                    ExtractorNode& child_node_pushed = lNodes.front();
                    vSizeAndPointerToNode.emplace_back(keys_count, &child_node_pushed);
                    child_node_pushed.lit = lNodes.begin();
                }
            }

            lit = lNodes.erase(lit);
        }

        // Finish if there are more nodes than required features or all nodes contain just one point
        if (lNodes.size() >= (std::size_t)N || lNodes.size() == prevSize)
            break;

        if ((lNodes.size() + nToExpand * 3) > (std::size_t)N)
        {
            do
            {
                prevSize = lNodes.size();

                std::vector<std::pair<std::size_t, ExtractorNode*> > vPrevSizeAndPointerToNode;
                vPrevSizeAndPointerToNode.swap(vSizeAndPointerToNode);

                std::sort(vPrevSizeAndPointerToNode.begin(), vPrevSizeAndPointerToNode.end());

                for (int j = vPrevSizeAndPointerToNode.size() - 1; j >= 0; --j)
                {
                    ExtractorNode* parent_node = vPrevSizeAndPointerToNode[j].second;

                    ExtractorNode childs[4];
                    parent_node->DivideNode(childs[0], childs[1], childs[2], childs[3]);

                    // Add childs if they contain points
                    for (int index = 0; index < 4; ++index)
                    {
                        ExtractorNode& child_node = childs[index];

                        std::size_t const keys_count = child_node.vKeys.size();

                        if (keys_count == 0)
                            continue;

                        lNodes.push_front(child_node);

                        if (keys_count > 1)
                        {
                            ExtractorNode& child_node_pushed = lNodes.front();
                            vSizeAndPointerToNode.emplace_back(keys_count, &child_node_pushed);
                            child_node_pushed.lit = lNodes.begin();
                        }
                    }

                    lNodes.erase(parent_node->lit);

                    if (lNodes.size() >= (std::size_t)N)
                        break;
                }
            }
            while (lNodes.size() < (std::size_t)N && lNodes.size() != prevSize);

            break;
        }
    }

    // Retain the best point in each node
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(N);

    for (ExtractorNode& node : lNodes)
    {
        vector<cv::KeyPoint>& vNodeKeys = node.vKeys;

        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;

        for (size_t k = 1; k < vNodeKeys.size(); ++k)
        {
            if (vNodeKeys[k].response > maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    allKeypoints.resize(nlevels);

    float const W = 30;

    for (int level = 0; level < nlevels; ++level)
    {
        int const minBorderX = EDGE_THRESHOLD - 3;
        int const minBorderY = minBorderX;

        int const maxBorderX = mvImagePyramid[level].cols - EDGE_THRESHOLD + 3;
        int const maxBorderY = mvImagePyramid[level].rows - EDGE_THRESHOLD + 3;

        vector<cv::KeyPoint> vToDistributeKeys;
        vToDistributeKeys.reserve(nfeatures * 10);

        float const width = maxBorderX - minBorderX;
        float const height = maxBorderY - minBorderY;

        int const nCols = width / W;
        int const nRows = height / W;

        int const wCell = ceil(width / nCols);
        int const hCell = ceil(height / nRows);

        for (int i = 0; i < nRows; ++i)
        {
            float const iniY = minBorderY + i * hCell;

            float maxY = iniY + hCell + 6;

            if (iniY >= maxBorderY - 3)
                continue;

            if (maxY > maxBorderY)
                maxY = maxBorderY;

            for (int j = 0; j < nCols; ++j)
            {
                const float iniX = minBorderX + j * wCell;
                if (iniX >= maxBorderX - 6)
                    continue;

                float maxX = iniX + wCell + 6;
                if (maxX > maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                cv::Mat mat_fast = mvImagePyramid[level].rowRange(iniY, maxY).colRange(iniX, maxX);
                FAST(mat_fast, vKeysCell, iniThFAST, true);

                if (vKeysCell.empty())
                    FAST(mat_fast, vKeysCell, minThFAST, true);

                for (cv::KeyPoint& kp : vKeysCell)
                {
                    kp.pt.x += j * wCell;
                    kp.pt.y += i * hCell;

                    vToDistributeKeys.push_back(kp);
                }
            }
        }

        vector<KeyPoint>& keypoints = allKeypoints[level];

        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
            minBorderY, maxBorderY, mnFeaturesPerLevel[level]);

        float const scaledPatchSize = PATCH_SIZE * mvScaleFactor[level];

        // Add border to coordinates and scale information
        for (KeyPoint& cp : keypoints)
        {
            cp.pt.x += minBorderX;
            cp.pt.y += minBorderY;
            cp.octave = level;
            cp.size = scaledPatchSize;
        }
    }

    // compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level]);
}

void ORBextractor::operator () (InputArray _image, InputArray _mask,
    std::vector<KeyPoint>& _keypoints, OutputArray _descriptors)
{ 
    if (_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1);

    // Pre-compute the scale pyramid
    ComputePyramid(image);

    std::vector<std::vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);

    std::size_t nkeypoints = 0;
    for (std::vector<KeyPoint> const& level_vector : allKeypoints)
        nkeypoints += level_vector.size();

    Mat descriptors;

    if (nkeypoints == 0)
        _descriptors.release();
    else
    {
        _descriptors.create((int)nkeypoints, 32, CV_8U);
        descriptors = _descriptors.getMat();
    }

    _keypoints.clear();
    _keypoints.reserve(nkeypoints);

    int offset = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        vector<KeyPoint>& keypoints = allKeypoints[level];

        int const nkeypointsLevel = (int)keypoints.size();

        if (nkeypointsLevel == 0)
            continue;

        // preprocess the resized image
        Mat workingMat = mvImagePyramid[level].clone();
        GaussianBlur(workingMat, workingMat, Size(7, 7), 2, 2, BORDER_REFLECT_101);

        // Compute the descriptors
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        computeDescriptors(workingMat, keypoints, desc);

        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        if (level != 0)
        {
            float const scale = mvScaleFactor[level];

            for (KeyPoint& keypoint : keypoints)
                keypoint.pt *= scale;
        }

        // And add the keypoints to the output
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        float const scale = mvInvScaleFactor[level];

        Size sz(cvRound((float)image.cols * scale), cvRound((float)image.rows * scale));
        Size wholeSize(sz.width + EDGE_THRESHOLD * 2, sz.height + EDGE_THRESHOLD * 2);

        Mat temp(wholeSize, image.type());

        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if (level != 0)
        {
            resize(mvImagePyramid[level - 1], mvImagePyramid[level], sz, 0, 0, INTER_LINEAR);

            copyMakeBorder(mvImagePyramid[level], temp,
                EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                BORDER_REFLECT_101 + BORDER_ISOLATED);
        }
        else
        {
            copyMakeBorder(image, temp,
                EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                BORDER_REFLECT_101);
        }
    }

}

} //namespace ORB_SLAM
