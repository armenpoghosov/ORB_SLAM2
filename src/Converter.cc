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


#include "Converter.h"

namespace ORB_SLAM2
{

std::vector<cv::Mat> Converter::toDescriptorVector(cv::Mat const& Descriptors)
{
    std::vector<cv::Mat> vDesc;
    vDesc.reserve(Descriptors.rows);

    for (int row = 0; row < Descriptors.rows; ++row)
        vDesc.push_back(Descriptors.row(row));

    return vDesc;
}

g2o::SE3Quat Converter::toSE3Quat(cv::Mat const& cvT)
{
    Eigen::Matrix<double, 3, 3> R;

    R << cvT.at<float>(0, 0), cvT.at<float>(0, 1), cvT.at<float>(0, 2),
         cvT.at<float>(1, 0), cvT.at<float>(1, 1), cvT.at<float>(1, 2),
         cvT.at<float>(2, 0), cvT.at<float>(2, 1), cvT.at<float>(2, 2);

    Eigen::Matrix<double, 3, 1> t(cvT.at<float>(0, 3), cvT.at<float>(1, 3), cvT.at<float>(2, 3));

    return g2o::SE3Quat(R,t);
}

cv::Mat Converter::toCvMat(g2o::SE3Quat const& SE3)
{
    Eigen::Matrix<double, 4, 4> eigMat = SE3.to_homogeneous_matrix();
    return toCvMat(eigMat);
}

cv::Mat Converter::toCvMat(g2o::Sim3 const& Sim3)
{
    Eigen::Matrix3d eigR = Sim3.rotation().toRotationMatrix();
    Eigen::Vector3d eigt = Sim3.translation();
    double s = Sim3.scale();
    return toCvSE3(s * eigR, eigt);
}

cv::Mat Converter::toCvMat(Eigen::Matrix<double, 4, 4> const& m)
{
    cv::Mat cvMat(4, 4, CV_32F);

    for (int i = 0; i < 4; ++i)
        for(int j = 0; j < 4; ++j)
            cvMat.at<float>(i, j) = (float)m(i, j);

    return cvMat;
}

cv::Mat Converter::toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3, 3, CV_32F);

    for(int i = 0; i < 3; ++i)
        for(int j=0; j<3; ++j)
            cvMat.at<float>(i, j) = m(i, j);

    return cvMat;
}

cv::Mat Converter::toCvMat(Eigen::Matrix<double, 3, 1> const& m)
{
    cv::Mat cvMat(3, 1, CV_32F);

    for (int i = 0; i <3;i++)
            cvMat.at<float>(i) = (float)m(i);

    return cvMat;
}

cv::Mat Converter::toCvSE3(Eigen::Matrix<double, 3, 3> const& R, Eigen::Matrix<double, 3, 1> const& t)
{
    cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
            cvMat.at<float>(i, j) = (float)R(i, j);

        cvMat.at<float>(i, 3) = (float)t(i);
    }

    return cvMat;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Mat &cvVector)
{
    Eigen::Matrix<double,3,1> v;
    v << cvVector.at<float>(0), cvVector.at<float>(1), cvVector.at<float>(2);

    return v;
}

Eigen::Matrix<double,3,1> Converter::toVector3d(const cv::Point3f &cvPoint)
{
    Eigen::Matrix<double,3,1> v;
    v << cvPoint.x, cvPoint.y, cvPoint.z;

    return v;
}

Eigen::Matrix<double,3,3> Converter::toMatrix3d(const cv::Mat &cvMat3)
{
    Eigen::Matrix<double,3,3> M;

    M << cvMat3.at<float>(0,0), cvMat3.at<float>(0,1), cvMat3.at<float>(0,2),
         cvMat3.at<float>(1,0), cvMat3.at<float>(1,1), cvMat3.at<float>(1,2),
         cvMat3.at<float>(2,0), cvMat3.at<float>(2,1), cvMat3.at<float>(2,2);

    return M;
}

std::vector<float> Converter::toQuaternion(cv::Mat const& M)
{
    Eigen::Matrix<double, 3, 3> eigMat = toMatrix3d(M);
    Eigen::Quaterniond q(eigMat);

    std::vector<float> v(4);
    v[0] = (float)q.x();
    v[1] = (float)q.y();
    v[2] = (float)q.z();
    v[3] = (float)q.w();

    return v;
}

} //namespace ORB_SLAM
