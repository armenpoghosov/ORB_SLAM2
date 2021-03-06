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

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <mfapi.h>

#include <opencv2/opencv.hpp>

#include <conio.h>

#include"System.h"

void LoadImages(std::string const& strSequence,
    std::vector<std::string>& vstrImageFilenames, std::vector<double>& vTimestamps);

int main(int argc, char **argv)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    if (argc != 4)
    {
        cerr << endl << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    cv::VideoCapture capture;
    if (!capture.open(argv[3]))
    {
        cerr << "could not open video file" << endl;
        return -1;
    }

    /*
    for (int index = 0; ; ++index)
    {
        cv::Mat img;
        capture >> img;
        if (img.empty())
            break;
        char buffer[256];
        sprintf(buffer, "C:\\Users\\ArmenPoghosovSystemA\\Downloads\\video_urban5_cut\\%d.jpg", index);
        cv::imwrite(buffer, img);
    }*/

    // Retrieve paths to images
    //vector<string> vstrImageFilenames;
    //vector<double> vTimestamps;
    //LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    //int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // Vector for tracking time statistics
    //vector<float> vTimesTrack;
    //vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    //cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat im;

    double tframe = 0;


    //for(int ni = 0; ni < nImages; ++ni)
    for (;;)
    {
        // Read image from file
        //im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);
        capture >> im;

        // double tframe = vTimestamps[ni];

        if (im.empty())
        {
            // cerr << endl << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            // return 1;
            break;
        }

        //std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        SLAM.TrackMonocular(im, tframe += 25);

        if (::_kbhit() != 0 && ::_getch() == ' ')
            ::_getch();


        //std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        //double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        //vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        //double T=0;
        //if(ni<nImages-1)
        //    T = vTimestamps[ni+1]-tframe;
        // else if(ni>0)
        //    T = tframe-vTimestamps[ni-1];

        // if(ttrack<T)
        //    std::this_thread::sleep_for(std::chrono::microseconds((uint64_t)(T - ttrack) * 1000000));
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    //sort(vTimesTrack.begin(), vTimesTrack.end());
    //float totaltime = 0;
    //for(int ni=0; ni<nImages; ni++)
    {
      //  totaltime+=vTimesTrack[ni];
    }
    //cout << "-------" << endl << endl;
    //cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    //cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");    

    std::chrono::seconds seconds = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - t0);

    printf("time = %u\n", (uint32_t)seconds.count());

    return 0;
}

void LoadImages(std::string const&  strPathToSequence,
    std::vector<string>& vstrImageFilenames, std::vector<double>& vTimestamps)
{
    std::ifstream fTimes;
    std::string strPathTimeFile = strPathToSequence + "/times.txt";

    fTimes.open(strPathTimeFile.c_str());

    while (!fTimes.eof())
    {
        std::string s;
        std::getline(fTimes,s);

        if (s.empty())
            continue;

        std::stringstream ss;
        ss << s;
        double t;
        ss >> t;
        vTimestamps.push_back(t);
    }

    std::string strPrefixLeft = strPathToSequence + "/image_0/";

    std::size_t const nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (std::size_t i = 0; i < nTimes; ++i)
    {
        std::stringstream ss;
        ss << setfill('0') << std::setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
