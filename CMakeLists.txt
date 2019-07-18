cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2)

add_compile_options(-bigobj)
add_compile_options($<IF:$<CONFIG:Debug>,/MTd,/MT>)

add_definitions(-DINITGUID)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${Pangolin_INCLUDE_DIRS}
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\tests\\eigen_test\\Eigen"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\include"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\include")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME}
    src/System.cc
    src/Tracking.cc
    src/LocalMapping.cc
    src/LoopClosing.cc
    src/ORBextractor.cc
    src/ORBmatcher.cc 
    src/FrameDrawer.cc
    src/Converter.cc
    src/MapPoint.cc
    src/KeyFrame.cc
    src/Map.cc
    src/MapDrawer.cc
    src/Optimizer.cc
    src/PnPsolver.cc
    src/Frame.cc
    src/KeyFrameDatabase.cc
    src/Sim3Solver.cc
    src/Initializer.cc
    src/Viewer.cc
    )

set(OpenCV_LIBS_Debug
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ade.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\IlmImfd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ippicvmt.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ippiwd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ittnotifyd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libjasperd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libjpeg-turbod.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libpngd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libprotobufd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libtiffd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libwebpd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_calib3d411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_core411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_dnn411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_features2d411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_flann411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_gapi411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_highgui411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_imgcodecs411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_imgproc411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_ml411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_objdetect411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_photo411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_stitching411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_video411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_videoio411d.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\quircd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\zlibd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\pangolin.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\glewd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\libpng16_staticd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\jpeg.lib"
"OpenGL32.Lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\ORB_SLAM2\\Thirdparty\\DBoW2\\lib\\Debug\\DBoW2.lib"
"Mfplat.lib"
"Mf.lib"
"Mfreadwrite.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\zlibstaticd.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\ORB_SLAM2\\Thirdparty\\g2o\\bin\\Debug\\g2o.lib"
"strmiids.lib"
)

set(OpenCV_LIBS_Release
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ade.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\IlmImf.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ippicvmt.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ippiw.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\ittnotify.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libjasper.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libjpeg-turbo.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libpng.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libprotobuf.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libtiff.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\libwebp.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_calib3d411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_core411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_dnn411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_features2d411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_flann411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_gapi411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_highgui411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_imgcodecs411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_imgproc411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_ml411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_objdetect411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_photo411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_stitching411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_video411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\opencv_videoio411.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\quirc.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\opencv\\build\\install\\x64\\vc15\\staticlib\\zlib.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\pangolin.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\glew.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\libpng16_static.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\jpeg.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\Pangolin\\build\\install\\lib\\zlibstatic.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\ORB_SLAM2\\Thirdparty\\g2o\\bin\\Release\\g2o.lib"
"C:\\Users\\ArmenPoghosovSystemA\\Documents\\Work\\Learning\\ORB_SLAM2\\Thirdparty\\DBoW2\\lib\\Release\\DBoW2.lib"
"OpenGL32.Lib"
"Mfplat.lib"
"Mf.lib"
"Mfreadwrite.lib"
"strmiids.lib"
)

target_link_libraries(${PROJECT_NAME} "$<IF:$<CONFIG:Debug>,${OpenCV_LIBS_Debug},${OpenCV_LIBS_Release}>")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti Examples/Stereo/stereo_kitti.cc )
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_kitti_segmented Examples/Monocular/mono_kitti_segmented.cc)
target_link_libraries(mono_kitti_segmented ${PROJECT_NAME})

add_executable(mono_euroc Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})


