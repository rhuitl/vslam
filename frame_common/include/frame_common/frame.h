/*********************************************************************
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
*********************************************************************/

//
// Visual Odometry classes and functions
//

#ifndef _FRAME_H_
#define _FRAME_H_

#ifndef EIGEN_USE_NEW_STDVECTOR
#define EIGEN_USE_NEW_STDVECTOR
#endif // EIGEN_USE_NEW_STDVECTOR

#include <stdio.h>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <vector>
#include <boost/shared_ptr.hpp>
#include <frame_common/stereo.h>
#include <frame_common/camparams.h>

// PCL headers
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/organized_data.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transform.h>
#include <pcl/registration/transforms.h>

#include <pcl/io/pcd_io.h>


// Forward declaration.
namespace pe { struct Match; };

namespace frame_common
{

  /// \brief The frame class hold image frames, and features found and matched within
  /// those frames.
  /// Optional image itself using OpenCV format.
  /// Camera params contain intrinsic params.
  /// Projection matrix is formed from cam params.
  class Frame
  {
  public:    
    int frameId; ///< Internal ID of the frame.

    // image section
    bool isStereo;              ///< True if the frame contains a stereo image pair.
    cv::Mat img;                ///< Image itself, if needed; can be empty.
    CamParams cam;              ///< Camera parameters.
    Eigen::Matrix3d iproj;      ///< Camera projection matrix.

    // feature section: keypoints and corresponding descriptors
    std::vector<cv::KeyPoint> kpts; /// Keypoints detected in the image.
    cv::Mat dtors;              /// Descriptors for the keypoints.

    // stereo
    cv::Mat imgRight;             ///< Right image (if stereo pair), can be empty.
    Eigen::Matrix4d disp_to_cart; ///< Transformation from disparity to cartesian.
    Eigen::Matrix4d cart_to_disp; ///< Transformation from cartesian to disparity.
    void setCamParams(const CamParams &c); ///< Set the camera parameters of the frame.

    // these are for stereo; do we need monocular ones?
    Eigen::Vector3d cam2pix(const Eigen::Vector3d &cam_coord) const;
    Eigen::Vector3d pix2cam(const Eigen::Vector3d &pix_coord) const;
    Eigen::Vector3d pix2cam(const cv::KeyPoint &pix_coord, double disp) const;

    /// \brief 3d points, linked to keypoints and SBA points for point-to-point matches.
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > pts;
    std::vector<char> goodPts;  ///< whether the points are good or not
    std::vector<double> disps;  ///< disparities
    std::vector<int> ipts;      ///< index into SBA system points; -1 if not present

    // these are for point-to-plane matches
    /// Pointcloud storage.
    pcl::PointCloud<pcl::PointXYZRGBNormal> pointcloud;
    
    /// Keypoints for pointcloud points as u, v, u-d.
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > pl_kpts;
    /// Points for point-plane matches.
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > pl_pts;
    /// Normals for point-plane matches.
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > pl_normals;
    std::vector<int> pl_ipts;  ///< Index into SBA system points; -1 if not present.
     
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW // needed for 16B alignment
  };


  /// \brief FrameProc is a processing engine for setting keypoints and
  ///   descriptors in a frame.
  class FrameProc
  {
  public:
    /// Create a FrameProc object using FAST feature detector (with a given threshold v)
    /// and SURF feature extractor.
    FrameProc(int v = 25);

    /// Feature Detector used for finding features in the image.
    cv::Ptr<cv::FeatureDetector> detector;
    /// Descriptor Extractor used for getting descriptors around image features.
    cv::Ptr<cv::DescriptorExtractor> extractor;

    /// Set the feature detector.
    void setFrameDetector(const cv::Ptr<cv::FeatureDetector>& detector);
    /// Set the descriptor extractor.
    void setFrameDescriptor(const cv::Ptr<cv::DescriptorExtractor>& extractor);

    /// \brief Set up a monocular frame from the passed in OpenCV image and fill 
    /// the given frame.
    /// \param frame The frame to process.
    /// \param img   Monocular image to process.
    void setMonoFrame(Frame& frame, const cv::Mat &img);

    /// \brief Do stereo processing on the frame, assuming descriptors and
    /// extractors are already set.
    /// \param frame The frame to process.
    /// \param nfrac Fractional disparity. If above 0, then stereo disparities
    ///              have already been passed in.
    void setStereoPoints(Frame &frame, int nfrac = 0);
    int ndisp;                  ///< Number of disparities to search.
    bool doSparse;              ///< True if using sparse stereo.

    /// \brief Set up stereo frame, assumes frame has camera parameters already set.
    /// \param frame The frame to be processed.
    /// \param img   Left camera image.
    /// \param imgr  Right camera image.
    /// \param nfrac Fractional disparity. If above 0, then imgr is a disparity
    ///              image instead.
    void setStereoFrame(Frame &frame, const cv::Mat &img, const cv::Mat &imgr, int nfrac = 0);
  };
  
  class PointcloudProc
  {
    public:
      // TODO: Have an initializer with some parameters here.
      
      /// \brief Add a pointcloud to the frame, doing all the necessary 
      /// pre-processing (downsampling, computing normals, and filering based on curvature).
      void setPointcloud(Frame &frame, const pcl::PointCloud<pcl::PointXYZRGB>& input_cloud);
      
      /// \brief Match points with previous frame, given an initial pose estimate.
      void match(const Frame& frame0, const Frame& frame1, 
                  const Eigen::Vector3d& trans, const Eigen::Quaterniond& rot, 
                  std::vector<pe::Match>& matches);
      
    private:
      /// \brief Subsample cloud for faster matching and processing, while
      /// filling in normals.
      void reduceCloud(const pcl::PointCloud<pcl::PointXYZRGB>& input, 
                        pcl::PointCloud<pcl::PointXYZRGBNormal>& output);

      /// \brief Project a 3D point into the image frame.
      Eigen::Vector3d projectPoint(Eigen::Vector4d& point, CamParams cam);
      
      /// \brief Find matches between two pointclouds using nearest neighbor
      /// KDtree search.
      void getMatchingIndices(const pcl::PointCloud<pcl::PointXYZRGBNormal>& input, 
          const pcl::PointCloud<pcl::PointXYZRGBNormal>& output, 
          std::vector<int>& input_indices, std::vector<int>& output_indices);
  };

  /// \brief Draw tracks from visual odometry on image over all given frames.
  /// \param image   Input image to draw over.
  /// \param frames  Frames containing tracks to draw.
  /// \param display Output image.
  void drawVOtracks(const cv::Mat &image,
                    const std::vector<Frame, Eigen::aligned_allocator<Frame> > &frames,
                    cv::Mat &display);

} // end of namespace frame_common

/// @todo Replace this with cv::DMatch. Moved here as quickest way to fix frame_extended.
namespace pe
{
  struct Match
  {
    int index1;      //!< The index of the descriptor from the first set.
    int index2;      //!< The index of the descriptor from the second set.
    float distance;

    Match(int index1, int index2, float distance = 0.0f)
      : index1(index1), index2(index2), distance(distance)
    {
    }
  };
} // namespace pe

#endif // _FRAME_H_

