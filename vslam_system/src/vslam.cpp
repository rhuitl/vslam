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

#include <vslam_system/vslam.h>
#include <cstdio>


using namespace sba;
using namespace pcl;

namespace vslam {

VslamSystem::VslamSystem(const std::string& vocab_tree_file, const std::string& vocab_weights_file,
  int min_keyframe_inliers, double min_keyframe_distance, double min_keyframe_angle)
  : frame_processor_(10), 
    vo_(boost::shared_ptr<pe::PoseEstimator>(new pe::PoseEstimator3d(1000,true,6.0,8.0,8.0)),
        40, 10, min_keyframe_inliers, min_keyframe_distance, min_keyframe_angle), // 40 frames, 10 fixed
    place_recognizer_(vocab_tree_file, vocab_weights_file),
    pose_estimator_(5000, true, 10.0, 3.0, 3.0)
{
  sba_.useCholmod(true);
  /// @todo Don't use windowed matching for place rec
  pose_estimator_.wx = 92;
  pose_estimator_.wy = 48;
  
  // pose_estimator_.windowed = false; // Commented out because this was breaking PR
  
  prInliers = 200;
  numPRs = 0;
  nSkip = 20;
}

bool VslamSystem::addFrame(const frame_common::CamParams& camera_parameters,
                           const cv::Mat& left, const cv::Mat& right, int nfrac)
{
  // Set up next frame and compute descriptors
  frame_common::Frame next_frame;
  next_frame.setCamParams(camera_parameters); // this sets the projection and reprojection matrices
  frame_processor_.setStereoFrame(next_frame, left, right, nfrac);
  next_frame.frameId = sba_.nodes.size(); // index
  next_frame.img = cv::Mat();   // remove the images
  next_frame.imgRight = cv::Mat();
  
  // Add frame to visual odometer
  bool is_keyframe = vo_.addFrame(next_frame);

  // grow full SBA
  if (is_keyframe)
  {
    addKeyframe(next_frame); 
  }

  if (frames_.size() > 1 && vo_.pose_estimator_->inliers.size() < 40)
    std::cout << std::endl << "******** Bad image match: " << std::endl << std::endl;

  return is_keyframe;
}

bool VslamSystem::addFrame(const frame_common::CamParams& camera_parameters,
                           const cv::Mat& left, const cv::Mat& right,
                           const pcl::PointCloud<PointXYZRGB>& ptcloud, int nfrac)
{
  // Set up next frame and compute descriptors
  frame_common::Frame next_frame;
  next_frame.setCamParams(camera_parameters); // this sets the projection and reprojection matrices
  frame_processor_.setStereoFrame(next_frame, left, right, nfrac);
  next_frame.frameId = sba_.nodes.size(); // index
  next_frame.img = cv::Mat();   // remove the images
  next_frame.imgRight = cv::Mat();
  
  if (pointcloud_processor_)
  {
    pointcloud_processor_->setPointcloud(next_frame, ptcloud);
    printf("[Pointcloud] set a pointcloud! %d\n", (int)next_frame.pointcloud.points.size());
  }
  
  // Add frame to visual odometer
  bool is_keyframe = vo_.addFrame(next_frame);

  // grow full SBA
  if (is_keyframe)
  {
    addKeyframe(next_frame);
    // Store the dense pointcloud.
    frames_.back().dense_pointcloud = ptcloud;
  }

  if (frames_.size() > 1 && vo_.pose_estimator_->inliers.size() < 40)
    std::cout << std::endl << "******** Bad image match: " << std::endl << std::endl;

  return is_keyframe;
}

void VslamSystem::addKeyframe(frame_common::Frame& next_frame)
{
    frames_.push_back(next_frame);
    vo_.transferLatestFrame(frames_, sba_);

    // Modify the transferred frame
    frame_common::Frame& transferred_frame = frames_.back();

    // Add any matches from place recognition
    // frameId indexes into frames
    std::vector<const frame_common::Frame*> place_matches;
    const size_t N = 5;
    place_recognizer_.findAndInsert(transferred_frame, transferred_frame.frameId, frames_, N, place_matches);
    printf("PLACEREC: Found %d matches\n", (int)place_matches.size());

    for (int i = 0; i < (int)place_matches.size(); ++i) 
    {
      frame_common::Frame& matched_frame = const_cast<frame_common::Frame&>(*place_matches[i]);
      
      // Skip if it's one of the previous nskip keyframes
      if (matched_frame.frameId >= (int)frames_.size() - nSkip - 1) 
      {
        //        printf("\tMatch %d: skipping, frame index %d\n", i, (int)matched_frame.frameId);
        continue;
      }

      // Geometric check for place recognizer
      int inliers = pose_estimator_.estimate(matched_frame, transferred_frame);
      //      printf("\tMatch %d: %d inliers, frame index %d\n", i, inliers, matched_frame.frameId);
      if (inliers > prInliers) 
	    {
	      numPRs++;

	      // More code copied from vo.cpp
	      Matrix<double,3,4> frame_to_world;
	      Node &matched_node = sba_.nodes[matched_frame.frameId];
	      Quaterniond fq0; /// @todo What does 'fq0' mean?
	      fq0 = matched_node.qrot;
	      transformF2W(frame_to_world, matched_node.trans, fq0);
            
	      addProjections(matched_frame, transferred_frame, frames_, sba_, pose_estimator_.inliers,
			     frame_to_world, matched_frame.frameId, transferred_frame.frameId);
	      printf("\t[PlaceRec] Adding PR link between frames %d and %d\n", transferred_frame.frameId, 
		     matched_frame.frameId);
	      double cst = sba_.calcRMSCost();
	      cout << endl << "*** RMS Cost: " << cst << endl << endl;
	      //        sba_.printStats();
	      if (cst > 4.0)
	        {
	          // fix all but last NNN frames
	          int n = sba_.nodes.size();
	          if (n < 100) n = 50;
	          else n = n - 50;
	          sba_.nFixed = n;
	          sba_.doSBA(3,1.0e-4,1);
	          sba_.nFixed = 1;
	        }
	    }
    }
}

void VslamSystem::refine(int initial_runs)
{
  /// @todo Make these arguments parameters?
  sba_.doSBA(initial_runs, 1.0e-4, 1);
  if (sba_.calcRMSCost() > 4.0)
    sba_.doSBA(10, 1.0e-4, 1);  // do more
  if (sba_.calcRMSCost() > 4.0)
    sba_.doSBA(10, 1.0e-4, 1);  // do more
}

} // namespace vslam
