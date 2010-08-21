/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2010, Willow Garage, Inc.
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
 *   * Neither the name of Willow Garage, Inc. nor the names of its
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
 * $Id: frame_extended.h 43557 2010-08-21 00:16:14Z mihelich $
 *
 */

#ifndef _FRAME_EXTENDED_H_
#define _FRAME_EXTENDED_H_

#include "frame_common/frame.h"
// PCL header
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/organized_data.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/transform.h>
#include <pcl/registration/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>

#include <pcl/io/pcd_io.h>

using namespace pcl;
using namespace Eigen;

namespace frame_common
{
  /** \brief Extended Frame class. */
  class FrameExtended : public Frame
  {
    public:
      /** \brief Point Cloud dataset. Contains: XYZ data + NormalXYZ data + Surface flatness (curvature) measure. */
      pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;

      /** \brief Set of point indices that correspond to the data that we care about in \href cloud. */
      pcl::PointIndices indices;

      /** \brief Estimates point normals and surface flatness measures for point-to-plane matching. */
      template <typename PointT, typename PointNT> void
        estimateNormals (const pcl::PointCloud<PointT> &input, const pcl::PointIndices &indices, pcl::PointCloud<PointNT> &output)
      {
        // Create an object for spatial search (e.g., nearest neighbor estimation)
        pcl::OrganizedDataIndex<PointT> tree;

        // Create a normal estimation object
        pcl::NormalEstimation<PointT, PointNT> ne;
        ne.setInputCloud (boost::make_shared<const pcl::PointCloud<PointT> > (input));      // pass the data
        ne.setIndices (boost::make_shared<std::vector<int> > (indices.indices));            // pass the indices
        ne.setSearchMethod (tree);                                                          // pass the spatial search
        // Use 10 nearest neighbors to estimate the normals
        ne.setKSearch (10);

        ne.compute (output);
      }
      
      /** \brief Match points with previous frame, given an initial pose estimate of trans and rot. */
      void match(frame_common::FrameExtended& frame, const Eigen::Vector3d& trans, const Eigen::Quaterniond& rot, std::vector<pe::Match>& matches)
      {
        PointCloud<PointXYZRGBNormal> transformed_cloud;
        
        // First, transform the current frame. (Is this inverse?) (Or just transform the other cloud?)
        transformPointCloudWithNormals<PointXYZRGBNormal>(frame.cloud, transformed_cloud, -trans.cast<float>(), rot.cast<float>().conjugate());
        
        Matrix3d rotmat = rot.toRotationMatrix();
        
        /* printf("Translation: [%f %f %f] Rotation: [%f %f %f; %f %f %f; %f %f %f]\n", 
            trans(0), trans(1), trans(2), rotmat(0), rotmat(1), rotmat(2), rotmat(3), rotmat(4), 
            rotmat(5), rotmat(6), rotmat(7), rotmat(8)); */
        
        // Optional/TODO: Perform ICP to further refine estimate.
        /*PointCloud<PointXYZRGBNormal> cloud_reg;

        IterativeClosestPointNonLinear<PointXYZRGBNormal, PointXYZRGBNormal> reg;
        reg.setInputCloud (boost::make_shared<const PointCloud<PointXYZRGBNormal> > (transformed_cloud));
        reg.setInputTarget (boost::make_shared<const PointCloud<PointXYZRGBNormal> > (cloud));
        reg.setMaximumIterations(50);
        reg.setTransformationEpsilon (1e-8);

        reg.align(cloud_reg); */
              
        // Find matches between pointclouds in frames. (TODO: also compare normals)
        std::vector<int> other_indices, this_indices;
        getMatchingIndices(transformed_cloud, cloud, other_indices, this_indices);
        
        // Fill in keypoints and projections of relevant features.
        // Currently just done when setting the pointcloud.
        
        // Convert matches into the correct format.
        matches.clear();
        // Starting at i=1 as a hack to not let through (0,0,0) matches (why is this in the ptcloud?))
        for (unsigned int i=1; i < other_indices.size(); i++)
        {           
          PointXYZRGBNormal &pt0 = transformed_cloud.points[other_indices[i]];
          PointXYZRGBNormal &pt1 = cloud.points[this_indices[i]];
          
          // Figure out distance and angle between normals
          Quaterniond normalquat;
          Vector3d norm0(pt0.normal[0], pt0.normal[1], pt0.normal[2]), norm1(pt1.normal[0], pt1.normal[1], pt1.normal[2]);
          normalquat.setFromTwoVectors(norm0, norm1);
          double angledist = normalquat.angularDistance(normalquat.Identity());
          double dist = (Vector3d(pt0.x, pt0.y, pt0.z)-Vector3d(pt1.x, pt1.y, pt1.z)).norm();
          
          //printf("[Normal]: %f %f %f [Match Normal]: %f %f %f\n", pt0.normal[0], pt0.normal[1], pt0.normal[2], pt1.normal[0], pt1.normal[1], pt1.normal[2]);
          
          //printf("Angular distance: %f, Distance: %f\n", angledist, dist);
          
          // Projection of point 0 in node 1: proj(p0, n1)
          // Projection of point 1 in node 1: keypoint_p1
          Vector4d p0_pt = Vector4d(pt0.x, pt0.y, pt0.z, 1.0);
          Vector3d expected_proj = projectPoint(p0_pt);
          
          Vector3d diff = expected_proj - pl_kpts[this_indices[i]].start<3>();
          diff(2) = diff(2) - diff(0);
          
          /* printf("[Expected projection]: %f %f %f [Actual projection]: %f %f %f [Difference] %f %f %f\n",
            expected_proj.x(), expected_proj.y(), expected_proj.z()-expected_proj.x(),
            pl_kpts[this_indices[i]].x(), pl_kpts[this_indices[i]].y(), pl_kpts[this_indices[i]].z()-pl_kpts[this_indices[i]].x(),
            diff(0), diff(1), diff(2)); */
          
          
          if ((norm0 - norm1).norm() < 0.1)
            matches.push_back(pe::Match(other_indices[i], this_indices[i], dist));
        }
        
        printf("[FrameExtended] Found %d matches, then converted %d matches.\n", (int)other_indices.size(), (int)matches.size());
      }
      
      /** \brief Add a pointcloud to the frame, doing all the necessary 
      pre-processing (downsampling, computing normals, and filering based on curvature). */
      void setPointcloud(const pcl::PointCloud<pcl::PointXYZRGB>& input_cloud)
      {
        //pcl::io::savePCDFileASCII("input_cloud.pcd", input_cloud);
        reduceCloud(input_cloud, cloud);
        
        // For now, let's keep a 1-1 mapping between pl_pts, keypts, etc., etc.
        // Basically replicating all the info in the pointcloud but whatever.
        // TODO: Do something more intelligent than this.
        pl_pts.clear();
        pl_kpts.clear();
        pl_normals.clear();
        pl_ipts.clear();
        
        unsigned int ptcloudsize = cloud.points.size();
        pl_pts.resize(ptcloudsize);
        pl_kpts.resize(ptcloudsize);
        pl_normals.resize(ptcloudsize);
        pl_ipts.resize(ptcloudsize);
        
        for (unsigned int i=0; i < cloud.points.size(); i++)
        {
          PointXYZRGBNormal &pt = cloud.points[i];
          //Eigen::Vector4f normal;
          //pcl::flipNormalTowardsViewpoint<PointXYZRGBNormal>(pt, 0, 0, 0, normal);
          
          pl_pts[i] = Eigen::Vector4d(pt.x, pt.y, pt.z, 1.0);
          pl_normals[i] = Eigen::Vector4d(pt.normal[0], pt.normal[1], pt.normal[2], 1.0);
          pl_kpts[i] = projectPoint(pl_pts[i]);
          pl_ipts[i] = -1;
        }
      }
      
    private:
      void reduceCloud(const PointCloud<PointXYZRGB>& input, PointCloud<PointXYZRGBNormal>& output)
      {
        PointCloud<PointXYZRGB> cloud_nan_filtered, cloud_box_filtered, cloud_voxel_reduced;
        PointCloud<Normal> normals;
        PointCloud<PointXYZRGBNormal> cloud_normals;
        
        std::vector<int> indices;
        
        // Filter out nans.
        removeNaNFromPointCloud(input, cloud_nan_filtered, indices);
        indices.clear();
        
        // Filter out everything outside a [200x200x200] box.
        Eigen::Vector4f min_pt(-100, -100, -100, -100);
        Eigen::Vector4f max_pt(100, 100, 100, 100);
        getPointsInBox(cloud_nan_filtered, min_pt, max_pt, indices);
        
        ExtractIndices<PointXYZRGB> boxfilter;
        boxfilter.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGB> >(cloud_nan_filtered));
        boxfilter.setIndices (boost::make_shared<vector<int> > (indices));
        boxfilter.filter(cloud_box_filtered);
        
        // Reduce pointcloud
        VoxelGrid<PointXYZRGB> voxelfilter;
        voxelfilter.setInputCloud (boost::make_shared<const PointCloud<PointXYZRGB> > (cloud_box_filtered));
        voxelfilter.setLeafSize (0.05, 0.05, 0.05);
        voxelfilter.filter (cloud_voxel_reduced);
        
        // Compute normals
        NormalEstimation<PointXYZRGB, Normal> normalest;
        normalest.setViewPoint(0, 0, 0);
        normalest.setSearchMethod (boost::make_shared<KdTreeANN<PointXYZRGB> > ());
        //normalest.setKSearch (10);
        normalest.setRadiusSearch (0.25);
        normalest.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGB> >(cloud_voxel_reduced));
        normalest.compute(normals);
        
        pcl::concatenateFields (cloud_voxel_reduced, normals, cloud_normals);

        // Filter based on curvature
        PassThrough<PointXYZRGBNormal> normalfilter;
        normalfilter.setFilterFieldName("curvature");
        normalfilter.setFilterLimits(0.0, 0.1);
        normalfilter.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(cloud_normals));
        normalfilter.filter(output);
      }
      
      void getMatchingIndices(PointCloud<PointXYZRGBNormal>& input, PointCloud<PointXYZRGBNormal>& output, 
                std::vector<int>& input_indices, std::vector<int>& output_indices)
      {
        // TODO: Don't calculate the KDTree each time.
        KdTreeANN<PointXYZRGBNormal> input_tree, output_tree;
          
        input_tree.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(input));
        output_tree.setInputCloud(boost::make_shared<const PointCloud<PointXYZRGBNormal> >(output));
        
        pcl::io::savePCDFileASCII("plane_frame0.pcd", input);
        pcl::io::savePCDFileASCII("plane_frame1.pcd", output);
        
        // Iterate over the output tree looking for all the input points and finding
        // nearest neighbors.
        for (unsigned int i = 0; i < input.points.size(); i++)
        {
          PointXYZRGBNormal input_pt = input.points[i];
          std::vector<int> input_indexvect(1), output_indexvect(1); // Create a vector of size 1.
          std::vector<float> input_distvect(1), output_distvect(1);
          
          // Find the nearest neighbor of the input point in the output tree.
          output_tree.nearestKSearch(input_pt, 1, input_indexvect, input_distvect);
          
          PointXYZRGBNormal output_pt = output.points[input_indexvect[0]];
          
          // Find the nearest neighbor of the output point in the input tree.
          input_tree.nearestKSearch(output_pt, 1, output_indexvect, output_distvect);
          
          // If they match, add them to the match vectors.
          if (output_indexvect[0] == (int)i)
          {
            input_indices.push_back(i);
            output_indices.push_back(input_indexvect[0]);
          }
        }
      }
      
      Eigen::Vector3d projectPoint(Eigen::Vector4d& point)
      {
        Eigen::Vector3d keypoint;
        
        keypoint(0) = (cam.fx*point.x()) / point.z() + cam.cx;
        keypoint(1) = (cam.fy*point.y()) / point.z() + cam.cy;
        keypoint(2) = (cam.fx*(point.x()-cam.tx)/point.z() + cam.cx);
        
        return keypoint;
      }
  };
  
  void drawVOtracks(const cv::Mat &image,
                  const std::vector<FrameExtended, Eigen::aligned_allocator<FrameExtended> > &frames,
                  cv::Mat &display);

}
#endif // _FRAME_EXTENDED_H_

