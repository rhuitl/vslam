#include <ros/ros.h>
#include <ros/time.h>

// Messages
#include <sba/CameraNode.h>
#include <sba/Projection.h>
#include <geometry_msgs/PointStamped.h>
#include <sba/MergeTracks.h>
#include <sba/AddNode.h>
#include <sba/AddPoint.h>
#include <visualization_msgs/Marker.h>

#include <sba/sba.h>
#include <sba/visualization.h>

using namespace sba;

class SBANode
{
  public:
    SysSBA sba;
    ros::NodeHandle n;
    //ros::Subscriber node_sub;
    //ros::Subscriber point_sub;
    ros::Subscriber proj_sub;
    ros::ServiceServer merge_tracks_serv;
    ros::ServiceServer add_node_serv;
    ros::ServiceServer add_point_serv;
    ros::Publisher cam_marker_pub;
    ros::Publisher point_marker_pub;
    
    ros::Timer timer;
    
    void addNode(const sba::CameraNode::ConstPtr& msg)
    {
      Vector4d trans(msg->trans.translation.x, msg->trans.translation.y, msg->trans.translation.z, 1.0);
      Quaternion<double> qrot(msg->trans.rotation.x, msg->trans.rotation.y, msg->trans.rotation.z, msg->trans.rotation.w);
      
      frame_common::CamParams cam_params;
      cam_params.fx = msg->fx;
      cam_params.fy = msg->fy;
      cam_params.cx = msg->cx;
      cam_params.cy = msg->cy;
      cam_params.tx = msg->baseline;
      
      bool fixed = msg->fixed;
      
      sba.addNode(trans, qrot, cam_params, fixed);
    }
    
    void addPoint(const geometry_msgs::PointStamped::ConstPtr& msg)
    {
      Vector4d point(msg->point.x, msg->point.y, msg->point.z, 1.0);
      sba.addPoint(point);
    }
    
    void addProj(const sba::Projection::ConstPtr& msg)
    {
      int camindex = msg->camindex;
      int pointindex = msg->pointindex;
      Vector3d keypoint(msg->u, msg->v, msg->d);
      bool stereo = msg->stereo;
      
      // Make sure it's valid before adding it.
      if (pointindex < (int)sba.tracks.size() && camindex < (int)sba.nodes.size())
      {
        sba.addProj(camindex, pointindex, keypoint, stereo);
      }
      else
      {
        ROS_INFO("Failed to add projection: C: %d, P: %d, Csize: %d, Psize: %d", 
                camindex, pointindex,(int)sba.nodes.size(),(int)sba.tracks.size());       
      }
    }
    
    void doSBA(const ros::TimerEvent& event)
    {
      if (sba.nodes.size() > 0)
      {
        // Copied from vslam.cpp: refine()
        sba.doSBA(3, 1.0e-4, SBA_SPARSE_CHOLESKY);
        
        double cost = sba.calcRMSCost();
        
        if (!(cost == cost)) // is NaN?
        {
          ROS_INFO("NaN cost!");  
        }
        else
        { 
          if (sba.calcRMSCost() > 4.0)
            sba.doSBA(10, 1.0e-4, SBA_SPARSE_CHOLESKY);  // do more
          if (sba.calcRMSCost() > 4.0)
            sba.doSBA(10, 1.0e-4, SBA_SPARSE_CHOLESKY);  // do more
        }
      }
      
      unsigned int projs = 0;
      // For debugging.
      for (int i = 0; i < (int)sba.tracks.size(); i++)
      {
        projs += sba.tracks.size();
      }
      ROS_INFO("SBA Nodes: %d, Points: %d, Projections: %d", (int)sba.nodes.size(),
        (int)sba.tracks.size(), projs);
        
      // Visualization
      if (cam_marker_pub.getNumSubscribers() > 0 || point_marker_pub.getNumSubscribers() > 0)
      { 
         drawGraph(sba, cam_marker_pub, point_marker_pub);
      }
    }
    
    bool mergeTracks(sba::MergeTracks::Request &req,
                     sba::MergeTracks::Response &res)
    {
      res.trackid = sba.mergeTracksSt(req.trackid0, req.trackid1);
      
      ROS_INFO("Merging tracks %d and %d into track %d.", 
                req.trackid0, req.trackid1, res.trackid);
      
      return true;
    }
    
    bool addNodeServ(sba::AddNode::Request &req,
                     sba::AddNode::Response &res)
    {
      sba::CameraNode msg = req.node;
      
      Vector4d trans(msg.trans.translation.x, msg.trans.translation.y, msg.trans.translation.z, 1.0);
      Quaternion<double> qrot(msg.trans.rotation.x, msg.trans.rotation.y, msg.trans.rotation.z, msg.trans.rotation.w);
      
      frame_common::CamParams cam_params;
      cam_params.fx = msg.fx;
      cam_params.fy = msg.fy;
      cam_params.cx = msg.cx;
      cam_params.cy = msg.cy;
      cam_params.tx = msg.baseline;
      bool fixed = msg.fixed;
      
      res.index = sba.addNode(trans, qrot, cam_params, fixed);
      
      return true;
    }
    
    bool addPointServ(sba::AddPoint::Request &req,
                      sba::AddPoint::Response &res)
    {
      geometry_msgs::PointStamped msg = req.point;
      Vector4d point(msg.point.x, msg.point.y, msg.point.z, 1.0);
      res.index = sba.addPoint(point);
      
      return true;
    }
  
    SBANode()
    {
      // Subscribe to topics.
      // node_sub = n.subscribe<sba::CameraNode>("/sba/nodes", 5000, &SBANode::addNode, this);
      // point_sub = n.subscribe<geometry_msgs::PointStamped>("/sba/points", 5000, &SBANode::addPoint, this);
      proj_sub = n.subscribe<sba::Projection>("/sba/projections", 5000, &SBANode::addProj, this);
      
      // Advertise services.
      add_node_serv = n.advertiseService("/sba/add_node", &SBANode::addNodeServ, this);
      add_point_serv = n.advertiseService("/sba/add_point", &SBANode::addPointServ, this);
      merge_tracks_serv = n.advertiseService("/sba/merge_tracks", &SBANode::mergeTracks, this);
      
      // Advertise visualization topics.
      cam_marker_pub = n.advertise<visualization_msgs::Marker>("/sba/cameras", 1);
      point_marker_pub = n.advertise<visualization_msgs::Marker>("/sba/points", 1);

      timer = n.createTimer(ros::Duration(10), &SBANode::doSBA, this);
      
      sba.useCholmod(true);
    }
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "sba_node");
  SBANode sbanode;
  ros::spin();
}
