#include <ros/ros.h>
#include <sba/sba.h>
#include <sba/visualization.h>


// For random seed.
#include <time.h>

#include <visualization_msgs/Marker.h>

using namespace sba;
using namespace std;

const double PI = 3.141592;

class Plane
{
  public:
    vector<Point, Eigen::aligned_allocator<Point> > points;
    Eigen::Vector3d normal;
    
    void rotate(const Eigen::Quaterniond& qrot)
    {
      Eigen::Matrix3d rotmat = qrot.toRotationMatrix();
      
      for (unsigned int i = 0; i < points.size(); i++)
      {
        points[i].start<3>() = rotmat*points[i].start<3>();
      }
      
      normal = rotmat*normal;
    }
    
    void rotate(double angle, double x, double y, double z)
    {
      Eigen::AngleAxis<double> angleaxis(angle, Vector3d(x, y, z));
      rotate(Eigen::Quaterniond(angleaxis));
    }
         
    void translate(const Eigen::Vector3d& trans)
    {
      for (unsigned int i = 0; i < points.size(); i++)
      {
        points[i].start<3>() += trans;
      }
    }
    
    void translate(double x, double y, double z)
    {
      Vector3d trans(x, y, z);
      translate(trans);
    }
    
    // Creates a plane with origin at (0,0,0) and opposite corner at (width, height, 0).
    void resize(double width, double height, int nptsx, int nptsy)
    {
      double w2 = width/2.0;
      double h2 = height/2.0;
      for (int ix = 0; ix < nptsx ; ix++)
      {
        for (int iy = 0; iy < nptsy ; iy++)
        {
          // Create a point on the plane in a grid.
          points.push_back(Point(width/nptsx*(ix+.5)-w2, -height/nptsy*(iy+.5)+h2, 0.0, 1.0));
        }
      }
      
      normal << 0, 0, -1;
    }
};


void setupSBA(SysSBA &sys)
{
    // Create camera parameters.
    frame_common::CamParams cam_params;
    cam_params.fx = 430; // Focal length in x
    cam_params.fy = 430; // Focal length in y
    cam_params.cx = 320; // X position of principal point
    cam_params.cy = 240; // Y position of principal point
    cam_params.tx = -30; // Baseline (no baseline since this is monocular)

    // Define dimensions of the image.
    int maxx = 640;
    int maxy = 480;

    // Create a plane containing a wall of points.
    Plane middleplane;
    middleplane.resize(3, 2, 10, 5);
    middleplane.rotate(-PI/4.0, 0, 1, 0);
    middleplane.translate(0.0, 0.0, 5.0);
    
    // Vector containing the true point positions.
    vector<Point, Eigen::aligned_allocator<Point> > points;
    vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > normals;
    
    points.insert(points.end(), middleplane.points.begin(), middleplane.points.end());
    normals.insert(normals.end(), middleplane.points.size(), middleplane.normal);
    
    // Create nodes and add them to the system.
    unsigned int nnodes = 2; // Number of nodes.
    double path_length = 0.5; // Length of the path the nodes traverse.

    // Set the random seed.
    unsigned short seed = (unsigned short)time(NULL);
    seed48(&seed);
    
    unsigned int i = 0;
    
    Vector3d inormal0(0, 0, 1);
    Vector3d inormal1(0, 0, 1);
    
    for (i = 0; i < nnodes; i++)
    { 
      // Translate in the x direction over the node path.
      Vector4d trans(i/(nnodes-1.0)*path_length, 0, 0, 1);
            
#if 1
      if (i >= 0)
	{
	  // perturb a little
	  double tnoise = 0.5;	// meters
	  trans.x() += tnoise*(drand48()-0.5);
	  trans.y() += tnoise*(drand48()-0.5);
	  trans.z() += tnoise*(drand48()-0.5);
	}
#endif

      // Don't rotate.
      Quaterniond rot(1, 0, 0, 0);
#if 1
      if (i >= 0)
	{
	  // perturb a little
	  double qnoise = 0.1;	// meters
	  rot.x() += qnoise*(drand48()-0.5);
	  rot.y() += qnoise*(drand48()-0.5);
	  rot.z() += qnoise*(drand48()-0.5);
	}
#endif
      rot.normalize();
      
      // Add a new node to the system.
      sys.addNode(trans, rot, cam_params, false);
      
      // set normal
      if (i == 0)
	inormal0 = rot.toRotationMatrix().transpose() * inormal0;
      else
	inormal1 = rot.toRotationMatrix().transpose() * inormal1;
    }
        
    double pointnoise = 1.0;
    
    // Add points into the system, and add noise.
    for (i = 0; i < points.size(); i++)
    {
      // Add up to .5 points of noise.
      Vector4d temppoint = points[i];
      temppoint.x() += pointnoise*(drand48() - 0.5);
      temppoint.y() += pointnoise*(drand48() - 0.5);
      temppoint.z() += pointnoise*(drand48() - 0.5);
      sys.addPoint(temppoint);
    }
    
    // Each point gets added twice
    for (i = 0; i < points.size(); i++)
    {
      // Add up to .5 points of noise.
      Vector4d temppoint = points[i];
      temppoint.x() += pointnoise*(drand48() - 0.5);
      temppoint.y() += pointnoise*(drand48() - 0.5);
      temppoint.z() += pointnoise*(drand48() - 0.5);
      sys.addPoint(temppoint);
    }

    Vector2d proj2d, proj2dp;
    Vector3d proj, projp, pc, pcp, baseline, baselinep;
    
    Vector3d imagenormal(0, 0, 1);
    
    Matrix3d covar0;
    covar0 << sqrt(imagenormal(0)), 0, 0, 0, sqrt(imagenormal(1)), 0, 0, 0, sqrt(imagenormal(2));
    Matrix3d covar;
    
    Quaterniond rotation;
    Matrix3d rotmat;
    
    unsigned int midindex = middleplane.points.size();
    printf("Normal for Middle Plane: [%f %f %f], index %d -> %d\n", middleplane.normal.x(), middleplane.normal.y(), middleplane.normal.z(), 0, midindex-1);
    
    int nn = points.size();

    // Project points into nodes.
    for (i = 0; i < points.size(); i++)
    {
      int k=i;
      // Project the point into the node's image coordinate system.
      sys.nodes[0].setProjection();
      sys.nodes[0].project2im(proj2d, points[k]);
      sys.nodes[1].setProjection();
      sys.nodes[1].project2im(proj2dp, points[k]);
        
      // Camera coords for right camera
      baseline << sys.nodes[0].baseline, 0, 0;
      pc = sys.nodes[0].Kcam * (sys.nodes[0].w2n*points[k] - baseline); 
      proj.start<2>() = proj2d;
      proj(2) = pc(0)/pc(2);
        
      baseline << sys.nodes[1].baseline, 0, 0;
      pcp = sys.nodes[1].Kcam * (sys.nodes[1].w2n*points[k] - baseline); 
      projp.start<2>() = proj2dp;
      projp(2) = pcp(0)/pcp(2);



      // If valid (within the range of the image size), add the stereo 
      // projection to SBA.
      if (proj.x() > 0 && proj.x() < maxx && proj.y() > 0 && proj.y() < maxy)
        {
          sys.addStereoProj(0, k, proj);
          sys.addStereoProj(1, k, projp);
          sys.addStereoProj(0, k+nn, proj);
          sys.addStereoProj(1, k+nn, projp);
          
          // Create the covariance matrix: 
          // image plane normal = [0 0 1]
          // wall normal = [0 0 -1]
          // covar = (R)T*[0 0 0;0 0 0;0 0 1]*R
          
          rotation.setFromTwoVectors(inormal0, normals[k]);
          rotmat = rotation.toRotationMatrix();
          covar = rotmat.transpose()*covar0*rotmat;
	  sys.setProjCovariance(0, k+nn, covar);

          rotation.setFromTwoVectors(inormal1, normals[k]);
          rotmat = rotation.toRotationMatrix();
          covar = rotmat.transpose()*covar0*rotmat;
	  sys.setProjCovariance(1, k, covar);
        }
      else
	cout << "ERROR! point not in view of nodes" << endl;
    }

    // Add noise to node position.
    
    double transscale = 2.0;
    double rotscale = 0.2;
    
    // Don't actually add noise to the first node, since it's fixed.
    for (i = 1; i < sys.nodes.size(); i++)
    {
      Vector4d temptrans = sys.nodes[i].trans;
      Quaterniond tempqrot = sys.nodes[i].qrot;
      
      // Add error to both translation and rotation.
      temptrans.x() += transscale*(drand48() - 0.5);
      temptrans.y() += transscale*(drand48() - 0.5);
      temptrans.z() += transscale*(drand48() - 0.5);
      tempqrot.x() += rotscale*(drand48() - 0.5);
      tempqrot.y() += rotscale*(drand48() - 0.5);
      tempqrot.z() += rotscale*(drand48() - 0.5);
      tempqrot.normalize();
      
      sys.nodes[i].trans = temptrans;
      sys.nodes[i].qrot = tempqrot;
      
      // These methods should be called to update the node.
      sys.nodes[i].normRot();
      sys.nodes[i].setTransform();
      sys.nodes[i].setProjection();
      sys.nodes[i].setDr(true);
    }
        
}

void processSBA(ros::NodeHandle node)
{
    // Create publisher topics.
    ros::Publisher cam_marker_pub = node.advertise<visualization_msgs::Marker>("/sba/cameras", 1);
    ros::Publisher point_marker_pub = node.advertise<visualization_msgs::Marker>("/sba/points", 1);
    ros::spinOnce();
    
    //ROS_INFO("Sleeping for 2 seconds to publish topics...");
    ros::Duration(0.2).sleep();
    
    // Create an empty SBA system.
    SysSBA sys;
    
    setupSBA(sys);
    
    // Provide some information about the data read in.
    unsigned int projs = 0;
    // For debugging.
    for (int i = 0; i < (int)sys.tracks.size(); i++)
    {
      projs += sys.tracks[i].projections.size();
    }
    ROS_INFO("SBA Nodes: %d, Points: %d, Projections: %d", (int)sys.nodes.size(),
      (int)sys.tracks.size(), projs);
        
    //ROS_INFO("Sleeping for 5 seconds to publish pre-SBA markers.");
    //ros::Duration(5.0).sleep();
        
    // Perform SBA with 10 iterations, an initial lambda step-size of 1e-3, 
    // and using CSPARSE.
    sys.doSBA(1, 1e-4, SBA_SPARSE_CHOLESKY);
    
    int npts = sys.tracks.size();

    ROS_INFO("Bad projs (> 10 pix): %d, Cost without: %f", 
        (int)sys.countBad(10.0), sqrt(sys.calcCost(10.0)/npts));
    ROS_INFO("Bad projs (> 5 pix): %d, Cost without: %f", 
        (int)sys.countBad(5.0), sqrt(sys.calcCost(5.0)/npts));
    ROS_INFO("Bad projs (> 2 pix): %d, Cost without: %f", 
        (int)sys.countBad(2.0), sqrt(sys.calcCost(2.0)/npts));
    
    ROS_INFO("Cameras (nodes): %d, Points: %d",
        (int)sys.nodes.size(), (int)sys.tracks.size());
        
    // Publish markers
    drawGraph(sys, cam_marker_pub, point_marker_pub, 1, sys.tracks.size()/2);
    ros::spinOnce();
    //ROS_INFO("Sleeping for 2 seconds to publish post-SBA markers.");
    ros::Duration(0.2).sleep();

    for (int j=0; j<50; j++)
      {
	sys.doSBA(1, 0, SBA_SPARSE_CHOLESKY);
	drawGraph(sys, cam_marker_pub, point_marker_pub, 1, sys.tracks.size()/2);
	ros::spinOnce();
	ros::Duration(0.2).sleep();
      }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "sba_system_setup");
    
    ros::NodeHandle node;
    
    processSBA(node);
    ros::spinOnce();

    return 0;
}

