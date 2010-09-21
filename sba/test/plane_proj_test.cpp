#include <Eigen3/Core>
#include <Eigen3/Geometry>
#include <Eigen3/LU>
#include <Eigen3/StdVector>

#include <stdio.h>

using namespace Eigen3;

int main()
{
  Vector3d point(1, 1, 1);
  Vector3d normal(0, 1, 0);
  
  Vector3d projpoint(-1, -2, 5);
  Vector3d w = point - projpoint;
  Vector3d proj = point - (w.dot(normal))*normal;
  
  printf("Projected point: %f %f %f\n", proj.x(), proj.y(), proj.z());
}
