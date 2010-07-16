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
// Running reduced pose system
//

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "sba/read_spa.h"
#include "sba/sba.h"
#include <Eigen/Cholesky>

using namespace Eigen;
using namespace std;
using namespace sba;

#include <sys/time.h>

// elapsed time in microseconds
static long long utime()
{
  timeval tv;
  gettimeofday(&tv,NULL);
  long long ts = tv.tv_sec;
  ts *= 1000000;
  ts += tv.tv_usec;
  return ts;
}


//
// first argument is the name of input file.
// files are in Freiburg's VERTEX3/EDGE3 format
// runs SPA
//

int main(int argc, char **argv)
{
  char *fin;

  if (argc < 2)
    {
      cout << "Arguments are:  <input filename> [<number of nodes to use>]" << endl;
      return -1;
    }

  // number of nodes to use
  int nnodes = 2200;

  if (argc > 2)
    nnodes = atoi(argv[2]);

  int doiters = 10;
  if (argc > 3)
    doiters = atoi(argv[3]);

  fin = argv[1];

  // node translation
  std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > ntrans;
  // node rotation
  std::vector< Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > nqrot;
  // constraint indices
  std::vector< Eigen::Vector2i, Eigen::aligned_allocator<Eigen::Vector2i> > cind;
  // constraint local translation 
  std::vector< Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > ctrans;
  // constraint local rotation as quaternion
  std::vector< Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d> > cqrot;
  // constraint covariance
  std::vector< Eigen::Matrix<double,6,6>, Eigen::aligned_allocator<Eigen::Matrix<double,6,6> > > cvar;
  // tracks
  std::vector<struct tinfo> tracks;


  ReadSPAFile(fin,ntrans,nqrot,cind,ctrans,cqrot,cvar,tracks);

  cout << "[ReadSPAFile] Found " << (int)ntrans.size() << " nodes and " 
       << (int)cind.size() << " constraints" << endl;


  // system
  SysSPA spa;

  // add in nodes
  for (int i=0; i<(int)ntrans.size(); i++)
    {
      if (i>=nnodes) break;

      Node nd;

      // rotation
      Quaternion<double> frq;
      frq.coeffs() = nqrot[i];
      frq.normalize();
      if (frq.w() <= 0.0) frq.coeffs() = -frq.coeffs();
      nd.qrot = frq.coeffs();

      // translation
      Vector4d v;
      v.start(3) = ntrans[i];
      
      // add in offset for later nodes, just for testing convergence
      if (i > 1000)
        v.start(3) += Vector3d::Constant(10.0 + 0.01*(double)(i-1000));

      v(3) = 1.0;
      nd.trans = v;
      nd.setTransform();        // set up world2node transform
      nd.setDr(true);

      // add to system
      spa.nodes.push_back(nd);
    }

  // add in constraints
  for (int i=0; i<(int)ctrans.size(); i++)
    {
      ConP2 con;
      con.ndr = cind[i].x();
      con.nd1 = cind[i].y();

      if (con.ndr >= nnodes || con.nd1 >= nnodes)
        continue;

      con.tmean = ctrans[i];
      Quaternion<double> qr;
      qr.coeffs() = cqrot[i];
      qr.normalize();
      con.qpmean = qr.inverse(); // inverse of the rotation measurement
      con.prec = cvar[i];       // ??? should this be inverted ???

      // need a boost for noise-offset system
      con.prec.block<3,3>(3,3) *= 10.0;

      spa.p2cons.push_back(con);
    }

  cout << "[ReadSPAFile] Using " << (int)spa.nodes.size() << " nodes and " 
       << (int)spa.p2cons.size() << " constraints" << endl;

  long long t0, t1;
  t0 = utime();
  spa.nFixed = 1;               // one fixed frame
  int niters = spa.doSPA(doiters,1.0e-4,true);
  t1 = utime();
  printf("[TestSPA] Compute took %0.2f ms/iter\n", 0.001*(double)(t1-t0)/(double)doiters);
  printf("[TestSPA] Accepted iterations: %d\n", niters);

  printf("[TestSPA] Distance cost: %0.2f\n", spa.calcCost(true));

  ofstream ofs("sphere-ground.txt");
  for (int i=0; i<(int)ntrans.size(); i++)
    ofs << ntrans[i].transpose() << endl;
  ofs.close();

  ofstream ofs2("sphere-opt.txt");
  for (int i=0; i<(int)spa.nodes.size(); i++)
    ofs2 << spa.nodes[i].trans.transpose().start(3) << endl;
  ofs2.close();

  spa.writeSparseA("sphere-sparse.txt",true);

  return 0;
}
