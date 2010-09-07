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

// run an (undistorted) Bundler file through SBA
// files are in ~/devel/sba-data/venice

#include "sba/sba_file_io.h"
#include "sba/sba.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sys/time.h>

using namespace std;
using namespace Eigen;
using namespace sba;
using namespace frame_common;

//
// test cholmod timing
//

#include <time.h>
#define CPUTIME ((double) (clock ( )) / CLOCKS_PER_SEC)

#ifdef SBA_CHOLMOD
void cholmod_timing(char *fA, char *fB)
{
    FILE *ff = NULL ;
    FILE *fb = NULL ;
    ff = fopen(fA,"r");
    fb = fopen(fB,"r");

    cholmod_sparse *A ;
    cholmod_dense *x, *b, *r ;
    cholmod_factor *L ;
    double one [2] = {1,0}, m1 [2] = {-1,0} ; // basic scalars 
    cholmod_common c ;
    cholmod_start (&c) ;			    /* start CHOLMOD */
    printf("Reading %s\n",fA);
    A = cholmod_read_sparse (ff, &c) ;              /* read in a matrix */
    cholmod_print_sparse (A, (char *)"A", &c) ; /* print the matrix */
    if (A == NULL || A->stype == 0)		    /* A must be symmetric */
    {
	cholmod_free_sparse (&A, &c) ;
	cholmod_finish (&c) ;
	return ;
    }
    printf("Reading %s\n",fB);
    if (fb)
      b = cholmod_read_dense(fb, &c);
    else
      b = cholmod_ones (A->nrow, 1, A->xtype, &c) ; /* b = ones(n,1) */
    double t0 = CPUTIME;
    L = cholmod_analyze (A, &c) ;		    /* analyze */
    cholmod_factorize (A, L, &c) ;		    /* factorize */
    x = cholmod_solve (CHOLMOD_A, L, b, &c) ;	    /* solve Ax=b */
    double t1 = CPUTIME;
    printf("Time: %12.4f \n", t1-t0);
    r = cholmod_copy_dense (b, &c) ;		    /* r = b */
    cholmod_sdmult (A, 0, m1, one, x, r, &c) ;	    /* r = r-Ax */
    printf ("norm(b-Ax) %8.1e\n",
	    cholmod_norm_dense (r, 0, &c)) ;	    /* print norm(r) */
    cholmod_free_factor (&L, &c) ;		    /* free matrices */
    cholmod_free_sparse (&A, &c) ;
    cholmod_free_dense (&r, &c) ;
    cholmod_free_dense (&x, &c) ;
    cholmod_free_dense (&b, &c) ;
    cholmod_finish (&c) ;			    /* finish CHOLMOD */
}
#endif

//
// first argument is the name of input file, Bundler format
//    expects good focal length and distortion correction
// runs sba
//

int main(int argc, char **argv)
{
  //  cholmod_timing((char *)"test/A819.venice-A.tri",(char *)"test/A819.venice-B.txt");

  char *fin;

  if (argc < 2)
    {
      cout << "Arguments are:  <input filename> [<min conn pts>]" << endl;
      return -1;
    }

  int minpts = 0;
  if (argc > 2)
    minpts = atoi(argv[2]);

  fin = argv[1];

  vector< Vector3d, Eigen::aligned_allocator<Vector3d> > camps;	// cam params <f d1 d2>
  vector< Matrix3d, Eigen::aligned_allocator<Matrix3d> > camRs;	// cam rotation matrix
  vector< Vector3d, Eigen::aligned_allocator<Vector3d> > camts;	// cam translation
  vector< Vector3d, Eigen::aligned_allocator<Vector3d> > ptps;	// point position
  vector< Vector3i, Eigen::aligned_allocator<Vector3i> > ptcs;	// point color
  vector< vector< Vector4d, Eigen::aligned_allocator<Vector4d> > > ptts; // point tracks - each vector is <camera_index kp_idex u v>

  int ret = sba::ParseBundlerFile(fin, camps, camRs, camts, ptps, ptcs, ptts);
  if (ret < 0)
    return -1;
  int ncams = camps.size();
  int npts  = ptps.size();
  int nprjs = 0;
  for (int i=0; i<npts; i++)
    nprjs += (int)ptts[i].size();
  cout << "Points: " << npts << "  Tracks: " << ptts.size() 
       << "  Projections: " << nprjs << endl;

  // construct an SBA system
  SysSBA sys;
  Node::initDr();

  // set up nodes/frames
  cout << "Setting up nodes..." << flush;
  for (int i=0; i<ncams; i++)
    {
      // camera params
      Vector3d &camp = camps[i];
      CamParams cpars = {camp[0],camp[0],0,0,0}; // set focal length, no offset

      //
      // NOTE: Bundler camera coords are rotated 180 deg around the X axis of
      //  the camera, so Z points opposite the camera viewing ray (OpenGL).
      // Note quite sure, but I think this gives the camera pose as
      //  [-R' -R'*t]

      // rotation matrix
      Matrix3d m180x;		// rotate 180 deg around X axis, to convert Bundler frames to SBA frames
      m180x << 1, 0, 0, 0, -1, 0, 0, 0, -1;
      Matrix3d camR = m180x * camRs[i]; // rotation matrix
      Quaternion<double> frq(camR.transpose());	// camera frame rotation, from Bundler docs
      if (frq.w() < 0.0)	// w negative, change to positive
	{
	  frq.x() = -frq.x();
	  frq.y() = -frq.y();
	  frq.z() = -frq.z();
	  frq.w() = -frq.w();
	}

      // translation
      Vector3d &camt = camts[i];
      Vector4d frt;
      frt.head<3>() = -camRs[i].transpose() * camt; // camera frame translation, from Bundler docs
      frt[3] = 1.0;

      Node nd;
      nd.qrot = frq.coeffs();	
      nd.normRot();
      //      cout << "Quaternion: " << nd.qrot.transpose() << endl;
      nd.trans = frt;
      //      cout << "Translation: " << nd.trans.transpose() << endl << endl;
      nd.setTransform();	// set up world2node transform
      nd.setKcam(cpars);	// set up node2image projection
      nd.setDr(true);		// set rotational derivatives
      sys.nodes.push_back(nd);
    }
  cout << "done" << endl;

  // set up points
  cout << "Setting up points..." << flush;
  for (int i=0; i<npts; i++)
    {
      // point
      Vector3d &ptp = ptps[i];
      Point pt;
      pt.head<3>() = ptp;
      pt[3] = 1.0;
      sys.addPoint(pt);
    }
  cout << "done" << endl;


  sys.useLocalAngles = true;    // use local angles
  sys.nFixed = 1;

  // set up projections
  int ntot = 0;
  cout << "Setting up projections..." << flush;
  for (int i=0; i<npts; i++)
    {
      // track
      vector<Vector4d, Eigen::aligned_allocator<Vector4d> > &ptt = ptts[i];
      int nprjs = ptt.size();
      for (int j=0; j<nprjs; j++)
	{
	  // projection
	  Vector4d &prj = ptt[j];
	  int cami = (int)prj[0];
	  Vector2d pt = prj.segment<2>(2);
	  pt[1] = -pt[1];	// NOTE: Bundler image Y is reversed
	  if (cami >= ncams)
	    cout << "*** Cam index exceeds bounds: " << cami << endl;
	  sys.addMonoProj(cami,i,pt); // camera indices aren't ordered
	  ntot++;

#if 0
	  if (ntot==1000000)
	    {
	      Node &nd = sys.nodes[cami];
	      Point &npt = sys.points[i];
	      cout << pt.transpose() << endl;
	      Vector3d pti = nd.w2i * npt;
	      pti = pti / pti[2];
	      cout << pti.transpose() << endl;
	      cout << nd.trans.transpose() << endl;
	      cout << nd.qrot.transpose() << endl;
	    }
#endif

	  //	  if ((ntot % 100000) == 0)
	  //	    cout << ntot << endl;
	}
    }
  cout << "done" << endl;


  if (minpts > 0)
    {
      int nrem = sys.reduceLongTracks(minpts); // tracks greater than minpts size are removed
    //      sys.remExcessTracks(minpts);
      cout << "Split " << nrem << " / " << sys.tracks.size() << " tracks" << endl; 
    }

  double cost = sys.calcCost();
  cout << "Initial squared cost: " << cost << ",  which is " << sqrt(cost/nprjs) << " rms pixels per projection"  << endl;

  sys.nFixed = 1;
  sys.printStats();
  sys.csp.useCholmod = true;


#if 1
  sba::writeLourakisFile((char *)"bra-340", sys);
  cout << endl << "Wrote SBA system in Lourakis format" << endl << endl;
#endif

#if 0
  cout << endl;

  cout << "Bad projs (> 100 pix): " << sys.countBad(100.0) 
       << "  Cost without: " << sqrt(sys.calcCost(100.0)/nprjs) << endl;
  cout << "Bad projs (> 20 pix): " << sys.countBad(20.0)
       << "  Cost without: " << sqrt(sys.calcCost(20.0)/nprjs) << endl;
  cout << "Bad projs (> 10 pix): " << sys.countBad(10.0)
       << "  Cost without: " << sqrt(sys.calcCost(10.0)/nprjs) << endl;
  int n = sys.removeBad(20.0);
  cout << "Removed " << n << " projs with >10px error" << endl;
  sys.printStats();
#endif

#if 0
  sys.doSBA(2);
  sys.setupSys(0.0);
  sys.writeSparseA((char *)"A819.venice");
#endif

  //  sys.setConnMat(minpts);
  //  sys.setConnMatReduced(minpts);             // finds spanning tree


#if 0
  // save sparsity pattern 
  cout << "[SBAsys] Saving sparsity pattern in <sparsity.txt>" << endl;
  sys.doSBA(1,1e-3,0);
  FILE *fd = fopen("sparsity.txt","w");
  int m = sys.B.size();
  for (int i=0; i<m; i+=6)
    {
      for (int j=0; j<m; j+=6)
        if (sys.A(i,j) != 0.0)
          fprintf(fd,"1 ");
        else
          fprintf(fd,"0 ");  
      fprintf(fd,"\n");
    }
  fclose(fd);
#endif


  //  sys.doSBA(10,1e-4,SBA_SPARSE_CHOLESKY);
  sys.doSBA(10,1e-4,SBA_BLOCK_JACOBIAN_PCG,1e-8,200);

  cout << endl << "Switch to full system" << endl;
  sys.connMat.resize(0);


  // reset projections here
  // just use old points
  sys.tracks.resize(npts);

  // set up projections
  sys.tracks.resize(0);
  cout << "Setting up projections..." << flush;
  for (int i=0; i<npts; i++)
    {
      // track
      vector<Vector4d, Eigen::aligned_allocator<Vector4d> > &ptt = ptts[i];
      int nprjs = ptt.size();
      for (int j=0; j<nprjs; j++)
	{
	  // projection
	  Vector4d &prj = ptt[j];
	  int cami = (int)prj[0];
	  Vector2d pt = prj.segment<2>(2);
	  pt[1] = -pt[1];	// NOTE: Bundler image Y is reversed
	  if (cami >= ncams)
	    cout << "*** Cam index exceeds bounds: " << cami << endl;
	  sys.addMonoProj(cami,i,pt); // camera indices aren't ordered
	  ntot++;
	}
    }
  cout << "done" << endl;


  sys.doSBA(20,1e-3,1);

  cout << "Bad projs (> 10 pix): " << sys.countBad(10.0) 
       << "  Cost without: " << sqrt(sys.calcCost(10.0)/nprjs) << endl;
  cout << "Bad projs (>  5 pix): " << sys.countBad( 5.0)
       << "  Cost without: " << sqrt(sys.calcCost( 5.0)/nprjs) << endl;
  cout << "Bad projs (>  2 pix): " << sys.countBad( 2.0)
       << "  Cost without: " << sqrt(sys.calcCost( 2.0)/nprjs) << endl << endl;

  sys.removeBad(10.0);
  cout << "Removed projs with >10px error" << endl;

  sys.doSBA(10,1e-3,true);
  cout << "Bad projs (> 10 pix): " << sys.countBad(10.0) << endl;
  cout << "Bad projs (>  5 pix): " << sys.countBad( 5.0) << endl;
  cout << "Bad projs (>  2 pix): " << sys.countBad( 2.0) << endl << endl;

  sys.doSBA(10);
  cout << "Bad projs (> 10 pix): " << sys.countBad(10.0) << endl;
  cout << "Bad projs (>  5 pix): " << sys.countBad( 5.0) << endl;
  cout << "Bad projs (>  2 pix): " << sys.countBad( 2.0) << endl << endl;

  sys.doSBA(10);
  cout << "Bad projs (> 10 pix): " << sys.countBad(10.0) << endl;
  cout << "Bad projs (>  5 pix): " << sys.countBad( 5.0) << endl;
  cout << "Bad projs (>  2 pix): " << sys.countBad( 2.0) << endl << endl;

  return 0;
}
