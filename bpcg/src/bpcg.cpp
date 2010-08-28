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
// block preconditioned conjugate gradient
// 6x6 blocks at present, should be templatized
//

#include "bpcg/bpcg.h"


namespace sba
{

  //
  // matrix multiply of compressed column storage + diagonal blocks by a vector
  //

void
mMV(vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > &diag,
    vector< map<int,Matrix<double,6,6>, less<int>, aligned_allocator<Matrix<double,6,6> > > > &cols,
    VectorXd &vin,
    VectorXd &vout)
  {
    // loop over diag entries
    for (int i=0; i<(int)diag.size(); i++)
      vout.segment<6>(i*6) = diag[i]*vin.segment<6>(i*6);

    // loop over off-diag entries
    if (cols.size() > 0)
    for (int i=0; i<(int)cols.size(); i++)
      {
	map<int,Matrix<double,6,6>, less<int>, 
	  aligned_allocator<Matrix<double,6,6> > > &col = cols[i];
	if (col.size() > 0)
	  {
	    map<int,Matrix<double,6,6>, less<int>, 
	      aligned_allocator<Matrix<double,6,6> > >::iterator it;
	    for (it = col.begin(); it != col.end(); it++)
	      {
		int ri = (*it).first; // get row index
		Matrix<double,6,6> &M = (*it).second; // matrix
		vout.segment<6>(i*6)  += M.transpose()*vin.segment<6>(ri*6);
		vout.segment<6>(ri*6) += M*vin.segment<6>(i*6);
	      }
	  }
      }

  }

void
mD(vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > &diag,
    VectorXd &vin,
    VectorXd &vout)
{
    // loop over diag entries
    for (int i=0; i<(int)diag.size(); i++)
      vout.segment<6>(i*6) = diag[i]*vin.segment<6>(i*6);
}


//
// jacobi-preconditioned block conjugate gradient
// returns number of iterations

int
bpcg_jacobi(int iters, double tol,
	    vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > &diag,
	    vector< map<int,Matrix<double,6,6>, less<int>, aligned_allocator<Matrix<double,6,6> > > > &cols,
	    VectorXd &x,
	    VectorXd &b)
{
  // set up local vars
  VectorXd r,rr,d,q,s;
  int n = diag.size();
  int n6 = n*6;
  r.setZero(n6);
  rr.setZero(n6);
  d.setZero(n6);
  q.setZero(n6);
  s.setZero(n6);

  // set up Jacobi preconditioner
  vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > J;
  J.resize(n);
  for (int i=0; i<n; i++)
    {
      J[i] = diag[i].inverse();
      //      J[i].setIdentity();
    }

  int i;
  r = b;
  mD(J,r,d);
  double dn = r.dot(d);
  double d0 = dn;

  for (i=0; i<iters; i++)
    {
      cout << "residual[" << i << "]: " << dn << endl;
      if (dn < tol*tol*d0) break; // done
      mMV(diag,cols,d,q);
      double a = dn / d.dot(q);
      x += a*d;
      // TODO: reset residual here every 50 iterations
      r -= a*q;
      mD(J,r,s);
      double dold = dn;
      dn = r.dot(s);
      double ba = dn / dold;
      d = s + ba*d;
    }

  return i;
}


//
// dense algorithm
//

int
bpcg_jacobi_dense(int iters, double tol,
		  MatrixXd &M,
		  VectorXd &x,
		  VectorXd &b)
{
  // set up local vars
  VectorXd r,ad,d,q,s;
  int n6 = M.cols();
  int n = n6/6;
  r.setZero(n6);
  ad.setZero(n6);
  d.setZero(n6);
  q.setZero(n6);
  s.setZero(n6);

  // set up Jacobi preconditioner
  vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > J;
  J.resize(n);
  for (int i=0; i<n; i++)
    {
      J[i] = M.block(i*6,i*6,6,6).inverse();
      //      J[i].setIdentity();
    }

  int i;
  r = b;
  mD(J,r,d);
  double dn = r.dot(d);
  double d0 = dn;

  for (i=0; i<iters; i++)
    {
      cout << "residual[" << i << "]: " << dn << endl;
      if (dn < tol*tol*d0) break; // done
      
      q = M*d;
      double a = dn / d.dot(q);
      x += a*d;
      // TODO: reset residual here every 50 iterations
      r -= a*q;
      mD(J,r,s);
      double dold = dn;
      dn = r.dot(s);
      double ba = dn / dold;
      d = s + ba*d;
    }

  return i;
}

} // end namespace sba

