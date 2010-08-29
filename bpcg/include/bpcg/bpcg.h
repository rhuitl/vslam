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

#ifndef _BPCG_H_
#define _BPCG_H_

#ifndef EIGEN_USE_NEW_STDVECTOR
#define EIGEN_USE_NEW_STDVECTOR
#endif // EIGEN_USE_NEW_STDVECTOR

#include <vector>
#include <map>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include <Eigen/StdVector>

using namespace Eigen;
using namespace std;

namespace sba
{

//
// matrix multiply of compressed column storage + diagonal blocks by a vector
//

void
mMV(vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > &diag,
    vector< map<int,Matrix<double,6,6>, less<int>, aligned_allocator<Matrix<double,6,6> > > > &cols,
    const VectorXd &vin,
    VectorXd &vout);

//
// jacobi-preconditioned block conjugate gradient
// returns number of iterations

int
bpcg_jacobi(int iters, double tol,
	    vector< Matrix<double,6,6>, aligned_allocator<Matrix<double,6,6> > > &diag,
	    vector< map<int,Matrix<double,6,6>, less<int>, aligned_allocator<Matrix<double,6,6> > > > &cols,
	    VectorXd &x,
	    VectorXd &b,
	    bool abstol = false,
	    bool verbose = false
         );

int
bpcg_jacobi_dense(int iters, double tol,
		  MatrixXd &M,
		  VectorXd &x,
		  VectorXd &b);

}  // end namespace sba

#endif // BPCG_H
