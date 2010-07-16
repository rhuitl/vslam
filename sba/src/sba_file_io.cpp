#include "sba/sba_file_io.h"

using namespace sba;
using namespace Eigen;
using namespace frame_common;
using namespace std;

int sba::readBundlerFile(char *filename, SysSBA& sbaout)
{ 
    // Create vectors to hold the data from the bundler file. 
    vector< Vector3d, Eigen::aligned_allocator<Vector3d> > camps;	// cam params <f d1 d2>
    vector< Matrix3d, Eigen::aligned_allocator<Matrix3d> > camRs;	// cam rotation matrix
    vector< Vector3d, Eigen::aligned_allocator<Vector3d> > camts;	// cam translation
    vector< Vector3d, Eigen::aligned_allocator<Vector3d> > ptps;	// point position
    vector< Vector3i, Eigen::aligned_allocator<Vector3i> > ptcs;	// point color
    vector< vector< Vector4d, Eigen::aligned_allocator<Vector4d> > > ptts; // point tracks - each vector is <camera_index kp_idex u v>

    int ret = ParseBundlerFile(filename, camps, camRs, camts, ptps, ptcs, ptts);
    if (ret < 0)
        return -1;
        
    int ncams = camps.size();
    int npts  = ptps.size();
    int nprjs = 0;
    for (int i=0; i<npts; i++)
        nprjs += (int)ptts[i].size();
    /* cout << "Points: " << npts << "  Tracks: " << ptts.size() 
         << "  Projections: " << nprjs << endl; */
         
    // cout << "Setting up nodes..." << flush;
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
        
        frq.normalize();

        // translation
        Vector3d &camt = camts[i];
        Vector4d frt;
        frt.start<3>() = camt;//-camRs[i].transpose() * camt; // camera frame translation, from Bundler docs
        frt[3] = 1.0;

        Node nd;
        
        sbaout.addNode(frt, frq, cpars);
    }
    // cout << "done" << endl;

    // set up points
    // cout << "Setting up points..." << flush;
    for (int i=0; i<npts; i++)
    {
        // point
        Vector3d &ptp = ptps[i];
        Point pt;
        pt.start<3>() = ptp;
        pt[3] = 1.0;
        sbaout.addPoint(pt);
    }
    // cout << "done" << endl;


    sbaout.useLocalAngles = true;    // use local angles
    sbaout.nFixed = 1;

    // set up projections
    int ntot = 0;
    // cout << "Setting up projections..." << flush;
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
            sbaout.addMonoProj(cami,i,pt); // Monocular projections
            ntot++;
        }
    }
    // cout << "done" << endl;
    
    return 0;
}

int sba::writeBundlerFile(char *filename, SysSBA& sbain)
{
    ofstream outfile(filename, ios_base::trunc);
    if (outfile == NULL)
    {
        cout << "Can't open file " << filename << endl;
        return -1;
    }
    
    outfile.precision(10);
    outfile.setf(ios_base::scientific);
    
    unsigned int i = 0;
    
    outfile << "# Bundle file v0.3" << endl;
    // First line is number of cameras and number of points.
    outfile << sbain.nodes.size() << ' ' << sbain.tracks.size() << endl;
    
    // Set up transform matrix for camera parameters
    Matrix3d m180x;		// rotate 180 deg around X axis, to convert Bundler frames to SBA frames
    m180x << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    
    
    // Then goes information about each camera, in <f> <k1> <k2>\n<R>\n<t> format.
    for (i = 0; i < sbain.nodes.size(); i++)
    {
        // Assuming fx = fy and using fx. Don't use k1 or k2.
        outfile << sbain.nodes[i].Kcam(0, 0) << ' ' << 0.0 << ' ' << 0.0 << endl;
        
        Quaternion<double> quat(sbain.nodes[i].qrot);
        /* cout << "\nQuat: [ " << sbain.nodes[i].qrot << " ]\n"; */ 
        quat.normalize();
        Matrix3d rotmat = m180x * quat.toRotationMatrix().transpose();
                       
        outfile << rotmat(0, 0) << ' ' << rotmat(0, 1) << ' ' << rotmat(0, 2) << endl;
        outfile << rotmat(1, 0) << ' ' << rotmat(1, 1) << ' ' << rotmat(1, 2) << endl;
        outfile << rotmat(2, 0) << ' ' << rotmat(2, 1) << ' ' << rotmat(2, 2) << endl;
        
        Vector3d trans = sbain.nodes[i].trans.start<3>();
        
        outfile << trans(0) << ' ' << trans(1) << ' ' << trans(2) << endl; 
    }
    
    outfile.setf(ios_base::fixed);
    
    // Then goes information about each point. <pos>\n<color>\n<viewlist>
    for (i = 0; i < sbain.tracks.size(); i++)
    {
        // World <x y z>
        outfile << sbain.tracks[i].point(0) << ' ' << sbain.tracks[i].point(1) 
                << ' ' << sbain.tracks[i].point(2) << endl;
        // Color <r g b> (Just say white instead)
        outfile << "255 255 255" << endl;
        // View list: <list_length><camera_index key u v>\n<camera_index key u v>
        // Key is the keypoint # in SIFT, but we just use point number instead.
        // We output these as monocular points because the file format does not
        // support stereo points.
        
        ProjMap &prjs = sbain.tracks[i].projections;
        
        // List length
        outfile << prjs.size() << ' ';
        
        // Output all the tracks as monocular tracks.
        for(ProjMap::iterator itr = prjs.begin(); itr != prjs.end(); itr++)
        {
            Proj &prj = itr->second;
            // y is reversed (-y)
            Node &node = sbain.nodes[prj.ndi];
            
            double cx = node.Kcam(0, 2);
            double cy = node.Kcam(1, 2);
            
            outfile << prj.ndi << ' ' << i << ' ' << prj.kp(0)-cx << ' ' 
                    << -(prj.kp(1)-cy) << ' ';
        }
        
        outfile << endl;
    }

    return 0;
} 

int  sba::ParseBundlerFile(char *fin,	// input file
		vector< Vector3d, Eigen::aligned_allocator<Vector3d> > &camp, // cam params <f d1 d2>
		vector< Matrix3d, Eigen::aligned_allocator<Matrix3d> > &camR, // cam rotation matrix
		vector< Vector3d, Eigen::aligned_allocator<Vector3d> > &camt, // cam translation
		vector< Vector3d, Eigen::aligned_allocator<Vector3d> > &ptp, // point position
		vector< Vector3i, Eigen::aligned_allocator<Vector3i> > &ptc, // point color
		vector< vector< Vector4d, Eigen::aligned_allocator<Vector4d> > > &ptt // point tracks - each vector is <camera_index u v>
		)
{
    ifstream ifs(fin);
    if (ifs == NULL)
      {
        cout << "Can't open file " << fin << endl;
        return -1;
      }
    ifs.precision(10);


    // read header
    string line;
    if (!getline(ifs,line) || line != "# Bundle file v0.3")
      {
        cout << "Bad header" << endl;
        return -1;
      }
    cout << "Found Bundler 3.0 file" << endl;

    // read number of cameras and points
    int ncams, npts;
    if (!(ifs >> ncams >> npts))
      {
        cout << "Bad header" << endl;  
        return -1;
      }
    cout << "Number of cameras: " << ncams << "  Number of points: " << npts << endl;
    
    cout << "Reading in camera data..." << flush;
    for (int i=0; i<ncams; i++)
      {
        double v1,v2,v3,v4,v5,v6,v7,v8,v9;
        if (!(ifs >> v1 >> v2 >> v3))
	  {
	    cout << "Bad camera params at number " << i << endl;
	    return -1;
	  }
        camp.push_back(Vector3d(v1,v2,v3));

        if (!(ifs >> v1 >> v2 >> v3 >> v4 >> v5 >> v6 >> v7 >> v8 >> v9))
	  {
	    cout << "Bad camera rotation matrix at number " << i << endl;
	    return -1;
	  }
        Matrix3d m;
        m << v1,v2,v3,v4,v5,v6,v7,v8,v9;
        camR.push_back(m);

        if (!(ifs >> v1 >> v2 >> v3))
	  {
	    cout << "Bad camera translation at number " << i << endl;
	    return -1;
	  }
        camt.push_back(Vector3d(v1,v2,v3));
      }
    cout << "done" << endl;

    ptt.resize(npts);

    cout << "Reading in pts data..." << flush;
    for (int i=0; i<npts; i++)
      {
        double v1,v2,v3;
        int i1,i2,i3;

        if (!(ifs >> v1 >> v2 >> v3))
	  {
	    cout << "Bad point position at number " << i << endl;
	    return -1;
	  }
        ptp.push_back(Vector3d(v1,v2,v3));

        if (!(ifs >> i1 >> i2 >> i3))
	  {
	    cout << "Bad point color at number " << i << endl;
	    return -1;
	  }
        ptc.push_back(Vector3i(i1,i2,i3));


        if (!(ifs >> i1))
	  {
	    cout << "Bad track count at number " << i << endl;
	    return -1;
	  }
        int nprjs = i1;

        vector<Vector4d, Eigen::aligned_allocator<Vector4d> > &prjs = ptt[i];
        for (int j=0; j<nprjs; j++)
	  {
	    if (!(ifs >> i1 >> i2 >> v1 >> v2))
	      {
	        cout << "Bad track parameter at number " << i << endl;
	        return -1;
	      }
	    prjs.push_back(Vector4d(i1,i2,v1,v2));
	  }

      } // end of pts loop
    cout << "done" << endl;

    // print some stats
    double nprjs = 0;
    for (int i=0; i<npts; i++)
      nprjs += ptt[i].size();
    cout << "Number of projections: " << (int)nprjs << endl;
    cout << "Average projections per camera: " << nprjs/(double)ncams << endl;
    cout << "Average track length: " << nprjs/(double)npts << endl;
    return 0;
}


// write out the system in an sba (Lourakis) format
// NOTE: Lourakis FAQ is wrong about coordinate systems
//   Cameras are represented by the w2n transform, converted to
//   a quaternion and translation vector
//
void sba::writeLourakisFile(char *fname, SysSBA& sba)
{
    char name[1024];
    sprintf(name,"%s-cams.txt",fname);
    FILE *fn = fopen(name,"w");
    if (fn == NULL)
      {
        cout << "[WriteFile] Can't open file " << name << endl;
        return;
      }
    
    // write out initial camera poses
    int ncams = sba.nodes.size();
    for (int i=0; i<ncams; i++)
      {
        Node &nd = sba.nodes[i];

        // Why not just use the Quaternion???
        Quaternion<double> frq(nd.w2n.block<3,3>(0,0)); // rotation matrix of transform
        fprintf(fn,"%f %f %f %f ", frq.w(), frq.x(), frq.y(), frq.z());
        Vector3d tr = nd.w2n.col(3);
        fprintf(fn,"%f %f %f\n", tr[0], tr[1], tr[2]);
      }
    fclose(fn);

    sprintf(name,"%s-pts.txt",fname);
    fn = fopen(name,"w");
    if (fn == NULL)
      {
        cout << "[WriteFile] Can't open file " << name << endl;
        return;
      }
    
    fprintf(fn,"# X Y Z  nframes  frame0 x0 y0  frame1 x1 y1 ...\n");

    // write out points
    
    for(size_t i=0; i<sba.tracks.size(); i++)
      {
        ProjMap &prjs = sba.tracks[i].projections;
        // Write out point
        Point &pt = sba.tracks[i].point;
        
        fprintf(fn,"%f %f %f  ", pt.x(), pt.y(), pt.z());
        fprintf(fn,"%d  ",(int)prjs.size());
        
        // Write out projections
        for(ProjMap::iterator itr = prjs.begin(); itr != prjs.end(); itr++)
          {
            Proj &prj = itr->second;      
            if (!prj.isValid) continue;
            int cami = prj.ndi;
            // NOTE: Lourakis projected y is reversed (???)
            fprintf(fn," %d %f %f ",cami,prj.kp[0],prj.kp[1]);
          }
        fprintf(fn,"\n");
      }

    fclose(fn);

    // write camera calibartion
    sprintf(name,"%s-calib.txt",fname);
    fn = fopen(name,"w");
    if (fn == NULL)
      {
        cout << "[WriteFile] Can't open file " << name << endl;
        return;
      }
    
    Matrix3d &K = sba.nodes[0].Kcam;
    fprintf(fn,"%f %f %f\n", K(0,0), K(0,1), K(0,2));
    fprintf(fn,"%f %f %f\n", K(1,0), K(1,1), K(1,2));
    fprintf(fn,"%f %f %f\n", K(2,0), K(2,1), K(2,2));

    fclose(fn);
}


// write out the precision matrix
void sba::writeA(char *fname, SysSBA& sba)
{
    ofstream ofs(fname);
    if (ofs == NULL)
      {
        cout << "Can't open file " << fname << endl;
        return;
      }

    // cameras
    Eigen::IOFormat pfmt(16);
    ofs << sba.A.format(pfmt) << endl;
    ofs.close();
}


// write out the precision matrix for CSparse
void sba::writeSparseA(char *fname, SysSBA& sba)
{
    char name[1024];
    sprintf(name,"%s-A.tri",fname);

    {
      ofstream ofs(name);
      if (ofs == NULL)
        {
          cout << "Can't open file " << fname << endl;
          return;
        }

      // cameras
      Eigen::IOFormat pfmt(16);

      int nrows = sba.A.rows();
      int ncols = sba.A.cols();
    
      cout << "[WriteSparseA] Size: " << nrows << "x" << ncols << endl;

      // find # nonzeros
      int nnz = 0;
      for (int i=0; i<nrows; i++)
        for (int j=i; j<ncols; j++)
          {
            double a = sba.A(i,j);
            if (a != 0.0) nnz++;
          }

      ofs << nrows << " " << ncols << " " << nnz << " 1" << endl;

      for (int i=0; i<nrows; i++)
        for (int j=i; j<ncols; j++)
          {
            double a = sba.A(i,j);
            if (a != 0.0)
              ofs << i << " " << j << " " << setprecision(16) << a << endl;
          }

      ofs.close();
    }

    sprintf(name,"%s-B.txt",fname);

    {
      ofstream ofs(name);
      if (ofs == NULL)
        {
          cout << "Can't open file " << fname << endl;
          return;
        }

      // cameras
      Eigen::IOFormat pfmt(16);

      int nrows = sba.B.rows();
    
      ofs << nrows << " " << 1 << endl;

      for (int i=0; i<nrows; i++)
        {
          double a = sba.B(i);
          ofs << setprecision(16) << a << endl;
        }
      ofs.close();
    }
}

