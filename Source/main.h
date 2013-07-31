/* DASM -- Dynamic Active Shape Models
 * Author: David Macurak
 * Version: v1.0
 * 
 * Copyright 2013 David Macurak

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.

 *	Current Dependencies:
 *	Boost: 1.53
 *	OpenCV: 2.45
 */

#ifndef ASM_SHAPE_H
#define ASM_SHAPE_H

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <omp.h>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/date_time/local_time/local_time.hpp>
#include <boost/exception/diagnostic_information.hpp>
#ifdef WITH_PITTPATT
#include "PittPatt.hpp"
#endif
#ifdef _MSC_VER
#include <Windows.h>
#endif

////////////////////////////////////////////////////////////////////////// DEFINES
#define MIN_CONVERGENCE1D .51f // Minimum convergence of suggested movements per level of the pyramid search
#define MIN_CONVERGENCE2D .51f
#define MAX_ITERATIONS 6 // Maximum iterations for each level of search
#define FAILURE 0		// Error Codes
#define SUCCESS 1		//
#define NUM_OFFSETS 3	// Offset distance from the original point to consider convergence
#define VERSION 1.0		// Program version

////////////////////////////////////////////////////////////////////////// NAME SPACES
using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace boost::program_options;
namespace pt = boost::posix_time;

////////////////////////////////////////////////////////////////////////// GLOBAL FUNCTIONS
int trainASM(path inputDir, path outputDir, path modelDir, path partsPath, path detectorName);
int searchASM(path inputDir, path modelPath, path detectorName, path outputDir);

////////////////////////////////////////////////////////////////////////// CLASS CONSTANTS (Global Singleton)
class Constants
{
	static Constants* m_pInstance;
private:
	int profLength2d;	// for 2d profiles (Square profile, so = width or height not width x height)
	int profLength1d;	// for 1d profiles
	int border;		// border padding to avoid out of bounds image accesses
	int num1DLevels;	// levels of the pyramid search for 1D profiles (resolutions)
	int num2DLevels;	// levels of pyramid search for 2D 
	int num_eigs;		// Maximum eigen vectors to use when conforming to the shape model
	int searchParam;	// search parameter 1d or 2d
	int stacked;		// Perform stacked model search
	bool verbose;		// Enter verbose mode
	int detector;		// enumerated value for the detector type:  0 = Viola-Jones, 1 = PittPatt, (add more as needed)
	float eigPercent;	// 
	Size stdSize;		// standard image size for training

	
	Constants()		// Default Values
	{
		profLength2d = 9; 
		profLength1d = 9;	
		border = profLength2d; 
		num1DLevels = 3;	
		num2DLevels = 3;
		searchParam = 1;
		stacked = 1;
		eigPercent = 0.75;
		verbose = false;
		stdSize.width = 340;
		stdSize.height = 400;
	}

public:

	static Constants* instance()
	{

		if (!m_pInstance)
		{
			m_pInstance = new Constants();
		}
		return m_pInstance;
	}

	void setFromModel(int n1d, int n2d, int p1d, int p2d) // Constructor from Model info
	{
		profLength2d = p2d; 
		profLength1d = p1d;	
		border = profLength2d; 
		num1DLevels = n1d;	
		num2DLevels = n2d;
	}

	void setP1d(int p){profLength1d = p;}
	void setP2d(int p){profLength2d = p;}
	void setb(int b){border = b;}
	void setN1d(int n){num1DLevels = n;}
	void setN2d(int n){num2DLevels = n;}
	void setNE(int n){num_eigs = n;}
	void setSP(int s){searchParam = s;}
	void setSt(int s){stacked = s;}
	void setVb(){verbose = true;}
	void setVJ(){detector = VJ;}
	void setPP(){detector = PP;}
	void setEP(float p){eigPercent = p;}
	void setSize(int w, int h){stdSize.width = w; stdSize.height = h;}

	int getDetector_t(){return detector;}
	int getProfLength2d(){return profLength2d;}
	int getProfLength1d(){return profLength1d;}
	int getBorder(){return border;}
	int getNum1DLevels(){return num1DLevels;}
	int getNum2DLevels(){return num2DLevels;}
	int getNumEigs(){return num_eigs;}
	int getSearchParam(){return searchParam;}
	int getStacked(){return stacked;}
	Size getSize(){return stdSize;}
	float getEigPercent(){return eigPercent;}
	bool isVerbose(){return verbose;}

	enum detectorEnum_t
	{ 
		VJ,
		PP
		// Add here if you have more detectors
	};
};

////////////////////////////////////////////////////////////////////////// CLASS PARTS
class Parts
{
private:
	bool isClosedBoundary;
public:
	static int loadPartsFile(vector<Parts> &allParts, path& partsFileName);
	bool getPartBoundary(){return isClosedBoundary;}
	void setPartsBoundary(int b){isClosedBoundary = b==1 ? true : false;}
	vector<int> partPts;
};
////////////////////////////////////////////////////////////////////////// CLASS PROFILE1D
class Profile1D
{
private:
	float slope;
	float orthogonalSlope;
public:
	vector<float> profileIntensities1D;
	vector<float> profileGradients1D;
	void setSlope(float m){slope = m;}
	void setOrthSlope(float m){orthogonalSlope = m;}
	float getSlope(){return slope;}
	float getOrthSlope(){return orthogonalSlope;}
	static float findSlopeOfProfile(const vector<Point2f> &p, vector<Parts> mp, const int c);
	static vector<Point2f> findProfileOffsets(const float m, const int x, const int y, Mat img, const int pl);
	static vector<float> findProfile1D(const int x, const int y, const float m, const int pl, Mat imgBordBuf);
	static void calcProfGradient1D(vector<float> &pg, const int pl);

};
////////////////////////////////////////////////////////////////////////// CLASS PROFILE2D
class Profile2D
{
private:
	float slope;
	Mat profile;
public:
	Profile2D(int width, int height) {profile.create(width,height,CV_32F); slope=0.0;}
	Profile2D(){};
	void init(int width, int height) {profile.create(width,height,CV_32F); slope=0.0;}
	void setSlope(float m){slope = m;}
	float getSlope(){return slope;}
	void setProfile(Mat p){profile = p.clone();}
	Mat getProfile(){return profile;}
	void convolve2dProfile();
	void normalize2dProfile();
	void equalize2dProfile();
	void weightMask2dProfile();
};
////////////////////////////////////////////////////////////////////////// CLASS SHAPE
class Shape
{
private:
	vector<Point2f> orig_Point; // backup copy of points
	vector<Point2f> m_Point;
	float meanX, meanY;
	vector< vector<Profile1D> > allLev1dProfiles;
	vector< vector<Profile2D> > allLev2dProfiles;
	Mat imageData;
	Mat origImgData;	// backup copy of image
	int nPoints;
	string filename;
	CvRect faceROI;		

public:
	void	setFilename(string s){filename = s;}
	string	getFilename(){return filename;}
	vector<Profile1D> get1DProfiles(int i){return allLev1dProfiles[i];}
	vector<Point2f> getAllPoints(){return m_Point;}
	Point2f getPoint(int i){return m_Point[i];}
	CvRect	getFaceROI(){return faceROI;}
	int		getNPoints(){return nPoints;}
	Mat&	getImgbyRef(){return imageData;}
	Mat		getOrigImgData(){return origImgData;}
	Mat		getImgData(){return imageData;}

	void	setImgData(Mat m){m.copyTo(imageData);}
	void	setOrigImgData(Mat m){m.copyTo(origImgData);}
	void	setFaceROI(CvRect f){faceROI = f;}
	void	setNPoints(int n){nPoints = n;}
	void	setPoint(Point2f p){m_Point.push_back(p);}
	void	setup1dDataStructures(int n){allLev1dProfiles.resize(n);}
	void	setup2dDataStructures(int n){allLev2dProfiles.resize(n);}
	void	setAllPoints(vector<Point2f> in_Points){m_Point = in_Points;}
	void	setPointAt(Point2f p, int i){m_Point[i] = p;}
	void	resetPoints(){m_Point = orig_Point;}

	const Point2f operator[] (int i)const{  return m_Point[i];	 }
	Point2f& operator[] (int i){  return m_Point[i];	}

	// Profile related funcs
	static void trainAllProfiles(vector<Shape> &Allshapes,vector<Parts> &parts, Size dsize);
	static void trainAll1DCovars(vector<Shape> &Allshapes,vector< vector<Profile1D> > &allMean1DProfiles,vector< vector<Mat> > &all1DCovarMatrices, int nSamples);
	static void trainAll2DCovars(vector<Shape> &Allshapes,vector< vector<Profile2D> > &allMean2DProfiles,vector< vector<Mat> > &all2DCovarMatrices, int nSamples);
	static void calc1DProfileCovar(vector<Shape> &Allshapes, vector<Profile1D> &mean_Profiles, vector<Mat> &covarMatrix, int nSamples, int curLevel);
	static void calc2DProfileCovar(vector<Shape> &allShapes, vector<Profile2D> &mean_Profiles, vector<Mat> &covarMatrix, int nSamples, int curLevel);
	void assign1dProfile(float m, int curLevel, int curPoint, Mat imgBordBuf);
	void train1DProfile(vector<Parts> &parts, int curLevel);
	void train2DProfile(int curLevel);
	void find1DProfile(vector<Parts> &parts, int curLevel, Mat imgBordBuf); 
	void find2DProfile(int curLevel, Mat imgBordBuf);
	void compute1DGradient(int curLevel);
	void compute2DGradient(int curLevel);

	// Load Directory related funcs
	static int loadDirectory(vector<Shape> &Allshapes, path& directoryName);
	static void readPtsFile(vector<Shape> &Allshapes, string filename);

	// PDM related funcs
	static void alignShapes(vector<Shape> &Allshapes, Shape &meanShape);
	static void calcMeanShape(Shape &meanShape, vector<Shape> &allShapes);
	static void computePCA(vector<Shape> &alignedSamples, Mat &eValues, Mat &eVectors);
	void	translation(const float x, const float y);	
	void	rotation();
	void	alignTransform(Shape refshape, float &a, float &b, float &tx, float &ty);
	void	centralize();
	void	centralize(float &x, float &y);
	void	findOrigin(float &x, float &y);
	void	alignToRef(Shape &refShape);
	void	similarityTransform(float a, float b, float tx, float ty);
	void	transformation(const float s00, const float s01, const float s10, const float s11);	
	void	conformToModel(Mat eVectors, Mat ieVectors, Mat eValues, const Shape &avgLevShape, vector<Point2f> suggestedPoints, Mat &b, const int eigMax);
	vector<Point2f> getSuggestedPoints1D(vector<Profile1D> ModelLev1DProfiles, vector<Mat> ModelLevCovars,vector<Parts> ModelParts, int ProfLen1d, float &convergence);
	vector<Point2f> getSuggestedPoints2D(vector<Profile2D> ModelLev2DProfiles, vector<Mat> ModelLevCovars,vector<Parts> ModelParts, int ProfLen2d, float &convergence);
	static Mat getPoseParameters(Mat x, vector<Point2f> sp);
	static Mat transformation(vector<Point2f> sp, Mat pose);
	static vector<Point2f> transformation(Mat m, Mat pose);

	// Face Detection related funcs
	static int alignShapesFromDetector(vector<Shape> &Allshapes, Shape &meanVJShape, float &transX, float &transY);
	int		initializeShape(const int d, const Shape s, const float transX, const float transY);

	// Other
	double	getDistance(Shape s1);
	Mat		addBorder();
	static int boundsCheck(Point2f pt, Mat img, int ProfLen);
	static void preScalePtsAndMat(vector<Shape> &Allshapes, const Size dsize, const int nSamples);
	Mat		scaleMat(Mat imageData, const Size dsize);
	void	scalePts(const float scaleX, const float scaleY);
	void	scalePtsAndMat(const float scaleX, const float scaleY, const Size dsize);
	int		writePointsFile(path outDirectory);


};
////////////////////////////////////////////////////////////////////////// CLASS DETECTOR (Abstract)
class Detector
{
public:
	virtual int init(path p) =0;
	virtual void detect(Shape &shape) const =0;
	virtual int detectAllImages(vector<Shape> &allShapes) const =0;

};
////////////////////////////////////////////////////////////////////////// CLASS VJ_DETECTOR is a DETECTOR
class VJ_Detector: public Detector
{
public:
 	int detectAllImages(vector<Shape> &allShapes) const;  // For training purposes only
 	int init(path p);
 	void detect(Shape &shape) const;
	CascadeClassifier* face_cascade;

};
////////////////////////////////////////////////////////////////////////// CLASS PP_DETECTOR is a DETECTOR
#ifdef WITH_PITTPATT
class PP_Detector: public Detector
{
public:
 	int init(path p);
 	void detect(Shape &shape) const;
 	int detectAllImages(vector<Shape> &allShapes) const;
	IISIS::ObjDetect::FaceDetector::PittPatt *pp;

};
#endif
////////////////////////////////////////////////////////////////////////// CLASS MODEL
class Model
{
	// ASM Parameters
private:
	vector<Parts> ModelParts;
	vector< vector <Profile1D> > ModelMean1DProfiles;		
	vector< vector <Profile2D> > ModelMean2DProfiles;		//
	vector< vector <Mat> > Model1DCovarMatrices;			//
	vector< vector <Mat> > Model2DCovarMatrices;			//
	Mat eigenValues;
	Mat eigenVectors;
	int nPoints;
	int nParts;
	int nLevels1D;
	int nLevels2D;
	int ProfLen1d;
	int ProfLen2d;
	int DetRefWidth;
	float transX;
	float transY;
	Shape ModelAvgShape;
	Shape DetectorAvgShape; 

public:
	// Constructor
	Model::Model(int num1DLevels, int num2DLevels, int profLength1d, int profLength2d)
	{nLevels1D = num1DLevels;
	nLevels2D = num2DLevels;
	ProfLen1d = profLength1d;
	ProfLen2d = profLength2d;
	};
	Model::Model();

	vector<Parts> getModelParts(){return ModelParts;}
	vector< vector <Profile1D> > getMean1DProfs(){return ModelMean1DProfiles;}
	vector< vector <Profile2D> > getMean2DProfs(){return ModelMean2DProfiles;}
	vector< vector <Mat> > get1DCovars(){return Model1DCovarMatrices;}
	vector< vector <Mat> > get2DCovars(){return Model2DCovarMatrices;}
	Mat getEigenValues() {return eigenValues;}
	int getDetRefWidth(){return DetRefWidth;}
	float getDetTransX(){return transX;}
	float getDetTransY(){return transY;}
	int getnLevels1D(){return nLevels1D;}
	int getnLevels2D(){return nLevels2D;}
	int getNPoints(){return nPoints;}
	int get1DprofLen(){return ProfLen1d;}
	int get2DprofLen(){return ProfLen2d;}
	Shape getDetetorAvgShape(){return DetectorAvgShape;}
	Shape getModelAvgShape(){return ModelAvgShape;}

	void setModelParts(vector<Parts> tempParts){ModelParts = tempParts; nParts = tempParts.size();}
	void setMean1DProfs(vector< vector <Profile1D> > temp1DProfs){ModelMean1DProfiles = temp1DProfs;}
	void setMean2DProfs(vector< vector <Profile2D> > temp2DProfs){ModelMean2DProfiles = temp2DProfs;}
	void set1DCovars(vector< vector <Mat> > temp1DCovars){Model1DCovarMatrices = temp1DCovars;}
	void set2DCovars(vector< vector <Mat> > temp2DCovars){Model2DCovarMatrices = temp2DCovars;}
	void setEigenValues(Mat e){eigenValues = e;}
	void setEigenVectors(Mat e){eigenVectors = e;}
	void setNPoints(int n){nPoints = n;}
	void setNLevels1D(int n){nLevels1D = n;}
	void setNLevels2D(int n){nLevels2D = n;}
	void setAvgShape(Shape s){ModelAvgShape = s;}
	void setAvgDetectorShape(Shape s){DetectorAvgShape = s;}
	void setRefWidth(int n){DetRefWidth = n;}
	void setDetTranslate(float x, float y){transX = x; transY = y;}
	void writePoints(ofstream &fileStream, Shape s);
	void writeMean1dProfs(ofstream &fileStream, vector<Profile1D> p, int lev);
	void writeMean2dProfs(ofstream &fileStream, vector<Profile2D> p, int lev);
	void writeCovars(ofstream &fileStream, vector<Mat> c, int lev);
	void writeParts(ofstream &fileStream, vector<Parts> p);

	int writeASMtoTxtFile(path outFileName);
	int writeASMtoBinFile(path outFileName);
	int loadASMfromTxtFile(path& inFileName);
	int loadASMfromBinFile(path& inFileName);

	static int loadImagesForSearch(path& pathName, vector<Shape> &toBeSearched);
	void searchLandmarks(vector<Shape> &toBeSearched, path outDirectory, Detector *det, vector<Shape> sModelShapes);
	void initSearch(vector<Shape> &toBeSearched, path outDirectory, path detectorName);
};


#endif
