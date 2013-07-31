/* Copyright 2013 David Macurak

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

*/

#include "main.h"

/*
	Profile.cpp
	This file contains all of the functionality to locate and store all of the profile information
		for each individual shape. The gradient information is computed and also stored 

*/

//-- Train all profiles of every sample shape 1D and 2D
void Shape::trainAllProfiles(vector<Shape> &Allshapes, vector<Parts> &parts, Size dsize){	

	const int num1DLevels = Constants::instance()->getNum1DLevels();
	const int num2DLevels = Constants::instance()->getNum2DLevels();
	float scale = 1.0;	// We start the process at the full image resolution
	const int nSamples = Allshapes.size();

	// Train profiles for every sample at each level of resolution
	for(int i = 0; i < (num1DLevels > num2DLevels ? num1DLevels : num2DLevels); ++i)
	{	
		for(int j = 0; j < nSamples; j++)
		{	
			if(i != 0)					// First iteration is at full resolution
				Allshapes[j].scalePtsAndMat(scale, scale, dsize);	// Subsequent iterations are sampled by half the previous resolution	

			if(i < num1DLevels){		// Train 1D profiles
				vector<Profile1D> prof1d;	// Each level gets a vector of profiles
				Allshapes[j].allLev1dProfiles.push_back(prof1d); // allLev1dProfiles stores each level of vectors of profiles
				Allshapes[j].train1DProfile(parts,i);		// Train current level of 1d profiles for the entire shape	
			}
			if(i < num2DLevels){		// Train 2D profiles
				vector<Profile2D> prof2d;	// Each level gets a vector of profiles
				Allshapes[j].allLev2dProfiles.push_back(prof2d); // allLev2dProfiles stores each level of vectors of profiles
				Allshapes[j].train2DProfile(i);		// Train current level of 2d profiles for the entire shape (parts aren't needed here)
			}
		}

		scale = 0.5;							// Halve the dimensions for the next level of training
		dsize.width = dsize.width / 2.0;		//	We scale by a power of 2
		dsize.height = dsize.height / 2.0;		//
	}

}
//-- Extract the profile pixel data
//--   and compute the profile gradients
void Shape::train1DProfile(vector<Parts> &parts, int curLevel){  // Training 1D profiles

	find1DProfile(parts, curLevel, addBorder()); // Find the profile pixel data

	compute1DGradient(curLevel);    // Compute the derivative profile of gradients

}

void Shape::train2DProfile(int curLevel){

	find2DProfile(curLevel, addBorder());

}

//-- Find the 1D profile for every point of the current level
void Shape::find1DProfile(vector<Parts> &parts, int curLevel, Mat imgBordBuf){
	
	// Iterate through every point using Parts information to determine the neighbors
	for(int i = 0; i < parts.size(); i++){					// For each Part
		int lastPoint = parts[i].partPts[parts[i].partPts.size() -1]; // Get the last point of this part
		for(int k = 0; k < parts[i].partPts.size(); k++){	// For each point of the part
			int currentPoint = parts[i].partPts[k];			// Get the current point being processed on
			float x = m_Point[currentPoint].x;				// Get the coordinates of the current point
			float y = m_Point[currentPoint].y;				//
			Profile1D tempProf;
			
			allLev1dProfiles[curLevel].push_back(tempProf);	// push current profile onto this shape's vector of profiles at the current Multi-Resolution level
			// find the profile direction 
			float m = 0.0;			// slope
			// Solve for the slope  y = mx + b
			if(parts[i].getPartBoundary()){					// Closed Boundary
				if(k == 0)									// First Point?					
					m = (m_Point[lastPoint].y - m_Point[currentPoint+1].y) / (m_Point[lastPoint].x - m_Point[currentPoint+1].x);  // Get the slope of the next point and the last point of the part
				else if(k == parts[i].partPts.size() -1)	// Last Point?
					m = (m_Point[currentPoint-1].y - m_Point[parts[i].partPts[0]].y) / (m_Point[currentPoint-1].x - m_Point[parts[i].partPts[0]].x); // Get the slope of the previous point and the first point of the part
				else										// All other points
					m = (m_Point[currentPoint-1].y - m_Point[currentPoint+1].y) / (m_Point[currentPoint-1].x - m_Point[currentPoint+1].x); // Get the slope of the previous point and the next point
			}
			else{											// Open Boundary
				if(k == 0)									// First Point?
					if(currentPoint == lastPoint)			// Only Point of part? (Usually nose tip or eye center)
						m = 0.0001;		// Assign a slope close to 0.0, will cause a horizontal profile
					else
						m =  (m_Point[currentPoint].y - m_Point[currentPoint+1].y) / (m_Point[currentPoint].x - m_Point[currentPoint+1].x); // Get the slope of the current point and the next point
				else if(k == parts[i].partPts.size() -1)	// Last Point?
					m = (m_Point[currentPoint-1].y - m_Point[currentPoint].y) / (m_Point[currentPoint-1].x - m_Point[currentPoint].x); // Get the slope of the current point and the previous point
				else										// All other points
					m = (m_Point[currentPoint-1].y - m_Point[currentPoint+1].y) / (m_Point[currentPoint-1].x - m_Point[currentPoint+1].x); // Get the slope of the previous point and the next point
			}
			if(m < -100)				// check for extreme values
				m = -100;				//
			else if( m > 100)			//
				m = 100;				//
			else if(abs(m) < 1.0e-4)	//
				m = 0.0;				//

			allLev1dProfiles[curLevel][currentPoint].setSlope(m);  // Store the slope for later use in the search methods
		
			if(m != 0.0)				// don't divide by zero
				m = -1 / m;				// take the orthogonal slope
			else						// if the slope was 0, invert it to near vertical (100)
				m = 100;
			if(abs(m) < 1.0e-4)			// check for extreme values
				m = 0.0;				
			allLev1dProfiles[curLevel][currentPoint].setOrthSlope(m); // Store the orthogonal slope for later use

			assign1dProfile(m,curLevel,currentPoint, imgBordBuf);
			
		}
	}
}


//-- Adds a border to the image to account for out of bounds exceptions arising from the profiles going off the edge of the image
//-- OpenCV uses border interpolation to extrapolate the new pixels on the border
Mat Shape::addBorder()
{
	const int border = Constants::instance()->getBorder();
	Mat imgBordBuf;

	// border is global variable
	copyMakeBorder( imageData, imgBordBuf, border, border, border, border, BORDER_REPLICATE, 0);
	if(imgBordBuf.empty())
		cerr << "Error creating the border." << endl;
	return imgBordBuf;
}

//-- Find all of the 2D profiles for this level of the search
//-- Calls the functions to perform the convolution, normalization, and equalization
void Shape::find2DProfile(int curLevel, Mat imgBordBuf)
{
	const int border = Constants::instance()->getBorder();
	const int profLength2d = Constants::instance()->getProfLength2d();
	const int nPoints = m_Point.size();

	for(int i = 0; i < nPoints; i++){

		int x = m_Point[i].x + border;				// Get the coordinates of the current point
		int y = m_Point[i].y + border;				// account for the border addition to the image
		
		Profile2D *tempProf = new Profile2D(profLength2d+2, profLength2d+2);
		allLev2dProfiles[curLevel].push_back(*tempProf);	// push current profile onto this shape's vector of profiles at the current Multi-Resolution level

		Mat temp(profLength2d+2,profLength2d+2,CV_32FC1);

		Point2f pt(x,y);
		// Check for out of bounds accesses into the image (I don't actually do anything if this fails, which needs to be fixed...)
		boundsCheck(pt, imgBordBuf, profLength2d);
		
		// extract the profile roi
		Rect roi = Rect(x-((profLength2d+2)/2),y-((profLength2d+2)/2),profLength2d+2,profLength2d+2); // (x,y (of top left corner), width, height)
		temp = imgBordBuf(roi);
		
		allLev2dProfiles[curLevel][i].setProfile(temp);
		
		allLev2dProfiles[curLevel][i].convolve2dProfile(); // filter the profile

		allLev2dProfiles[curLevel][i].normalize2dProfile();  // normalize the profile

		allLev2dProfiles[curLevel][i].equalize2dProfile(); // equalize by applying sigmoid transform

	//	allLev2dProfiles[curLevel][i].weightMask2dProfile(); // weight masking (not performed)
		
	}
}

//-- Perform a 3x3 convolution filter on the profile region to determine the gradients (Actually performs correlation filter)
void Profile2D::convolve2dProfile(){

	const int profLength2d = Constants::instance()->getProfLength2d();
	Mat tempResult(profLength2d+2,profLength2d+2,CV_32FC1);

	Mat kernel = (Mat_ <float>(3,3) << 0, 0,  0,
									   0, -2, 1, 
									   0, 1,  0);	
	filter2D(profile,tempResult,CV_32F, kernel);	// (src, dst, type, filter)

	// Grab the middle of the temp matrix
	profile.create(profLength2d,profLength2d,CV_32FC1); // resize it to actual profile size
	// get the actual profile info
	for(int i = 0; i<profLength2d; i++)
		for(int j = 0; j<profLength2d; j++)
			profile.at<float>(i,j) = tempResult.at<float>(i+1,j+1);
}

//-- Normalize the 2D profile between 0-1
void Profile2D::normalize2dProfile(){

	normalize(profile,profile);
	if(profile.at<float>(0) != profile.at<float>(0))
		cerr << "ERROR Normalizing the 2d profile " << profile.at<float>(0) << endl;

}

//-- Equalize using Fast Sigmoid Transform
void Profile2D::equalize2dProfile(){

	const int profLength2d = Constants::instance()->getProfLength2d();
	const int c = 10;  // c is the shape constant

	for(int i = 0; i<profLength2d; i++){
		for(int j = 0; j<profLength2d; j++){
			profile.at<float>(i,j) = profile.at<float>(i,j) / (abs(profile.at<float>(i,j)) + c);
		}
	}
}

//-- Weight masking the 2D profile
void Profile2D::weightMask2dProfile(){
	const int profLength2d = Constants::instance()->getProfLength2d();
	const float e = 2.71828;
	const int offset = profLength2d/2;

	for(int i = 0; i < profLength2d; i++){
		for(int j = 0; j < profLength2d; j++){
			profile.at<float>(i,j) = profile.at<float>(i,j) * pow(e,(-((abs(offset-i) * abs(offset-i)) + (abs(offset-j) * abs(offset-j))) / 2.0f));	
		}
	}
}

//-- Checks if the new point locations are outside the bounds of the image
int Shape::boundsCheck(Point2f pt, Mat image, int profLen){
	// Check for out of bounds accesses into the image
	if(pt.y + ((profLen+2)/2) > image.rows){
		//cout << "Row Index Out Of Range: point " << pt.y + ((profLen+2) / 2) << " > " << image.rows << endl;
		//cout << "skipping this point in search" << endl;
		return FAILURE;
	}
	if(pt.x + ((profLen+2)/2) > image.cols)	{
		//cout << "Col Index Out Of Range: point " << pt.x + ((profLen+2) / 2) << " > " << image.cols << endl;
		//cout << "skipping this point in search" << endl;
		return FAILURE;
	}
	if(pt.y - ((profLen+2)/2) < 0) {
		//cout << "Row Index Out Of Range: point " << pt.y - ((profLen+2) / 2) << " < 0 " << endl;
		//cout << "skipping this point in search" << endl;
		return FAILURE;
	}
	if(pt.x - ((profLen+2)/2) < 0)	{
		//cout << "Col Index Out Of Range: point " << pt.x - ((profLen+2)/2) << " < 0" << endl;
		//cout << "skipping this point in search" << endl;
		return FAILURE;
	}
	return SUCCESS;
}


void Shape::assign1dProfile(float m, int curLevel, int currentPoint, Mat imgBordBuf)
{
	const int border = Constants::instance()->getBorder();
	const int profLength1d = Constants::instance()->getProfLength1d();
	float x = m_Point[currentPoint].x + border;				// Get the coordinates of the current point
	float y = m_Point[currentPoint].y + border;				// add to account for the border
	Point2f pt1, pt2, tempPoint;

	float d = profLength1d/2 + 1.1;		// Distance from the current point, Determined by profLength, add Scalar to ensure proper profile length
	if(abs(m) > .59 && abs(m) < 1.66)   // Fixes the problem when the slope is close to 1
		d = profLength1d/2 + 2.0;			// Causing the profiles to be less than the desired length

	pt1.x= x - (d / sqrt(1+(m*m)));    // solve for x
	pt1.y = m*(pt1.x-x)+y;				// solve for y
	pt2.x= x + (d / sqrt(1+(m*m)));    // solve for x	
	pt2.y = m*(pt2.x-x)+y;				// solve for y
	
	// DEBUG
	/*
	if(curLevel == 0){
		//Mat outimg;
		//cvtColor(imgBordBuf,imgBordBuf,CV_GRAY2BGR);
		line(imgBordBuf,pt1,pt2,CV_RGB(255, 0, 0),1,8);
		Point2f ppt = Point(x,y);
		circle(imgBordBuf, ppt, 1, CV_RGB(0, 0, 255));
		imshow("profiles", imgBordBuf);
		cvWaitKey();
	}
	*/
	// Use a line iterator to access the pixel coordinates along the line
	// 8 connected graph (Bresenham's algorithm)
	// If there are more than the desired pixels found, we just ignore the extras
	// We add 1 to the profLength to get the gradient for the first position in the profile
	const int connectivity = 8; // 8 or 4 connectivity for the line
	LineIterator lineIt(imgBordBuf,pt1,pt2,connectivity);

	// DEBUG
	//if(lineIt.count <= 7)
	//	cout << m <<endl;
	int pixel = 0;
	for(int j = 0; j < lineIt.count; j++, ++lineIt){
		// DEBUG
		//cout << lineIt.pos() << endl;
		tempPoint = lineIt.pos();
		// Handle pixels on the edge of the image boundaries (Shouldn't ever happen but just in case, we don't want to go out of bounds)
		if(tempPoint.x < imgBordBuf.cols || tempPoint.y < imgBordBuf.rows){ // if this happens, we'll fix it in compute1dGradient
			pixel = imgBordBuf.at<uchar>(tempPoint.y, tempPoint.x);  
			allLev1dProfiles[curLevel][currentPoint].profileIntensities1D.push_back(pixel);
			if(pixel!=pixel) // check for NAN's
				cerr << "ERROR Retrieving the pixel from the image: " << pixel << endl;
		}		
	}
}

//-- Compute the 1D gradients for each point
void Shape::compute1DGradient(int curLevel){

	const int profLength1d = Constants::instance()->getProfLength1d();
	const int nPoints = m_Point.size();

	for(int i = 0; i < nPoints; i++){	 // For every point
		allLev1dProfiles[curLevel][i].profileGradients1D.resize(profLength1d); // set the size of the derivative profile
		while(allLev1dProfiles[curLevel][i].profileIntensities1D.size() < profLength1d+1)   // Ensure the profile length is correct
			allLev1dProfiles[curLevel][i].profileIntensities1D.push_back(allLev1dProfiles[curLevel][i].profileIntensities1D[allLev1dProfiles[curLevel][i].profileIntensities1D.size()-1]); // If not, duplicate the last value of the profile and push onto the end of the profile
		for(int j = 0; j < profLength1d; j++){			  // For the length of the profile (If the profile itself is longer we just ignore the last values)
			float gradientValue = allLev1dProfiles[curLevel][i].profileIntensities1D[j+1] - allLev1dProfiles[curLevel][i].profileIntensities1D[j];  // Solve for the derivative profiles (difference of the next pixel intensity and the current intensity along the profile)
			allLev1dProfiles[curLevel][i].profileGradients1D[j] = gradientValue;	
			if(gradientValue!=gradientValue) // check for NAN's
				cerr << "ERROR problem computing the gradient value: " << gradientValue << endl;
		}
		normalize(allLev1dProfiles[curLevel][i].profileGradients1D,allLev1dProfiles[curLevel][i].profileGradients1D); // Normalize the derivative profile	
	}
}


