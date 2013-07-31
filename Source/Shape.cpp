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
	Shape.cpp
	This file handles all of the routines which align the shapes for the model
		and calculates the mean shape of the model. Also included is the functionality to conform a shape 
		to the model during search.

*/

//-- Function to perform a translation of points
void Shape::translation(const float x, const float y){

	// Assuming Origin is top left of image
	const int n = m_Point.size();
	// shift by x and y
	for(int i = 0; i < n; i++){
		m_Point[i].x += x;
		m_Point[i].y += y;
	}
}

//-- Function to resize points
void Shape::scalePts(const float scaleSizeX, const float scaleSizeY){		// Scales the points using affine transformation given the scaling parameter

	const int n = m_Point.size();

	for(int i = 0; i < n; i++){
		m_Point[i].x *= scaleSizeX;
		m_Point[i].y *= scaleSizeY;
	}
}

//-- Function to resize an image
Mat Shape::scaleMat(Mat imageData, const Size dsize){		// Scale the image to a given size

	resize(imageData, imageData, dsize,0,0,INTER_CUBIC);   // Resize image: Different interpolation methods will likely change the results
	return imageData;
}

//-- Function to pre-scale all the points and images to a standard size
void Shape::preScalePtsAndMat(vector<Shape> &allShapes, const Size dsize, const int nSamples){	// prescale the images and points to a preset size

	// TODO: Provide a better means of scaling by the aspect ratio to preserve image integrity
	for(int i = 0; i < nSamples; i++){

		if(allShapes[i].imageData.rows < dsize.height || allShapes[i].imageData.cols < dsize.width){
			// Pad image if it is less than the standard size
			Mat tempImg;
			int bottom, right;
			bottom = dsize.height - allShapes[i].imageData.rows;
			right = dsize.width - allShapes[i].imageData.cols;
			// border is global variable
			copyMakeBorder( allShapes[i].imageData, tempImg, 0, bottom, 0, right, BORDER_REPLICATE, 0);

			tempImg.copyTo(allShapes[i].imageData);
			if(allShapes[i].imageData.empty())
				cerr << "Error Padding the image." << endl;
			continue;
		}
		if(allShapes[i].imageData.rows != dsize.height || allShapes[i].imageData.cols != dsize.width){
			// Otherwise scale the image down to the standard size
			float scaleX = (float) dsize.width / allShapes[i].imageData.cols;
			float scaleY = (float) dsize.height / allShapes[i].imageData.rows;

			allShapes[i].scalePts(scaleX, scaleY);

			allShapes[i].imageData = allShapes[i].scaleMat(allShapes[i].imageData, dsize);
			allShapes[i].origImgData = allShapes[i].scaleMat(allShapes[i].origImgData, dsize);
		}
	}
}
//-- Function to scale both the points and image
void Shape::scalePtsAndMat(const float scaleX, const float scaleY, const Size dsize){
	
	scalePts(scaleX, scaleY);

	imageData = scaleMat(imageData, dsize);
}
//-- Returns the euclidean distance between the reference shape and the given shape
double Shape::getDistance(Shape refShape){		

	double d=0.0;
	const int n = m_Point.size();
	
#pragma omp parallel for reduction(+:d)  // parallel reduction
	for(int  j =0; j < n; j++){
		d += pow((refShape[j].x - this->m_Point[j].x),2) + pow((refShape[j].y - this->m_Point[j].y),2);
	}

	d = sqrt(d);
	
	return d;
}

// Similarity Transformation as given in algorithm C.3 in Appendix C of [Cootes & Taylor, 2004]
void Shape::alignTransform(Shape refshape, float &a, float &b, float &tx, float &ty){

	double Sx=0, Sy=0, Sxx=0, Syy=0, Sx1=0, Sy1=0, Sxy1=0, Syx1=0, Sxx1=0, Syy1=0;
	double W = m_Point.size();
	double x1, y1, x2, y2;
	const int n = W;

	for(int i = 0; i < n; i++){

		x1 = refshape[i].x;  // Reference Shape
		y1 = refshape[i].y;
		x2 = m_Point[i].x;	 // Source Shape
		y2 = m_Point[i].y;

		Sx += x2; 
		Sy += y2;
		Sx1 += x1;
		Sy1 += y1;
		Sxx += x2 * x2;
		Syy += y2 * y2;
		Sxx1 += x2 * x1;
		Syy1 += y2 * y1;
		Sxy1 += x2 * y1;
		Syx1 += y2 * x1;

	}
	// Ax = B
	double solnA[4][4] = { {Sxx + Syy,	0	 ,	Sx,	Sy},
						   {  0	  , Sxx + Syy, -Sy, Sx},
						   { Sx	  ,	   -Sy	 ,  W , 0 },
						   {  Sy  ,	    Sx   ,  0 , W } };
	Mat A = Mat(4,4, CV_64FC1, solnA);

	double solnB[4] = { Sxx1 + Syy1, Sxy1 - Syx1, Sx1, Sy1 };
	Mat B = Mat(4,1,CV_64FC1, solnB); 

	Mat C(1,4, CV_64FC1);  // The solution vector
	solve(A, B, C, CV_SVD); // OpenCV function to solve the system of linear equations

	a = C.at<double>(0, 0);
	b = C.at<double>(1, 0);  
	tx = C.at<double>(2,0);
	ty = C.at<double>(3,0);

}

void Shape::alignShapes(vector<Shape> &allShapes, Shape &meanShape){	// Function to align all the shapes and calculate the mean shape

	const int nSamples = allShapes.size();
	// Reset shapes to their original points because we altered them during the profile search
#pragma omp parallel for
	for(int i = 0; i < nSamples; i++){
		allShapes[i].m_Point = allShapes[i].orig_Point;
	}

	// translate all shapes to the origin
#pragma omp parallel for
	for(int i = 0; i < nSamples; i++){
		allShapes[i].centralize();
	}

	calcMeanShape(meanShape,allShapes);		// calculate initial mean shape 

	Shape initRefShape = meanShape;		// Assign an initial reference shape to constrain the model
	Shape ref_Shape = meanShape;		// Assign reference shape to the mean shape

	float errDistance;
	const float minErr = 0.00019;
	const int max_i = 30;

	for(int iter = 0; iter < max_i; iter++)
	{
#pragma omp parallel for // Parallelize the entire alignment
		for(int i = 0; i < nSamples; i++)
		{ 
			allShapes[i].alignToRef(ref_Shape);
		}

		calcMeanShape(meanShape, allShapes);	// Re-estimate the mean shape from the newly aligned shapes
		
		meanShape.alignToRef(initRefShape);		// Constrain new mean shape by aligning it to the initial ref shape
		
		errDistance = meanShape.getDistance(ref_Shape);		// Find the total error distance

		if(Constants::instance()->isVerbose())
			cout << "Iteration: " << iter << "  Error Distance: " << errDistance << endl;

		if(errDistance < minErr)
			break; // Break when the minimum error threshold is reached.
		
		ref_Shape = meanShape;	// Set reference shape to newly acquired mean shape
	}
}

void Shape::alignToRef(Shape &refShape){
	// similarity transform variables
	float a, b, tx, ty;
	// Find the values for the similarity transformation
	alignTransform(refShape, a, b, tx, ty);
	// Perform the similarity transformation
	similarityTransform(a, b, tx, ty);

}

// Perform the similarity transformation
void Shape::similarityTransform(float a, float b, float tx, float ty){
	
	// T(x) = { a -b, b a }x + { tx, ty }

	transformation(a, -b, b, a);

	// | a, -b, tx |
	// | b,  a, ty |
	// | 0,  0,  1 |
	
	translation(tx, ty);
}

// Perform the transformation
void Shape::transformation(const float s00, const float s01, const float s10, const float s11){

	float x, y;
	const int n = m_Point.size();
	for(int i = 0; i < n; i++)
	{
		x = m_Point[i].x;
		y = m_Point[i].y;
		m_Point[i].x = s00*x + s01*y;
		m_Point[i].y = s10*x + s11*y;
	}
}
// Perform the transformation (overloaded 1)
Mat Shape::transformation(vector<Point2f> sp, Mat pose){

	Mat out_Points(sp.size(),2,CV_32FC1);
	for(int i = 0; i < sp.size(); i++){
		// This transform performs scaling rotation and translation
		const float x = sp[i].x;
		const float y = sp[i].y;
		out_Points.at<float>(i,0) = pose.at<float>(0,0) * x + pose.at<float>(0,1) * y + pose.at<float>(0,2);
		out_Points.at<float>(i,1) = pose.at<float>(1,0) * x + pose.at<float>(1,1) * y + pose.at<float>(1,2);

	}

	return out_Points;
}
// Perform the transformation (overloaded 2)
vector<Point2f> Shape::transformation(Mat m, Mat pose){

	vector<Point2f> finalShape;
	Point2f newPoint;
	for(int i = 0; i < m.rows/2; i++){

		const float x = m.at<float>(i*2);
		const float y = m.at<float>(i*2 + 1);
		newPoint.x = pose.at<float>(0,0) * x + pose.at<float>(0,1) * y + pose.at<float>(0,2);
		newPoint.y = pose.at<float>(1,0) * x + pose.at<float>(1,1) * y + pose.at<float>(1,2);
		finalShape.push_back(newPoint);
	}

	return finalShape;
}

void Shape::calcMeanShape(Shape &meanShape, vector<Shape> &allShapes){
	
	meanShape.m_Point.resize(allShapes[0].m_Point.size()); // resize will initialize each element to 0 to avoid problems with the summation below
	const int nSamples = allShapes.size();
	const int n = allShapes[0].m_Point.size();

	for (int i = 0; i < nSamples; i++){
		for(int j = 0; j < n; j++){
		// sum the x's and y's of all the shapes
			meanShape.m_Point[j].x += allShapes[i].m_Point[j].x;
			meanShape.m_Point[j].y += allShapes[i].m_Point[j].y;

		}
	}
	// This will scale the mean shape
	for (int i = 0; i < n; i++){

		meanShape.m_Point[i].x /= (float)nSamples;
		meanShape.m_Point[i].y /= (float)nSamples;
	}

}
//-- Centralize the shape to the origin
void Shape::centralize(){

	float x, y;
	findOrigin(x,y);  // Find the distance to the origin

	translation(-x,-y); // translate by the distance found prior
}
//-- Centralize the shape to the origin
void Shape::centralize(float &x, float &y){

	findOrigin(x,y);  // Find the distance to the origin

	translation(-x,-y); // translate by the distance found prior
}
//-- Locate the shape origin
void Shape::findOrigin(float &x, float &y){

	x = 0.0;
	y = 0.0;
	const int n = m_Point.size();

	for (int i = 0; i < n; i++){ // sum the x's and y's
		x += m_Point[i].x;
		y += m_Point[i].y;
	}
	x = x / m_Point.size();      // find the mean x and y
	y = y / m_Point.size();
	
}	
//-- Align current shape to the detected average shape
int Shape::alignShapesFromDetector(vector<Shape> &allShapes, Shape &meanDetectedShape, float &transX, float &transY){

	meanDetectedShape.setNPoints(allShapes[0].getNPoints());
	meanDetectedShape.m_Point.resize(allShapes[0].m_Point.size());
	vector <Shape> totalDetectedShapes;
	const int nSamples = allShapes.size();
	float ref_width = 0;
	float count = 0;

	for(int i = 0; i < nSamples; i++){	// Determine the reference width of the detected rectangle for the model
		if(allShapes[i].getFaceROI().width > 0){	// we found a face for this image
			count++;
			ref_width += allShapes[i].getFaceROI().width;	// Sum the widths of each box
		}
	}
	ref_width /= count;	// take the avg of the sum

	if(Constants::instance()->isVerbose())
		cout << "Reference width: " << ref_width << endl;

	float boxCenterSumX = 0, boxCenterSumY = 0;
	float shapeMeanSumX = 0, shapeMeanSumY = 0;
	count = 0;
	for(int i = 0; i < nSamples; i++){

		if(allShapes[i].getFaceROI().width > 0){	// we found a face for this image
			++count;
			
			Shape shape;			
			shape.m_Point = allShapes[i].orig_Point;

			int cx, cy;
			CvRect rct = allShapes[i].getFaceROI();
			cx = rct.x + rct.width*0.5;  // center coordinates of the rectangle
			cy = rct.y + rct.height*0.5;

			// Find the mean of the current shape
			//		sum across all of the detected shapes and sum the center of the detected boxes
			//		take the average of the sum of each
			//		determine the relationship between the two means according to the ratio of the ref width
			boxCenterSumX += cx;
			boxCenterSumY += cy;
		
			float sx, sy;
			shape.findOrigin(sx, sy);

			shapeMeanSumX += sx;
			shapeMeanSumY += sy;

			// The remaining part of this loop is for finding the detected average shape (which I don't think is necessary)
			shape.translation(-cx, -cy);  // translate to the center of the rectangle

			// Tells the model how to scale the points according to the width of the detected box
			shape.scalePts(ref_width / rct.width, ref_width / rct.height);  // scale by the reference width
			
			totalDetectedShapes.push_back(shape);
		}
	}
	// Compute the averages
	boxCenterSumX /= (float)count;
	boxCenterSumY /= (float)count;
	shapeMeanSumX /= (float)count;
	shapeMeanSumY /= (float)count;

	// Find the difference (x,y) from the mean center of the detected box
	float diffX = boxCenterSumX - shapeMeanSumX;
	float diffY = boxCenterSumY - shapeMeanSumY;

	transX = diffX / ref_width;
	transY = diffY / ref_width;

	// Calculate the mean detector shape
	calcMeanShape(meanDetectedShape,totalDetectedShapes);

	if(Constants::instance()->isVerbose())
		cout << "Found this many faces: " << count << endl;

	return ref_width;
}

//-- Conform the newly found points to the model  ( x = x_m + phi * b)
void Shape::conformToModel(Mat eigVectors, Mat t_eigVectors, Mat eigValues, const Shape &ModelShape, vector<Point2f> suggestedPoints, Mat &b, const int eigMax){

	/* 1. Initialize  the shape parameters b to zero for the first iteration(Gets passed in)
	   2. Generate the model instance x = model_x + eigVectors * b
	   3. Find the pose parameters which map x to Y
	   4. Invert the pose parameters and project Y into the model coordinate frame
	   5. Update the model parameters to match Y: b = transform(eigVectors) * (Y - model_x)
	   6. Apply constraints to b
	   7. Update current positions with the new instance
	   8. Transform back to the shape origin
	   9. Loop until convergence or max iterations
	*/
	
	// Convert the ModelShape's points to a 1d column vector (x,y,x,y ... )
	Mat model_x = Mat(ModelShape.m_Point).reshape(1,1).t();

	Mat x = model_x + (t_eigVectors * b);	// x is the new model instance (first iteration will be the same as the model points)

	Mat pose(3,3,CV_32FC1);
	pose = getPoseParameters(x, suggestedPoints); // Get the pose parameters by aligning the model instance to my suggested points

	Mat ipose;
	invert(pose,ipose);	

	Mat y = transformation(suggestedPoints, ipose);

	Mat y_asColVector = y.reshape(1,1).t(); // npoints x 2 converted to npoints*2 x 1

	b = eigVectors * (y_asColVector - model_x);
	
	// Apply constraints to b
	int i;
	for(i = 0; i < eigMax; i++){
		float maxB = 1.8*sqrt(eigValues.at<float>(i));
		if(b.at<float>(i) > maxB)
			b.at<float>(i) = maxB;
		else if(b.at<float>(i) < -maxB)
			b.at<float>(i) = -maxB;
	}
	while(i<b.rows)
		b.at<float>(i++) = 0.0;

	Mat newInstance = t_eigVectors * b;  // create the new instance

	Mat outPoints = model_x + newInstance;

	m_Point = transformation(outPoints, pose); // move the new points back to the shape origin and set to this->m_points
	
}

//-- Find this shape's pose parameters and return them
Mat Shape::getPoseParameters(Mat _x, vector<Point2f> suggestedPoints)
{
	Mat parameters(3,3,CV_32FC1);

	float a, b, tx, ty;
	// Find the values for the similarity transformation
	
	double Sx=0, Sy=0, Sxx=0, Syy=0, Sx1=0, Sy1=0, Sxy1=0, Syx1=0, Sxx1=0, Syy1=0;
	double W = suggestedPoints.size();
	double x1, y1, x2, y2;
	int n = W;

	for(int i = 0; i < n; i++){

		x1 = suggestedPoints[i].x;  // Suggested Points
		y1 = suggestedPoints[i].y;
		x2 = _x.at<float>(i*2);	    // Model Instance x
		y2 = _x.at<float>(i*2 + 1);

		Sx += x2; 
		Sy += y2;
		Sx1 += x1;
		Sy1 += y1;
		Sxx += x2 * x2;
		Syy += y2 * y2;
		Sxx1 += x2 * x1;
		Syy1 += y2 * y1;
		Sxy1 += x2 * y1;
		Syx1 += y2 * x1;

	}
	// Ax = B
	double solnA[4][4] = { {Sxx + Syy,	0	 ,	Sx,	Sy},
						{  0	  , Sxx + Syy, -Sy, Sx},
						{ Sx	  ,	   -Sy	 ,  W , 0 },
						{  Sy  ,	    Sx   ,  0 , W } };
	Mat A = Mat(4,4, CV_64FC1, solnA);

	double solnB[4] = { Sxx1 + Syy1, Sxy1 - Syx1, Sx1, Sy1 };
	Mat B = Mat(4,1,CV_64FC1, solnB); 

	Mat C(1,4, CV_64FC1);  // The solution vector
	solve(A, B, C, CV_SVD); // OpenCV function to solve the system of linear equations

	a = C.at<double>(0, 0);
	b = C.at<double>(1, 0);  
	tx = C.at<double>(2,0);
	ty = C.at<double>(3,0);

	parameters.at<float>(0,0) = a;
	parameters.at<float>(0,1) = -b;
	parameters.at<float>(0,2) = tx;
	parameters.at<float>(1,0) = b;
	parameters.at<float>(1,1) = a;
	parameters.at<float>(1,2) = ty;
	parameters.at<float>(2,0) = 0;
	parameters.at<float>(2,1) = 0;
	parameters.at<float>(2,2) = 1;

	return parameters;
}

