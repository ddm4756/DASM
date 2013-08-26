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
	search.cpp
	This file handles all of the object detection (in our case faces and eyes)
	Including: routine to setup the detector class
			   the detection functions
*/

//-- This function handles the face detection on the training end for every shape in the set (VJ)
int VJ_Detector::detectAllImages(vector<Shape> &allShapes) const{
	
	const int nSamples = allShapes.size();

	for(int i = 0; i < nSamples; i++)
		detect(allShapes[i]);
	return SUCCESS;
	
}
//-- This function handles the face detection on the training end for every shape in the set (PP)
#ifdef WITH_PITTPATT
int PP_Detector::detectAllImages(vector<Shape> &allShapes) const{

	const int nSamples = allShapes.size();

	for(int i = 0; i < nSamples; i++)
		detect(allShapes[i]);
	return SUCCESS;

}
#endif
//-- Sets up the Viola Jones face detector from OpenCV
int VJ_Detector::init(path face_cascade_name){

	if( !face_cascade->load( face_cascade_name.string() ) ){ cout << "--(!)Error loading " << face_cascade_name.string() <<endl; return FAILURE; };

	return SUCCESS;
}
//-- Sets up the PittPatt face detector
#ifdef WITH_PITTPATT
int PP_Detector::init(path pittpattFile){

	pp = new IISIS::ObjDetect::FaceDetector::PittPatt(pittpattFile);

	return SUCCESS;
}
#endif
//-- Produces a rectangle based on the location of the largest face in an image
//-- If an image has multiple faces, you should crop each face into its own image
//-- There is no guarantee that a face will be found (or if what is detected is actually a face :-/ )
void VJ_Detector::detect(Shape &shape) const{
	
	vector<Rect> face;
	const double scaleFactor = 1.1;  // standard VJ parameter search values
	const int minNeighbors = 3;	   // 

	Mat frame_gray = shape.getOrigImgData(); 
	Size halfSize = Size(frame_gray.cols/2, frame_gray.rows/2); // Half the size of the image to speed up the detection (So don't use really small images...)
	resize(frame_gray, frame_gray, halfSize,0,0,INTER_CUBIC);

	face_cascade->detectMultiScale( frame_gray, face, scaleFactor, minNeighbors, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );  // Perform the OpenCV detection

	if(face.size() > 0){  // If face is found, continue
		
		face[0].width *= 2;   // account for the fact that we halved the image earlier
		face[0].height *= 2;  // set the box to its actual size in the image
		face[0].x *= 2;
		face[0].y *= 2;
		shape.setFaceROI(face[0]);		// Only use the biggest face found (1st face found)
	}
	else{ // Face was not found, store a -1.

		Rect temp;
		temp.width = -1;
		shape.setFaceROI(temp);
	}
}
//-- Produces a rectangle based on the location of the largest face in an image
//-- If an image has multiple faces, you should crop each face into its own image
//-- There is no guarantee that a face will be found (or if what is detected is actually a face :-/ )
#ifdef WITH_PITTPATT
void PP_Detector::detect(Shape &shape) const{

	Mat frame_gray = shape.getOrigImgData(); 
	Size halfSize = Size(frame_gray.cols/2, frame_gray.rows/2); // Half the size of the image to speed up the detection (So don't use really small images...)
	resize(frame_gray, frame_gray, halfSize,0,0,INTER_CUBIC);

	IISIS::Face f = pp->GetLargestFace(frame_gray);
	
	f.Bounds.width *= 2;
	f.Bounds.height *= 2;
	f.Bounds.x *= 2;
	f.Bounds.y *= 2;

	shape.setFaceROI(f.Bounds);
}
#endif