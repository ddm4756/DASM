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
	Search.cpp
	This file contains the functionality to perform image search:
	
	1. Load 
		1.1 the model and 
		1.2 the input directory of images/single image
	2. Initialize the face detector and model shapes
	3. For each input image
		3.1 Detect the face in the image, if successful continue, else skip image
		3.2 Initialize the shape using the model average shape and the detected bounding rectangle
		3.3 Perform the profile search (1D or 2D) (starting at the lowest resolution)
		3.4 Find the suggested landmark movements based on the model mean profiles
		3.5 Conform suggested shape to the Model shape
		3.6 Repeat for some finite number of iterations or until converged
		3.7 Loop for each level of image resolution
		3.8 Save out the points to file

*/

int searchASM(path inDir, path modelPath, path detectorName, path outDirectory){

	if(!is_directory(outDirectory)) // If the directory already exists, use it.
	{
		if(!create_directory(outDirectory)){ // If it does not exist, let's try to create it
			cout << "Could not create the output directory for the points: " << outDirectory.string() << endl;
			outDirectory = current_path();
		}
	}
	cout << "Points will be output to: " << outDirectory.string() << endl;
	Model model(0,0,0,0); // Create the model instance

	if(!model.loadASMfromBinFile(modelPath)){ // Load the model from the binary file.
		cerr << "Model failed to load from: " << modelPath.string() << endl; 
		return FAILURE;
	}
	Constants::instance()->setFromModel(model.getnLevels1D(), model.getnLevels2D(), model.get1DprofLen(), model.get2DprofLen()); // Set the singleton variables for global access to model parameters

	// Determine the amount of eigenvalues to use for the search
	const float retainPercent = Constants::instance()->getEigPercent();
	Mat modelEigValues = model.getEigenValues();
	Scalar valSum = sum(modelEigValues); // based on the sum of the eigenvalues
	valSum.val[0] *= retainPercent;  
	int idx = 0;
	while (valSum.val[0] > 0)
	{
		valSum.val[0] -= modelEigValues.at<float>(idx);
		idx++;
	}

	Constants::instance()->setNE(idx);	// set the number of eigenvalues to the singleton

	vector<Shape> toBeSearched; // vector which holds all of the images to be searched
	Model::loadImagesForSearch(inDir, toBeSearched);
	int nShapes = toBeSearched.size();


	int DetRefWidth = model.getDetRefWidth();
	//ASM_Shape ModelAvgShape = Model.getDetetorAvgShape();     
	Shape ModelAvgShape = model.getModelAvgShape(); // initialize using the Model avg shape instead of the detected avg shape
	
	double start = omp_get_wtime( ); // record start time

	model.initSearch(toBeSearched, outDirectory, detectorName); // Begin the search

	double end = omp_get_wtime( );  // record end time
	pt::ptime now = pt::second_clock::local_time();
	if(Constants::instance()->isVerbose()){
		cout << "\nSearch Finished " << now.date() << " " << now.time_of_day() << endl;
		cout << "Average search time per image: " << (end-start) / nShapes << endl;
	}
	return SUCCESS;
}

//-- Initialize the search, detector, and model shapes
void Model::initSearch(vector<Shape> &toBeSearched, path outDirectory, path detectorName){

	// Setup the Face Detector
	Detector *det;
	if(Constants::instance()->getDetector_t() == Constants::VJ){
		VJ_Detector VJDet;  // Detector object
		VJDet.face_cascade = new CascadeClassifier();
		det = &VJDet;
		det->init(detectorName); // Setup VJ detector
	}
#ifdef WITH_PITTPATT
	else if(Constants::instance()->getDetector_t() == Constants::PP){
		PP_Detector PPDet;
		det = &PPDet;
		det->init(detectorName); // Setup PittPatt detector
	}
#endif
	vector<Shape> scaledModelShapes;
	scaledModelShapes.push_back(ModelAvgShape); // Level 0 is the full resolution image/points
	for(int lev = 1; lev < (nLevels1D > nLevels2D ? nLevels1D : nLevels2D); lev++){  // scale the Model shape for each level, will be used when we conform to the shape model
		ModelAvgShape.scalePts(0.5, 0.5);
		scaledModelShapes.push_back(ModelAvgShape); // Each level is half the resolution of the previous
		// ModelAvgShape has now been changed, so use the vector to access each levels model shape from here on.
	}


	searchLandmarks(toBeSearched, outDirectory, det, scaledModelShapes);
	

}

//-- Perform the landmark search using 2D (and 1D profiles if applicable)
void Model::searchLandmarks(vector<Shape> &toBeSearched, path outDirectory, Detector *det, vector<Shape> scaledModelShapes){

	// Initializations
	const int stacked = Constants::instance()->getStacked();
	const int numEigs = Constants::instance()->getNumEigs();
	const Mat EigenVectors_t = eigenVectors.t();	// transposed eigenVectors matrix is used during the shape alignment to the model
	const int nShapes = toBeSearched.size();

#pragma omp parallel // Begin Parallel Region
	{

	int scaleFactor;
	int flag = 1;
	Size dsize;
	if(nLevels1D>nLevels2D)		// Determine the scaling factor for the image pyramid
		scaleFactor = pow(2,(float)nLevels1D-1); // 2^n
	else
		scaleFactor = pow(2,(float)nLevels2D-1);

#pragma omp for schedule(guided)
	for(int i = 0; i < nShapes; i++){  // For each image in the directory

#pragma omp critical // face detection function occasionally was causing reading violations in parallel so I made it a critical section
		{				// If you know your face detector is thread safe, remove the critical
			det->detect(toBeSearched[i]);	// Perform the face detection for this image
		}
		
		if(Constants::instance()->isVerbose()){ 
			#pragma omp critical // protect the std out from threads overwriting each other
			{
				cout << toBeSearched[i].getFilename() << endl;
			}			
		}
		
		toBeSearched[i].setNPoints(ModelAvgShape.getNPoints());  // Set the number of points for the image (Will match the model nPoints)

		for(int secondPass = 0; secondPass < stacked; secondPass++){   // Perform the search again, using the previous shape as the reference shape
			// It would be more sensible to move this loop outside of this function to allow for more flexibility to what model options 
			//	can be adjusted for each search. 

			if(nLevels1D>nLevels2D) // set up the data structures
				toBeSearched[i].setup1dDataStructures(nLevels1D - nLevels2D);
			toBeSearched[i].setup2dDataStructures(nLevels2D);
			
			if(secondPass == 0)
				flag = toBeSearched[i].initializeShape(DetRefWidth,scaledModelShapes[0],transX,transY);	// Initialize each shapes starting points to the model's points

			if(flag == 0) // Face detection failed, so skip this image.
				continue;

			int eigMax = numEigs; // # of eigen values that are used in the shape alignment

			if(nLevels1D > nLevels2D){  // If the number of 1d levels is greater than the number of 2d levels

				// Perform the search process for each of the 1d profile levels (if there are any)
				for(int lev = nLevels1D; lev > nLevels2D ; lev--){	// for each level of resolution, start at the lowest level (ex. 4 = lowest, 1 = highest) until nLevels2D
					// reset image data to the original size each level of the pyramid.

					toBeSearched[i].setImgData(toBeSearched[i].getOrigImgData());
					// scale the points and image to the current resolution

					if(lev == nLevels1D)
					{  // Scale down to the lowest resolution on the first pass
						dsize = Size(toBeSearched[i].getImgData().cols/scaleFactor,toBeSearched[i].getImgData().rows/scaleFactor);

						toBeSearched[i].scalePtsAndMat((float)1/scaleFactor, (float)1/scaleFactor, dsize);
					}
					else
					{  // Double the previous resolution
						// translate to the origin prior to scaling, then translate back to the new center
						float x,y;
						toBeSearched[i].centralize(x,y);
						dsize.width = dsize.width * 2.0;		//
						dsize.height = dsize.height * 2.0;		//
						toBeSearched[i].scalePtsAndMat(2.0, 2.0, dsize);
						toBeSearched[i].translation(x*2,y*2);
					}
					
					Mat b = Mat::zeros(eigenVectors.rows, 1, CV_32FC1);	// b is a column vector: number of rows of the eigen vectors x 1, initialized to zero each level

					for(int maxIters = 0; maxIters < MAX_ITERATIONS; maxIters++){	// Determines the maximum iterations it will search per level

#ifdef DEBUG						
						//debug
						/////////////////////////////////////////////////////////////////////////////////////////////
						Scalar red(0,0,255);
						Mat test;
						cvtColor(toBeSearched[i].getImgData(),test,CV_GRAY2BGR);
						for(int qq = 0; qq < nPoints; qq++)
						{
							Point xp = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);

							circle(test, xp, 1, red);
						}
						imshow("before",test);
						cvWaitKey(0);
						/////////////////////////////////////////////////////////////////////////////////////////////
#endif
						float convergence = 0.0;
						
						// Find the suggested movements along the 1d profiles for the new points
						vector<Point2f> suggestedPoints = toBeSearched[i].getSuggestedPoints1D(ModelMean1DProfiles[lev-1], Model1DCovarMatrices[lev-1], ModelParts, ProfLen1d, convergence);
					
#ifdef DEBUG
						////////////////////////////////////////////////////////////////////////////////////////////
						//  debug
						cvtColor(toBeSearched[i].getImgData(),test,CV_GRAY2BGR);
						for(int qq = 0; qq < nPoints; qq++){

							//Point x = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);
							circle(test, suggestedPoints[qq], 1, Scalar(255,0,0));
						}
						imshow("after",test);
						cvWaitKey(0);
						/////////////////////////////////////////////////////////////////////////////////////////////
#endif


						// Conform suggested movements to the shape model
						toBeSearched[i].conformToModel(eigenVectors, EigenVectors_t, eigenValues, scaledModelShapes[lev-1], suggestedPoints, b, eigMax);
					
#ifdef DEBUG
						/////////////////////////////////////////////////////////////////////////////////////////////
						// debug
						for(int qq = 0; qq < nPoints; qq++){

							Point x = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);
							circle(test, x, 1, Scalar(0,255,0));
						}
						imshow("after",test);
						cvWaitKey(0);
						/////////////////////////////////////////////////////////////////////////////////////////////
#endif

						// Stopping criteria
						convergence = (float)convergence / nPoints;
					
						if(convergence > MIN_CONVERGENCE1D){
							break;
						}
						
					} // end max iterations loop
				} // end level loop
			} // end if 1d > 2d

			// Perform the remainder of the search levels for 2d profiles
			for(int lev = nLevels2D; lev > 0; lev--)
			{  
				// reset image data to the original size each level of the pyramid.
				toBeSearched[i].setImgData(toBeSearched[i].getOrigImgData());

				if(nLevels1D <= nLevels2D && lev == nLevels2D){ // if true: We are only searching with 2d profiles, so setup the image and points accordingly
					dsize = Size(toBeSearched[i].getImgData().cols/scaleFactor,toBeSearched[i].getImgData().rows/scaleFactor);
					toBeSearched[i].scalePtsAndMat((float)1/scaleFactor, (float)1/scaleFactor, dsize);
				}
				else{
					// scale the points and image to the current resolution
					float xp,yp;
					toBeSearched[i].centralize(xp,yp);
					dsize.width = dsize.width * 2.0;		
					dsize.height = dsize.height * 2.0;		
					toBeSearched[i].scalePtsAndMat(2.0, 2.0, dsize);
					toBeSearched[i].translation(xp*2,yp*2);
				}
				
				Mat b = Mat::zeros(eigenVectors.rows, 1,CV_32FC1);	// b is a column vector: number of rows of the eigen vectors x 1, initialized to zero each level

				for(int iter = 0; iter < MAX_ITERATIONS; iter++){	// Determines the maximum iterations it will search per level
#ifdef DEBUG
					/////////////////////////////////////////////////////////////////////////////////////////////
					Scalar red(0,0,255);
					Mat test;
					cvtColor(toBeSearched[i].getImgData(),test,CV_GRAY2BGR);
					for(int qq = 0; qq < nPoints; qq++){

						Point xpt = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);

						circle(test, xpt, 1, red);
					}
					imshow("before",test);
					cvWaitKey(0);
					/////////////////////////////////////////////////////////////////////////////////////////////
#endif

					float convergence = 0.0;
					vector<Point2f> suggestedPoints = toBeSearched[i].getSuggestedPoints2D(ModelMean2DProfiles[lev-1],Model2DCovarMatrices[lev-1],ModelParts,ProfLen2d,convergence);
#ifdef DEBUG
					////////////////////////////////////////////////////////////////////////////////////////////
					//  debug
					cvtColor(toBeSearched[i].getImgData(),test,CV_GRAY2BGR);
					for(int qq = 0; qq < nPoints; qq++){

						//Point x = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);
						circle(test, suggestedPoints[qq], 1, Scalar(255,0,0));
					}
					imshow("after",test);
					cvWaitKey(0);
					/////////////////////////////////////////////////////////////////////////////////////////////
#endif

					// Conform suggested movements to the shape model		
					toBeSearched[i].conformToModel(eigenVectors, EigenVectors_t, eigenValues, scaledModelShapes[lev-1], suggestedPoints, b, eigMax);
#ifdef DEBUG
					/////////////////////////////////////////////////////////////////////////////////////////////
					// debug
					for(int qq = 0; qq < nPoints; qq++){

						Point x = Point(toBeSearched[i].getPoint(qq).x,toBeSearched[i].getPoint(qq).y);
						circle(test, x, 1, Scalar(0,255,0));
					}
					imshow("after",test);
					cvWaitKey(0);
					/////////////////////////////////////////////////////////////////////////////////////////////
#endif

					// Convergence?
					convergence = (float)convergence / nPoints;

					if(convergence > MIN_CONVERGENCE2D){				
						break;
					}
				} // end iteration loop

				if(lev == 2) // Loosen up the model for the final iteration (full image resolution)
					eigMax+=10;
					if(getEigenValues().rows < eigMax) // if we accidentally went over the # of eigs we have, set it back.
						eigMax-=10;

			}// end level loop
		} // end second pass loop

		if(flag != 0){
			if(toBeSearched[i].writePointsFile(outDirectory) == 0)
				cerr << "Error Writing out points file: " << toBeSearched[i].getFilename() << endl;
		}
	} // end for each shape loop
	} // end parallel region
} // end function


//-- Get the suggested movements of points from 1D profile search
vector<Point2f> Shape::getSuggestedPoints1D(vector<Profile1D> ModelLev1DProfiles, vector<Mat> ModelLevCovars,vector<Parts> ModelParts, int ProfLen1d, float &convergence )
{
	const int border = Constants::instance()->getBorder();
	vector<Point2f> newPoints; // Points to be returned out
	Mat imgWithBorder = addBorder(); // Include a border to avoid points searching outside the bounds of the image, the original image is not affected

	for(int i = 0; i < nPoints; i++){
		// initial distance value
		float distance = 1e10;

		// I send this all the shape's points because it uses the prior and next points to determine the profile
		float m = Profile1D::findSlopeOfProfile(m_Point, ModelParts, i );
		float x = m_Point[i].x + border;  // account for the addition of the border
		float y = m_Point[i].y + border;
		Point2f newPt;
		newPt.x = x;
		newPt.y = y;
		// Finds all the offset points along the current point's profile
		vector<Point2f> potentialProfiles = Profile1D::findProfileOffsets(m, x, y, imgWithBorder, ProfLen1d); // imgWithBorder  imageData
		int pSize = potentialProfiles.size();
		// The covariance matrix is already inverted from the training stage
		Mat icovar = ModelLevCovars[i];  // grab the inverse covariance Mat for this point
		int finalPt = 0; // used for determining convergence
		
		for(int p = 0; p < pSize; p++)
		{ // for each offset of the profile, find the best match to the model profile
			vector<float> profileGradients = Profile1D::findProfile1D(potentialProfiles[p].x, potentialProfiles[p].y, m, ProfLen1d, imgWithBorder); 
			Profile1D::calcProfGradient1D(profileGradients,ProfLen1d); 
			
			// Distance = (p(k) - m(k))' * Cp^-1 * (p(k) - m(k))
			float temp = Mahalanobis(profileGradients,ModelLev1DProfiles[i].profileGradients1D,icovar);
			if(temp < distance){ // Record the point with the most similar profile to the model point at this level
				// keep the x y coordinates of this profile's point
				distance = temp;
				newPt.x = potentialProfiles[p].x - border;
				newPt.y = potentialProfiles[p].y - border;
				finalPt = 0;
				if(p > pSize/2 - NUM_OFFSETS && p < pSize/2 + NUM_OFFSETS) // if the current point in the profile search is within the minimum distance from the old point, record it for convergence
					finalPt = 1;
			}
		}
		newPoints.push_back(newPt); // Update the vector to the newly found point

		if(finalPt == 1) // The new point is within NUM_OFFSETS of the old point, meaning that this point is converging
			convergence++;	

	}

	return newPoints;
}

//-- This function returns the suggested movements of the points using a 2D profile search
vector<Point2f> Shape::getSuggestedPoints2D(vector<Profile2D> ModelLev2DProfiles, vector<Mat> ModelLevCovars,vector<Parts> ModelParts, int profLen2d, float &convergence)
{
	const int border = Constants::instance()->getBorder();
	vector<Point2f> newPoints;	// Points to be returned out
	Mat imgWithBorder = addBorder(); // Include a border to avoid points searching outside the bounds of the image, the original image is not affected
	
	for(int i = 0; i < nPoints; i++)
	{
		// initial distance value
		float distance = 1e10;

		// I send this all the shape's points because it uses the prior and next points to determine the profile
		float m = Profile1D::findSlopeOfProfile(m_Point, ModelParts, i );  // Slope is needed to determine the profile search direction

		Point2f newPt;
		vector<Point2f> potentialProfiles;
		float d = -1*(profLen2d/2);		// Distance from the current point, Determined by profLength

		float x = m_Point[i].x + border;			// Get the coordinates of the current point
		float y = m_Point[i].y + border;			// account for the border addition to the image
		// 
		for(int j = 0; j < (profLen2d/2) + 1; j++, d+=2)
		{
			
			newPt.x = x + (d / sqrt(1 + (m*m)));    // solve for x
			newPt.y = (m * (newPt.x - x) + y);		// solve for y
			//circle(test, newPt, 1, Scalar(255,0,0));
			if(boundsCheck(newPt,imgWithBorder,profLen2d))
				potentialProfiles.push_back(newPt); // if it passed the bounds checks, push it onto our list of potential points to search
			
		}
		newPt.x = x; // Neighbor above current point
		newPt.y = y - 1;
		//circle(test, newPt, 1, Scalar(255,0,0));
		if(boundsCheck(newPt,imgWithBorder,profLen2d))
			potentialProfiles.push_back(newPt);
		newPt.y = y + 1; // Neighbor below current point
		//circle(test, newPt, 1, Scalar(255,0,0));
		if(boundsCheck(newPt,imgWithBorder,profLen2d))
			potentialProfiles.push_back(newPt);

		//imshow("potential",test);
		//cvWaitKey(0);

		int finalPt = 0;

		// Check x y and neighbors
		int pSize = potentialProfiles.size();

		for(int j = 0; j < pSize; j++)
		{
			Mat temp(profLen2d+2,profLen2d+2,CV_32F); // This is our filter space

			Point2f curPoint = potentialProfiles[j];

			Rect roi = Rect(curPoint.x-((profLen2d+2)/2),curPoint.y-((profLen2d+2)/2),profLen2d+2,profLen2d+2); // (x,y (of top left corner), width, height) curPoint x and y represents the center coordinate of our 2d profile
			temp = imgWithBorder(roi); // Extract the roi from the image w/ the border

			Profile2D tempProf;			// create a temporary profile just for the ease of accessing the profile functions
			
			tempProf.setProfile(temp);
			tempProf.convolve2dProfile();  // convolve the profile
			tempProf.normalize2dProfile(); // normalize the profile
			tempProf.equalize2dProfile();  // equalize the profile
		
			vector<float> pf = tempProf.getProfile().reshape(0,1);	
			vector<float> mf = ModelLev2DProfiles[i].getProfile().reshape(0,1);
			
			float tempd = Mahalanobis(pf,mf,ModelLevCovars[i]); // Compare to the model profile

			if(tempd < distance){
				// keep the x y coordinates of this profile
				distance = tempd;
				newPt.x = curPoint.x - border;	// subtract back the border
				newPt.y = curPoint.y - border;

				finalPt = 0;
				if((newPt.x < m_Point[i].x + NUM_OFFSETS && newPt.x > m_Point[i].x - NUM_OFFSETS) &&
						(newPt.y < m_Point[i].y + NUM_OFFSETS && newPt.y > m_Point[i].y - NUM_OFFSETS)) // if the current point in the profile search is within the minimum distance from the old point, record it for convergence
					finalPt = 1;
			}
		}
		newPoints.push_back(newPt);
		if(finalPt == 1) // The new point is within NUM_OFFSETS of the old point, meaning that this point is converging
			convergence++;	
	}

	return newPoints;
}

//-- Calculate the profile gradients and normalize the vector
void Profile1D::calcProfGradient1D(vector<float> &profileGradients, const int profLen1d)
{
	while(profileGradients.size() < profLen1d+1)   // Ensure the profile length is correct
		profileGradients.push_back(profileGradients[profileGradients.size()-1]); // If not, duplicate the last value of the profile and push onto the end of the profile
	for(int j = 0; j < profLen1d; j++){			  // For the length of the profile (If the profile itself is longer we just ignore the last values)
		float gradientValue = profileGradients[j+1] - profileGradients[j];  // Solve for the derivative profiles (difference of the next pixel intensity and the current intensity along the profile)
		profileGradients[j] = gradientValue;	
		if(gradientValue != gradientValue) // NAN check
			cerr << "ERROR problem computing the gradient value: " << gradientValue << endl;
	}
	while(profileGradients.size() > profLen1d)  // Remove extra profile values off the end
		profileGradients.pop_back();
	
	normalize(profileGradients,profileGradients);	// Normalize the profile

}

//-- Find the intensity values of each pixel of the profile
vector<float> Profile1D::findProfile1D(const int x, const int y, const float m, const int profLen1d, Mat imageData)
{ 

	vector<float> profilePoints; // vector that will be returned

	Point2f pt1, pt2;
	float d = profLen1d/2 + 1.1;		// Distance from the current point, Determined by profLength, add Scalar to ensure proper profile length
	if(abs(m) > .59 && abs(m) < 1.66)   // Fixes the problem when the slope is close to 1 (Arbitrarily tested values)
		d = profLen1d/2 + 2.0;			// Causing the profiles to be less than the desired length

	pt1.x = x - (d / sqrt(1 + (m*m)));  // solve for x1
	pt1.y = m * (pt1.x - x) + y;		// solve for y1
	pt2.x = x + (d / sqrt(1+(m*m)));    // solve for x2
	pt2.y = m*(pt2.x-x)+y;				// solve for y2

	/*
	Mat outimg;
	cvtColor(imageData,outimg,CV_GRAY2BGR);
	line(outimg,pt1,pt2,CV_RGB(255, 0, 0),1,8);
	Point2f ppt = Point(x,y);
	circle(outimg, ppt, 1, CV_RGB(0, 0, 255));
	imshow("profiles2", outimg);
	cvWaitKey();
	*/
	// Use a LineIterator to access the pixel coordinates along the line
	// 8 connected graph (Bresenham's algorithm)
	// If there are more than the desired pixels found, we just ignore the extras
	// We add 1 to the profLength to get the gradient for the first position in the profile
	// LineIterator will not break if it hits the boundary of the image,
	//	if that is the case, we must account for additional pixels for the line
	const int connectivity = 8; // 8 or 4 connectivity for the line
	LineIterator lineIt(imageData,pt1,pt2,connectivity);

	int pixel = 0;
	for(int j = 0; j < lineIt.count; j++, ++lineIt){
		Point2f tempPoint = lineIt.pos();
		// Handle pixels on the edge of the image boundaries (Shouldn't ever happen but just in case, we don't want to go out of bounds)
		if(tempPoint.x < imageData.cols  || tempPoint.y < imageData.rows ){  // if this happens, we'll fix it in compute1dGradient
			pixel = imageData.at<uchar>(tempPoint.y,tempPoint.x);		// extract the pixel at the current location along the profile
			profilePoints.push_back(pixel);
		}		
	}

	return profilePoints;
}

//-- Solves the slope of the current point to determine the profile direction.
//-- Uses the parts file information to determine the correct direction.
float Profile1D::findSlopeOfProfile(const vector<Point2f> &m_Point, vector<Parts> m_parts, const int currentPoint){
	
	// Find which part the current point is in
	int pidx = -1;
	for(int i = 0; i < m_parts.size(); i++){
		for(int j = 0; j < m_parts[i].partPts.size(); j++){
			if(m_parts[i].partPts[j] == currentPoint){
				pidx = i; // we found which part we're in, break out of the loop
				break;
			}
		}
		if(pidx!=-1)
			break;
	}
	if(pidx == -1){ // This should never happen... but just in case
		cerr << "ERROR, Point " << currentPoint << " was not located in any Part" << endl;
		return FAILURE;
	}

	int	lastPoint = m_parts[pidx].partPts[m_parts[pidx].partPts.size() -1]; // Get the last point of this part
	int firstPoint = m_parts[pidx].partPts[0];								// Get the first point of this part
	float x = m_Point[currentPoint].x ;				// Get the coordinates of the current point
	float y = m_Point[currentPoint].y ;				//
		
	// find the profile direction 
	float m = 0.0;		// slope

	// Solve for the slope  y = mx + b
	if(m_parts[pidx].getPartBoundary()){		// Closed Boundary
		if(currentPoint == 0)					// First Point?					
			m = (m_Point[lastPoint].y - m_Point[currentPoint+1].y) / (m_Point[lastPoint].x - m_Point[currentPoint+1].x);  // Get the slope of the next point and the last point of the part
		else if(currentPoint == lastPoint)	    // Last Point?
			m = (m_Point[currentPoint-1].y - m_Point[firstPoint].y) / (m_Point[currentPoint-1].x - m_Point[firstPoint].x); // Get the slope of the previous point and the first point of the part
		else									// All other points
			m = (m_Point[currentPoint-1].y - m_Point[currentPoint+1].y) / (m_Point[currentPoint-1].x - m_Point[currentPoint+1].x); // Get the slope of the previous point and the next point
	}
	else{										// Open Boundary
		if(currentPoint == 0)					// First Point?
			if(currentPoint == lastPoint)		// Only Point of part? (Usually nose tip or eye center)
				m = 0.0001;		// Assign a slope close to 0.0, will cause a horizontal profile ( will get inverted to vertical when we take the orthogonal slope below)
			else
				m =  (y - m_Point[currentPoint+1].y) / (x - m_Point[currentPoint+1].x); // Get the slope of the current point and the next point
		else if(currentPoint == lastPoint)	    // Last Point?
			m = (m_Point[currentPoint-1].y - y) / (m_Point[currentPoint-1].x - x); // Get the slope of the current point and the previous point
		else									// All other points
			m = (m_Point[currentPoint-1].y - m_Point[currentPoint+1].y) / (m_Point[currentPoint-1].x - m_Point[currentPoint+1].x); // Get the slope of the previous point and the next point
	}
	
	if(m != 0.0)				// don't divide by zero
		m = -1 / m;				// take the orthogonal slope
	else						// if the slope was 0, invert it to near vertical (100)
		m = 100;
	if(abs(m) < 1.0e-4)			// check for extreme values
		m = 0.0;

	return m; // return the orthogonal slope
}

//-- This function finds the pixels along the current point's profile which we will search against,
//-- The offsets are every pixel along the profile
vector<Point2f> Profile1D::findProfileOffsets(const float m, const int x, const int y, Mat imageData, const int profLen1d)
{
	const int border = Constants::instance()->getBorder();
	vector<Point2f> outPoints;
	Point2f pt1, pt2;
	float d = profLen1d/2 + 1.1;		// Distance from the current point, Determined by profLength, add Scalar to ensure proper profile length

	if(abs(m) > .59 && abs(m) < 1.66)   // Fixes the problem when the slope is close to 1
		d = profLen1d/2 + 2.0;			// Causing the profiles to be less than the desired length

	pt1.x = x - (d / sqrt(1 + (m*m)));  // solve for x
	pt1.y = m * (pt1.x - x) + y;		// solve for y
	pt2.x = x + (d / sqrt(1+(m*m)));    // solve for x	
	pt2.y = m*(pt2.x-x)+y;				// solve for y
	
	// Use a line iterator to access the pixel coordinates along the line
	// 8 connected graph (Bresenham's algorithm)
	// If there are more than the desired pixels found, we just ignore the extras
	// Unfortunately this does not ensure that the current pixel is centered in the profile,
	//  but a difference of a single pixel is negligible
	/*
	Mat outimg;
	cvtColor(imgBordBuf,outimg,CV_GRAY2BGR);
	line(outimg,pt1,pt2,CV_RGB(255, 0, 0),1,8);
	Point2f ppt = Point(x,y);
	circle(outimg, ppt, 1, CV_RGB(0, 0, 255));
	imshow("profiles2", outimg);
	cvWaitKey();
	*/
	const int connectivity = 8; // 8 or 4 connectivity for the line
	LineIterator lineIt(imageData,pt1,pt2,connectivity);
	Point2f p;
	// sample the entire profile
	for(int j = 0; j < profLen1d; j++, ++lineIt){
		
		if(lineIt.pos().x >= 1 && lineIt.pos().x < imageData.cols - border
			&& lineIt.pos().y >= 1 && lineIt.pos().y < imageData.rows - border) // Bounds check
		{
				p.x = lineIt.pos().x;
				p.y = lineIt.pos().y;
				outPoints.push_back(p);
		}
	}
	
	return outPoints;
}

//-- Align the Model's average shape to this shapes detection rectangle (if it exists)
int Shape::initializeShape(const int detRefWidth, const Shape modelShape, const float transX, const float transY)
{
	// It is possible that the face detector failed and no face was detected, in which case return with FAILURE
	if(faceROI.width > 0)
	{
		this->m_Point = modelShape.m_Point;	// Assign the Avg model points to the current shape
		float cx, cy;
		CvRect rct = this->faceROI;

		// Reduce the size of the detected rectangle to ensure the points start within the face and work outward
		// This showed to produce better results in most cases but not all.
 		rct.width *= 0.95; // Reduce the box by 5%
 		rct.height *= 0.95;
 		rct.x += rct.x * 0.05;
 		rct.y += rct.y * 0.05;

		cx = rct.x + rct.width*0.5;  // center coordinates of the bounding rectangle
		cy = rct.y + rct.height*0.5;

		float adjustedX = transX * rct.width;
		float adjustedY = transY * rct.width;

#ifdef DEBUG
		Mat testImg;
		cvtColor(getImgbyRef(),testImg,CV_GRAY2BGR);
#endif

		this->scalePts((float)rct.width/(detRefWidth),(float)rct.height/(detRefWidth));   // Scale the points according to the rectangle size

		this->translation(cx - adjustedX ,cy - adjustedY);	// Translate points from the origin to the rectangle center

#ifdef DEBUG		
		rectangle(testImg,rct,Scalar( 0, 0, 255 ),1,8,0);
		for(int i = 0; i < nPoints; i++){
			Point2f ppt = Point(m_Point[i].x, m_Point[i].y);
			circle(testImg, ppt, 1, CV_RGB(0, 255, 0));
		}
		
		imshow("Inital Points",testImg);
		cvWaitKey();
#endif		
		orig_Point = m_Point; // get a backup copy of the points
		return SUCCESS;
	}
	else{
#pragma omp critical
		{
			cout << "Face was not detected for image: " << filename << "  No landmarks will be produced" << endl;
		}
		
		return FAILURE;
	}	
}