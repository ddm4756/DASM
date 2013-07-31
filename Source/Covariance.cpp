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
	Covariance.cpp
	This file contains all of the functionality to compute the covariance matrices for both
		the 1d and 2d profiles. It uses the OpenCV calcCovarMatrix, which also computes
		the mean profiles for the 1d and 2d 

*/

//-- Function handles the training of all covariance matrices associated with 1D profiles
void Shape::trainAll1DCovars(vector<Shape> &allShapes,vector< vector<Profile1D> > &allMean1DProfiles,vector< vector<Mat> > &all1DCovarMatrices, int nSamples){

	const int nPoints = allShapes[0].m_Point.size();
	const int num1DLevels = Constants::instance()->getNum1DLevels();
	allMean1DProfiles.resize(num1DLevels);				// We have mean profiles and covars for each resolution level
	all1DCovarMatrices.resize(num1DLevels);

	for(int i = 0; i < num1DLevels; i++){				// For each level of the resolution change
		allMean1DProfiles[i].resize(nPoints);			// Each level has a vector of mean profiles (1 mean profile per point)
		all1DCovarMatrices[i].resize(nPoints);			// Each level has a vector of covariances Matrix (1 covar matrix per point)
		calc1DProfileCovar(allShapes,allMean1DProfiles[i],all1DCovarMatrices[i],nSamples,i);
	}
	// TRIM COVARIANCE HERE!
}

//-- Function handles the training of all covariance matrices associated with 2D profiles
void Shape::trainAll2DCovars(vector<Shape> &allShapes,vector< vector<Profile2D> > &allMean2DProfiles,vector< vector<Mat> > &all2DCovarMatrices, int nSamples){

	const int nPoints = allShapes[0].m_Point.size();
	const int num2DLevels = Constants::instance()->getNum2DLevels();
	allMean2DProfiles.resize(num2DLevels);				// We have mean profiles and covars for each resolution level
	all2DCovarMatrices.resize(num2DLevels);

	for(int i = 0; i < num2DLevels; i++){				// For each level of the resolution change
		allMean2DProfiles[i].resize(nPoints);			// Each level has a vector of mean profiles (1 mean profile per point)
		all2DCovarMatrices[i].resize(nPoints);			// Each level has a vector of covariances Matrix (1 covar matrix per point)
		calc2DProfileCovar(allShapes,allMean2DProfiles[i],all2DCovarMatrices[i],nSamples,i);
	}
	// TRIM COVARIANCE HERE!
}

//-- Calculate the covariance matrices and  mean profiles for each point
//-- The inverse of the covariance matrices is taken to reduce computation on the search side (needed for the Mahalanobis calculation)
void Shape::calc1DProfileCovar(vector<Shape> &allShapes, vector<Profile1D> &mean_Profiles, vector<Mat> &covarMatrix, int nSamples, int curLevel){
	
	const int profLength1d = Constants::instance()->getProfLength1d();
	const int nPoints = allShapes[0].m_Point.size();
#pragma omp parallel for
	for(int j = 0; j < nPoints; j++){							
		covarMatrix[j].create(profLength1d,profLength1d,CV_32FC1);	// Set the size of each covariance matrix according to profLength x profLength
	}

	vector<Mat> samples;
	samples.resize(nPoints);

	for(int i = 0; i < nPoints; i++){			// Create the covariance for each point of the scheme

		samples[i] = Mat(nSamples,profLength1d,CV_32F);				// Create Mat for each Point, will contain all the samples profiles
		for(int j = 0; j < nSamples; j++){
			for(int k = 0; k < profLength1d; k++)	// Can I do this assignment row at a time?
				samples[i].at<float>(j,k) = allShapes[j].allLev1dProfiles[curLevel][i].profileGradients1D[k];	// Assign the data to the Mat from each profile

		}
		Mat cov, meanVector;
		calcCovarMatrix(samples[i],cov,meanVector, CV_COVAR_ROWS | CV_COVAR_NORMAL, CV_32FC1);	// OpenCV Function which calculates the covariance matrices and mean vectors
		// take the inverse covariance matrix
		Mat icovar;
		invert(cov,icovar); // NOTE that we only store the inverse covariance matrix, as the standard covariance is not needed in search
		icovar.copyTo(covarMatrix[i]);	 // Store the data
		mean_Profiles[i].profileGradients1D = meanVector;			// Store the data
		
	}

}

//-- Calculate the covariance matrices and  mean profiles for each point
//-- The inverse of the covariance matrices is taken to reduce computation on the search side (needed for the Mahalanobis calculation)
void Shape::calc2DProfileCovar(vector<Shape> &allShapes, vector<Profile2D> &mean_Profiles, vector<Mat> &covarMatrix, int nSamples, int curLevel){

	const int nPoints = allShapes[0].m_Point.size();
	const int profLength2d = Constants::instance()->getProfLength2d();

#pragma omp parallel for
	for(int i = 0; i < nPoints; i++){
		covarMatrix[i].create(profLength2d*profLength2d,profLength2d*profLength2d,CV_32FC1);
	}
	// From Milborrow 5.5.1: The covariance matrix of a set of 2D profiles is formed by treating each 2D
	//		profile matrix as a long vector (by appending the rows end to end), and calculating the
	//		covariance of the vectors.

	Mat cov, meanVector, icovar, flattened; // calc covar over samples[i]
	vector<Mat> samples;
	samples.resize(nPoints);

	for(int i = 0; i < nPoints; i++){			// Create the covariance for each point of the scheme
		for(int j = 0; j < nSamples; j++)
		{
			// flatten the 2D profile
			flattened = allShapes[j].allLev2dProfiles[curLevel][i].getProfile().reshape(0,1);
			// each iteration of samples vector has a mat of size nSamples, which contains the flattened 2d profiles for that point across each sample
			samples[i].push_back(flattened);
		}

		// Calculate the covariance matrix and the mean profile vector for this point
		calcCovarMatrix(samples[i],cov,meanVector, CV_COVAR_ROWS | CV_COVAR_NORMAL);
		mean_Profiles[i].setProfile(meanVector);

		// NOTE that we only store the inverse covariance matrix, as the standard covariance is not needed in search
		invert(cov,icovar);
		icovar.copyTo(covarMatrix[i]);
	}	
}