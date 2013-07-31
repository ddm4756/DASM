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
	PCA.cpp
	This file has one function which computes the PCA on the aligned shapes.

*/

void Shape::computePCA(vector<Shape> &alignedSamples, Mat &eigenvalues, Mat &eigenvectors){
	float start_time = omp_get_wtime();  //record start time

	const float retainPercent = .98;

	int nSamples = alignedSamples.size();
	int nPoints = alignedSamples[0].m_Point.size();
	int maxComponents = nSamples-1;
	// convert the samples vector to a Mat for PCA processing
	Mat pcaset(nSamples, nPoints*2, CV_64FC1);	// required to be 64 bit precision for training

	for(int i = 0; i < nSamples; i++){
		for(int j = 0; j < nPoints; j++){
			pcaset.at<double>(i, j*2   ) = alignedSamples[i].m_Point[j].x;
			pcaset.at<double>(i, j*2 +1) = alignedSamples[i].m_Point[j].y;
		}
	}
	// Construct the PCA object from the projection,
	// Has to use double precision, float will break openCV
	PCA pca(pcaset, // data as const Mat&
		Mat(),  // mean vector as const Mat&
		CV_PCA_DATA_AS_ROW,   // the data is stored in row-major order
		maxComponents // number of PC's to retain
		);

	float end_time = omp_get_wtime(); // record end time
	if(Constants::instance()->isVerbose())
		cout << "Total Time for PCA: " << end_time - start_time << endl;
	
	Scalar valSum = sum(pca.eigenvalues);
	valSum.val[0] *= retainPercent; // amount of Eigen values to retain

	int idx = 0;
	while (valSum.val[0] > 0)
	{
		valSum.val[0] -= pca.eigenvalues.at<double>(idx);
		idx++;
	}
	// Convert pca data from 64 bit to 32 bit to speed up processing
	eigenvalues = pca.eigenvalues.rowRange(0,idx);
	eigenvectors = pca.eigenvectors.rowRange(0, idx);
	eigenvalues.convertTo(eigenvalues,CV_32FC1);
	eigenvectors.convertTo(eigenvectors,CV_32FC1);

	if(Constants::instance()->isVerbose())
		cout << "Retaining " << retainPercent*100 << "% = " << idx << " values out of maximum possible " << maxComponents << endl;

	pcaset.release();
}
