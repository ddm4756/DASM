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
	Train.cpp
	This file is the driving function for training the ASM
	It handles all of the functionality to load the necessary files, train the profiles, align the shapes, compute the PCA,
		and compute the bounding box coordinates. 

*/

int trainASM(path inputDir, path outputDir, path modelDir, path partsPath, path detectorPath){

	const Size dsize = Constants::instance()->getSize();	// User-defined image scale size

	Model model(Constants::instance()->getNum1DLevels(), Constants::instance()->getNum2DLevels(), Constants::instance()->getProfLength1d(), Constants::instance()->getProfLength2d());  // Implicitly calls the default constructor
	
	vector<Parts> allParts;		// Vector to hold the Parts information
	if(Constants::instance()->isVerbose())
		cout << "Loading parts file... ";
	if(Parts::loadPartsFile(allParts, partsPath) == 0){		// Load the parts file
		cerr << "Load Parts File Error...";
		return FAILURE;
	}
	model.setModelParts(allParts);
	if(Constants::instance()->isVerbose())
		cout << "done\n";

	vector<Shape> allShapes;
	if(Constants::instance()->isVerbose())
		cout << "Loading directory of images... ";
	if(Shape::loadDirectory(allShapes, inputDir) == 0){	// Load all points into allShapes vector, also computes the profiles for each image
		cerr << "Load Image/Pts Directory Error...";
		return FAILURE;
	}

	int nSamples = allShapes.size();
	if(Constants::instance()->isVerbose())
		cout << "done, Loaded " << nSamples << " images\n";
	model.setNPoints(allShapes[0].getNPoints());
	
	Shape::preScalePtsAndMat(allShapes, dsize, nSamples); // scale all images and points to preset size
	if(Constants::instance()->isVerbose()){
		cout << "All images and points pre-scaled or padded to " << dsize.width << "x" << dsize.height << endl;
		cout << "Training all profiles... ";
	}
	Shape::trainAllProfiles(allShapes, model.getModelParts(), dsize);	// Train all of the profiles for each image
	if(Constants::instance()->isVerbose())
		cout << "done\n";

	vector< vector <Profile1D> > allMean1DProfiles;		// Profile and Covariance data structures that will get stored to the Model
	vector< vector <Profile2D> > allMean2DProfiles;		//
	vector< vector <Mat> > all1DCovarMatrices;			//
	vector< vector <Mat> > all2DCovarMatrices;			//
	if(Constants::instance()->isVerbose())
		cout << "Training all Covariance Matrices and Mean Profiles... ";
	Shape::trainAll1DCovars(allShapes, allMean1DProfiles, all1DCovarMatrices, nSamples);		// Train all of the covariance matrices for each level of resolution 
	Shape::trainAll2DCovars(allShapes, allMean2DProfiles, all2DCovarMatrices, nSamples);		// Also finds the mean profiles for each level
	if(Constants::instance()->isVerbose())
		cout << "done\n";

	// Set the model objects
	model.setMean1DProfs(allMean1DProfiles);
	model.setMean2DProfs(allMean2DProfiles);
	model.set1DCovars(all1DCovarMatrices);
	model.set2DCovars(all2DCovarMatrices);

	Shape avgShape;
	if(Constants::instance()->isVerbose())
		cout << "Aligning all Shapes... \n";
	Shape::alignShapes(allShapes,avgShape);	 // Align all shapes 
	if(Constants::instance()->isVerbose())
		cout << "done\n";
	
	model.setAvgShape(avgShape);

	Mat eigenValues;
	Mat eigenVectors;

	if(Constants::instance()->isVerbose())
		cout << "Computing PCA... \n";
	Shape::computePCA(allShapes, eigenValues, eigenVectors);	// Compute the PCA
	if(Constants::instance()->isVerbose())
		cout << "done\n";

	model.setEigenValues(eigenValues);
	model.setEigenVectors(eigenVectors);

	Detector *det;
	if(Constants::instance()->getDetector_t() == Constants::VJ){
		VJ_Detector VJDet;
		VJDet.face_cascade = new CascadeClassifier();
		det = &VJDet;
		det->init(detectorPath);
	}
#ifdef WITH_PITTPATT
	else if(Constants::instance()->getDetector_t() == Constants::PP){
		PP_Detector PPDet;
		det = &PPDet;
		det->init(detectorPath);
	}
#endif
	if(Constants::instance()->isVerbose())
		cout << "Computing Face Detection... \n";
	if(det->detectAllImages(allShapes) == 0){
		cerr << "Face Detection Function Error... Program Exiting";
		return FAILURE;
	}
	if(Constants::instance()->isVerbose())
		cout << "done\n";

	Shape avgDetectorShape;
	float transX, transY;
	int ref_width = Shape::alignShapesFromDetector(allShapes,avgDetectorShape, transX, transY);
	// Assign the detector average shape to the model
	model.setAvgDetectorShape(avgDetectorShape);
	model.setRefWidth(ref_width);
	model.setDetTranslate(transX, transY);

	// Training Complete
	// Write ASM to binary/text file
	if(Constants::instance()->isVerbose())
		cout << "Writing Model to file... \n\n";
#ifdef _MSC_VER
	if(model.writeASMtoTxtFile(modelDir) == 0){
		cerr << "Failed to write ASM to specified .txt file!!!" << endl;
	}
#endif
	if(model.writeASMtoBinFile(modelDir) == 0){
		cerr << "Failed to write ASM to specified file!!!" << endl;
		return FAILURE;
	}

	eigenValues.release();
	eigenVectors.release();


	if(Constants::instance()->isVerbose())
		cout << "Finished!" << endl;
	return SUCCESS;
}

