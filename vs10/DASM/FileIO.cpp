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
	FileIO.cpp
	This file contains all of the loading functionality for the program
*/

/*	During training, this function is used to load the supplied directory of images and points.
	They must be located in the same directory.
	The image data and associated points are compiled into a vector of Shape objects associated with
		each image/pts file.
*/
int Shape::loadDirectory(vector<Shape> &allShapes,  path& directoryName){ 

	if(!exists(directoryName)){
		std::cout << "Path does not exist" << endl;
		return FAILURE;
	}
	if(is_directory(directoryName)){		// path is a directory

		directory_iterator _end_itr;
		directory_iterator _itr;
		string extensions[6] = {".jpg",".bmp",".JPG",".BMP",".tif",".TIF"};
		int numOfExtensions = extensions->length();
		string line; int npoints;

		for( _itr = directory_iterator(directoryName); _itr != _end_itr ; ++_itr ) {

			if(is_directory(*_itr))	// if there is a subdirectory, skip it
				continue;
			if (exists(*_itr)){		// if it found a file, let's try to load it
				if(extension(_itr->path()) == ".pts"){ // if pts file, continue
					Point2f temppoint;
					Shape tempShape;
					Mat img;
					path imgName = _itr->path();
					bool flag = true;
					for(int i = 0; i < numOfExtensions; i++){	
						imgName.replace_extension(extensions[i]);// test each extension
						img = imread(imgName.string(), CV_LOAD_IMAGE_GRAYSCALE);	// force image to gray scale
						if(!img.empty()){
							tempShape.imageData = img;				// Assign the image to the current shape
							tempShape.origImgData = img;			// Assign the backup
							break;
						}
						if(i==numOfExtensions-1){									// no image found -> skip pts file
							std::cout << "No image associated with pts file: " << _itr->path().string() << endl;
							flag = false;
						}
					}
					if(flag){		// if image was found continue			
						ifstream fileStream(_itr->path().string());			// open the file stream
						if( fileStream.is_open() ){					// Parse the entire file						
							std::getline(fileStream,line);				// version: 1
							fileStream >> line;						// n_points: 
							fileStream >> npoints;					// gets the number of points
							tempShape.nPoints = npoints;
							std::getline(fileStream,line);				// {
							std::getline(fileStream,line);				// skip

							for(int i = 0; i < npoints; i++){		// get the points
								fileStream >> temppoint.x;			// x
								fileStream >> temppoint.y;			// y
								tempShape.m_Point.push_back(temppoint);
							}

							float sumX = 0, sumY = 0;
							for(int i = 0; i < npoints; i++){		// find mean X and Y for the shape
								sumX += tempShape.m_Point[i].x;
								sumY += tempShape.m_Point[i].y;
							}
							tempShape.meanX = sumX / (float)npoints;	// store the mean shape values
							tempShape.meanY = sumY / (float)npoints;	//
							tempShape.orig_Point = tempShape.m_Point;
							allShapes.push_back(tempShape);			// add shape to vector

						}					
						else{
							std::cout << "Error opening the points file: " << _itr->path().string() << endl;
							return FAILURE;
						}
						fileStream.close();		// cleanup						
					}
					img.release();			// cleanup
				}
			}
		}
	}
	return SUCCESS;
}

int Shape::writePointsFile(path outDirectory){

	// Points are written out using the same exact format as STASM for consistency
	path _filename = filename;
	_filename.replace_extension(".pts");
	outDirectory /= _filename;
	//string test = outDirectory.string();
	//cout << test << endl;
	ofstream fileStream(outDirectory.string(), ios::out);
	if( fileStream.is_open() ){

		fileStream << "version: 1" << endl;
		fileStream << "n_points: " << nPoints << endl;
		fileStream << "{" << endl;
		for(int i = 0; i < nPoints; i++){
			fileStream << m_Point[i].x << " " << m_Point[i].y << endl;
		}
		fileStream << "}" << endl;
		fileStream.close();
		return SUCCESS;
	}
	return FAILURE;
}

int Parts::loadPartsFile(vector<Parts> &allParts, path& partsFileName){

	if(!exists(partsFileName)){
		std::cout << "Path does not exist" << endl;
		return FAILURE;
	}

	string line;
	if(extension(partsFileName) == ".parts"){
		ifstream fileStream(partsFileName.string());			// open the file stream
		if( fileStream.is_open() ){					// Parse the entire file
			while(!fileStream.eof()){				// Continue until the end of file
				std::getline(fileStream,line);		// "part "..." {"
				if(line.empty())					// break if there are trailing new line characters at the eof
					break;
				fileStream >> line;					// "indices(0,1,2,3 ... )"
				line = line.substr(line.find("(")+1);
				stringstream ss(line);
				
				Parts tempPart;				
				int i;
				while(ss >> i){						// parse through the line to get the points
					tempPart.partPts.push_back(i);
					if(ss.peek() == ',')
						ss.ignore();
				}					
				std::getline(fileStream,line);			// ""		
				std::getline(fileStream,line);			// "form open/closed_boundary"
				size_t found = line.find("open");   // Open or Closed Boundary?
				if(found!=std::string::npos)
					tempPart.isClosedBoundary = false;
				else
					tempPart.isClosedBoundary = true;

				allParts.push_back(tempPart);		// Add part to the allParts vector

				std::getline(fileStream,line);			// "jagged"
				std::getline(fileStream,line);			// "point_density 1"
				std::getline(fileStream,line);			// "}"
			}
			fileStream.close();
		}
	}
	else{
		cout << "Error: Parts File not Readable" << endl;
		return FAILURE;
	}
	return SUCCESS;
}

//-- Function to load all the images to be searched for landmarks
int Model::loadImagesForSearch(path& pathName, vector<Shape> &toBeSearched){

	const Size dsize = Constants::instance()->getSize();	// Preset image scale size

	if(!exists(pathName)){
		cerr << "Path does not exist" << endl;
		return FAILURE;
	}
	if(is_directory(pathName)){		// path is a directory

		directory_iterator _end_itr;
		if(Constants::instance()->isVerbose())
			std::cout << "Image Directory: " << pathName.string() << endl;

		for(directory_iterator _itr = directory_iterator(pathName); _itr != _end_itr ; ++_itr ) {
			Mat image;
			Shape shape;
			if(is_directory(*_itr))	// if there is a subdirectory, skip it
				continue;
			if (exists(*_itr))		// if it found a file, let's try to load it
				image = imread(_itr->path().string(),CV_LOAD_IMAGE_GRAYSCALE);
			if(!image.empty()){		// if it was an image and was loaded successfully, let's add it to our vector

				// TODO:: Scale all input images to a standard size (this produces the optimal results)
				//			Keep in mind that you will have to scale the final points back to the original size before 
				//			saving out the points to file

				shape.setImgData(image);	// set the image data
				shape.setOrigImgData(image); // set up the backup image copy
				shape.setFilename(_itr->path().filename().string());

				toBeSearched.push_back(shape);
			}		
		}
	}
	else{	// path is a single image

		Shape shape;
		if(Constants::instance()->isVerbose())
			cout << pathName.string();
		Mat image = imread(pathName.string(),CV_LOAD_IMAGE_GRAYSCALE);
		if(image.empty()){
			cerr << "Not a valid image format" << endl;
			return FAILURE;
		}

		shape.setImgData(image);	// set the image data
		shape.setOrigImgData(image); // set up the backup image copy
		shape.setFilename(pathName.filename().string());

		toBeSearched.push_back(shape);
	}
	return SUCCESS;
}

//-- Function to load the model from a text file
int Model::loadASMfromTxtFile(path& inFileName){

	ifstream fileStream(inFileName.string(), ios::in );

	if( fileStream.is_open() ){
		string line;
		int val;
		getline(fileStream,line);		// "DASM Model File"
		getline(fileStream,line);		// ""
		getline(fileStream,line);		// ".. --Model Attributes--.."
		getline(fileStream,line);		// "{"
		fileStream >> line;				// "nPoints"	
		fileStream >> this->nPoints;	// 252 / 69 / etc.
		fileStream >> line;				// "nLevels1dProfile"
		fileStream >> this->nLevels1D;	// 4 / 2 / etc.
		fileStream >> line;				// "nLevels2dProfile"
		fileStream >> this->nLevels2D;	// 4 / 2 / etc.
		fileStream >> line;				// "1dProfileLength"
		fileStream >> this->ProfLen1d;	// 9 / 11 / etc.
		fileStream >> line;				// "2dProfileLength"
		fileStream >> this->ProfLen2d;	// 9 / 11 / etc.
		fileStream >> line;				// "DetectorRefWidth"
		fileStream >> this->DetRefWidth;// 200 / 100 / etc.
		fileStream >> line;				// DetectorTranslation X
		fileStream >> this->transX;		// 0.5 / etc.
		fileStream >> line;				// DetectorTranslation Y
		fileStream >> this->transY;		// 0.5 / etc.
		fileStream >> line;				// "nParts"
		fileStream >> this->nParts;		// 36 / 12 / etc.
		getline(fileStream,line);		// "}"
		getline(fileStream,line);		// ""
		getline(fileStream,line);		// ".. --Parts Info--.."
		for(int i = 0; i < nParts; i++){
			getline(fileStream,line);			// "{ Part 0 / 1 / etc.
			getline(fileStream,line);			// 0 1 2 ...
			stringstream ss(line);
			Parts tempPart; 
			int j;
			while(ss >> j){						// parse through the line to get the points
				tempPart.partPts.push_back(j);
				if(ss.peek() == ' ')
					ss.ignore();	
			}
			fileStream >> line;			// "Boundary "
			fileStream >> line;			// "Parts "
			fileStream >> val;			// 1 / 0
			tempPart.setPartsBoundary(val);
			getline(fileStream,line);	// ""
			getline(fileStream,line);	// "}"
			ModelParts.push_back(tempPart);
		}
		getline(fileStream,line);		// ".. --Avg Procrustes Shape-- .."
		getline(fileStream,line);		// "{"
		for(int i = 0; i < nPoints; i++){
			getline(fileStream,line);
			stringstream ss(line);
			Point2f p;
			ss >> p.x;			
			ss >> p.y;
			ModelAvgShape.setPoint(p);		
		}
		getline(fileStream,line);		// "}"
		getline(fileStream,line);		// ".. --Avg Detector Shape-- .."
		getline(fileStream,line);		// "{"
		for(int i = 0; i < nPoints; i++){
			getline(fileStream,line);
			stringstream ss(line);
			Point2f p;
			ss >> p.x;			
			ss >> p.y;
			DetectorAvgShape.setPoint(p);		
		}
		getline(fileStream,line);		// "}"
		getline(fileStream,line);		// ".. --Eigen Values-- .."
		fileStream >> line;				// "{"
		int nRows, nCols;
		fileStream >> nRows;			// #rows
		getline(fileStream,line);		// ""
		getline(fileStream,line);		// "[1160.1523; ... "
		stringstream ss(line); 
		float eval;
		ss.ignore();					// "["
		while(ss >> eval){	
			eigenValues.push_back(eval);
			if(ss.peek() == ';')
				ss.ignore();	
		}
		getline(fileStream,line);		// "}"
		getline(fileStream,line);		// ".. --Eigen Vectors-- .."
		fileStream >> line;				// "{"
		fileStream >> nRows;			// get the #rows again just in case.
		fileStream >> nCols;			// #cols
		eigenVectors.create(nRows,nCols,CV_32F);
		getline(fileStream,line);		// ""
		for(int i = 0; i < nRows; i++){
			int j = 0;
			getline(fileStream,line);	
			stringstream ss(line);
			if(i==0)ss.ignore();
			while(ss >> eval){
				eigenVectors.at<float>(i,j) = eval;
				j++;
				if(ss.peek() == ',' || ss.peek() == ';')
					ss.ignore();	
			}
		}
		getline(fileStream,line);			// "}"
		getline(fileStream,line);			// ".. --Mean Profiles 1D-- .."
		ModelMean1DProfiles.resize(nLevels1D);
		for(int i = 0; i < nLevels1D; i++){
			ModelMean1DProfiles[i].resize(nPoints);
			getline(fileStream,line);		// "--! Level # .."
			for(int j = 0; j < nPoints; j++){
				getline(fileStream,line);	// "{ Pt #"
				getline(fileStream,line);	// the profile values
				ss.clear();
				ss.str(line);
				ModelMean1DProfiles[i][j].profileGradients1D.resize(ProfLen1d);
				for(int k = 0; k< ProfLen1d; k++){
					ss >> eval;
					ModelMean1DProfiles[i][j].profileGradients1D[k] = eval;
				}
				getline(fileStream,line);	// "}"
			}
		}
		getline(fileStream,line);			// ".. --Mean Profiles 2D-- .."
		ModelMean2DProfiles.resize(nLevels2D);
		for(int i = 0; i < nLevels2D; i++){
			ModelMean2DProfiles[i].resize(nPoints);
			getline(fileStream,line);		// "--! Level # .."
			for(int j = 0; j < nPoints; j++){
				getline(fileStream,line);	// "{ Pt #"
				getline(fileStream,line);	// the profile values
				ss.clear();
				ss.str(line);
				ss.ignore();				// ignore '['
				ModelMean2DProfiles[i][j].init(ProfLen2d,ProfLen2d);
				Mat temp(ProfLen2d,ProfLen2d,CV_32F);
				
				int col = 0, row = 0;
				while(ss >> eval){
					temp.at<float>(row,col) = (float)eval;
					if(ss.peek() == ','){
						ss.ignore();
						col++;
						if(col == ProfLen2d){
							col = 0;
							row++;
						}
					}
					else if(row == ProfLen2d)
						break;			
				}
				ModelMean2DProfiles[i][j].setProfile(temp);
				getline(fileStream,line);	// "}"
			}
		}
		getline(fileStream,line);			// ".. --Covariance Matrices 1D-- .."
		Model1DCovarMatrices.resize(nLevels1D);
		for(int i = 0; i < nLevels1D; i++){
			Model1DCovarMatrices[i].resize(nPoints);
			getline(fileStream,line);		// "--! Level # .."
			for(int j = 0; j < nPoints; j++){
				getline(fileStream,line);	// "{ Pt #"
				Model1DCovarMatrices[i][j].create(ProfLen1d,ProfLen1d,CV_32F);
				for(int k = 0; k < ProfLen1d; k++){
					getline(fileStream,line);	// the covar values
					ss.clear();
					ss.str(line);
					if(k==0)ss.ignore();		// ignore '['
					int col = 0;
					while(ss >> eval){
						Model1DCovarMatrices[i][j].at<float>(k,col) = eval;
						if(++col==ProfLen1d)break;
						if(ss.peek() == ',')
							ss.ignore();

					}
				}
				getline(fileStream,line);		// "}"
			}
		}
		getline(fileStream,line);				// ".. --Covariance Matrices 2D-- .."
		Model2DCovarMatrices.resize(nLevels2D);
		for(int i = 0; i < nLevels2D; i++){
			Model2DCovarMatrices[i].resize(nPoints);
			getline(fileStream,line);			// "--! Level # .."
			for(int j = 0; j < nPoints; j++){
				getline(fileStream,line);		// "{ Pt #"
				Model2DCovarMatrices[i][j].create(ProfLen2d*ProfLen2d, ProfLen2d*ProfLen2d,CV_32F);
				for(int k = 0; k < ProfLen2d*ProfLen2d; k++){
					getline(fileStream,line);	// the covar values
					ss.clear();
					ss.str(line);
					if(k==0)ss.ignore();		// ignore '['
					int col = 0;
					while(ss >> eval){
						Model2DCovarMatrices[i][j].at<float>(k,col) = eval;
						if(++col==ProfLen2d*ProfLen2d)break;
						if(ss.peek() == ',')
							ss.ignore();
					}
				}
				getline(fileStream,line);		// "}"
			}
		}
		fileStream.close();
		return SUCCESS;
	}
	else
		return FAILURE;
}

//-- Function to load the model from binary
int Model::loadASMfromBinFile(path& inFileName){

	ifstream fileStream (inFileName.string(), ios::binary);

	if(fileStream.is_open()){

		int nEigs, lenEigVects, boundary, partSize;

		fileStream.read((char*)(&nPoints),sizeof(int));		// Model Attributes
		fileStream.read((char*)(&nLevels1D), sizeof(int));
		fileStream.read((char*)(&nLevels2D), sizeof(int));
		fileStream.read((char*)(&ProfLen1d), sizeof(int));
		fileStream.read((char*)(&ProfLen2d), sizeof(int));
		fileStream.read((char*)(&DetRefWidth), sizeof(int));
		fileStream.read((char*)(&transX), sizeof(float));
		fileStream.read((char*)(&transY), sizeof(float));
		fileStream.read((char*)(&nParts), sizeof(int));
		fileStream.read((char*)(&nEigs), sizeof(int));
		fileStream.read((char*)(&lenEigVects), sizeof(int));
		
		ModelParts.resize(nParts);
		for(int i = 0; i < nParts; i++){				// Parts
			fileStream.read((char*)(&partSize),sizeof(int));
			for(int j = 0; j < partSize; j++){
				int var;
				fileStream.read((char*)(&var),sizeof(int));
				ModelParts[i].partPts.push_back(var);
			}
			fileStream.read((char*)(&boundary),sizeof(int));
			ModelParts[i].setPartsBoundary(boundary);
		}
		ModelAvgShape.setNPoints(nPoints);
		for(int i = 0; i < nPoints; i++){				// Mean Shape
			float x,y;
			fileStream.read((char*)&x, sizeof(float));
			fileStream.read((char*)&y, sizeof(float));
			Point2f point;
			point.x = x;
			point.y = y;
			ModelAvgShape.setPoint(point);
		}
		DetectorAvgShape.setNPoints(nPoints);
		for(int i = 0; i < nPoints; i++){				// Detector Avg Shape
			float x,y;
			fileStream.read((char*)&x, sizeof(float));
			fileStream.read((char*)&y, sizeof(float));
			Point2f point;
			point.x = x;
			point.y = y;
			DetectorAvgShape.setPoint(point);
		}
		eigenValues.create(nEigs,1,CV_32F);			// Eigens
		eigenVectors.create(nEigs,lenEigVects,CV_32F);
		fileStream.read((char*)eigenValues.data, nEigs*sizeof(float));
		fileStream.read((char*)eigenVectors.data, nEigs*lenEigVects*sizeof(float));

		// 1D mean profiles
		ModelMean1DProfiles.resize(nLevels1D);
		for(int i = 0; i < nLevels1D; i++){
			ModelMean1DProfiles[i].resize(nPoints);
			for(int j = 0; j < nPoints; j++){
				for(int k = 0; k < ProfLen1d; k++){
					float f;
					fileStream.read((char*)&f, sizeof(f));
					ModelMean1DProfiles[i][j].profileGradients1D.push_back(f);
				}
			}
		}

		// 2D mean profiles
		ModelMean2DProfiles.resize(nLevels2D);
		for(int i= 0; i < nLevels2D; i++){
			ModelMean2DProfiles[i].resize(nPoints);
			for(int j = 0; j < nPoints; j++){
				Mat m(ProfLen2d,ProfLen2d,CV_32FC1);
				fileStream.read((char*)m.ptr<char>(), ProfLen2d*ProfLen2d*sizeof(float));
				ModelMean2DProfiles[i][j].setProfile(m);
			}
		}

		// 1D Covariance
		Model1DCovarMatrices.resize(nLevels1D);
		for(int i = 0; i < nLevels1D; i++){
			Model1DCovarMatrices[i].resize(nPoints);
			for(int j = 0; j < nPoints; j++){
				Mat m(ProfLen1d,ProfLen1d,CV_32FC1);
				fileStream.read((char*)m.data, ProfLen1d*ProfLen1d*sizeof(float));
				m.copyTo(Model1DCovarMatrices[i][j]);
			}
		}

		// 2D Covariance
		Model2DCovarMatrices.resize(nLevels2D);
		for(int i = 0; i < nLevels2D; i++){
			Model2DCovarMatrices[i].resize(nPoints);
			for(int j = 0; j < nPoints; j++){
				Mat m(ProfLen2d*ProfLen2d,ProfLen2d*ProfLen2d,CV_32FC1);
				fileStream.read((char*)m.data, ProfLen2d*ProfLen2d*ProfLen2d*ProfLen2d*sizeof(float));
				m.copyTo(Model2DCovarMatrices[i][j]);
			}
		}

		fileStream.close();
		return SUCCESS;
	}
	else
		return FAILURE;
}

//-- Write the model out in binary format
int Model::writeASMtoBinFile(path outFileName){

	outFileName /= "asm.bin";
	ofstream fileStream (outFileName.string(), ios::binary);

	/*
		//--Model Attributes
		nPoints
		nLevels1dProfile
		nLevels2dProfile
		1dProfileLength
		2dProfileLength
		DetectorRefWidth
		DetectorTranslationX
		DetectorTranslationY
		nParts
		nEigenValues
		LengthOfEigenVectors
		Parts data
		Model Avg Shape Points x y
		Detector Avg Shape Points x y
		Eigen Values
		Eigen Vectors
		Mean Profiles Per level 1D / 2D
		Covariance Matrices Per level 1D / 2D

	*/
	if( fileStream.is_open() ){

		fileStream.write((char*)&nPoints, sizeof(int));
		fileStream.write((char*)&nLevels1D, sizeof(int));
		fileStream.write((char*)&nLevels2D, sizeof(int));
		fileStream.write((char*)&ProfLen1d, sizeof(int));
		fileStream.write((char*)&ProfLen2d, sizeof(int));
		fileStream.write((char*)&DetRefWidth, sizeof(int));
		fileStream.write((char*)&transX, sizeof(float));
		fileStream.write((char*)&transY, sizeof(float));
		fileStream.write((char*)&nParts, sizeof(int));
		fileStream.write((char*)&eigenValues.rows, sizeof(int));
		fileStream.write((char*)&eigenVectors.cols, sizeof(int));

		for(int i = 0; i < nParts; i++){			// Parts
			int size = ModelParts[i].partPts.size();
			fileStream.write((char*)&size, sizeof(int));
			for(int j = 0; j < size; j++){
				fileStream.write((char*)&ModelParts[i].partPts[j], sizeof(int));
			}
			int boundary = ModelParts[i].getPartBoundary();
			fileStream.write((char*)(&boundary), sizeof(int));
		}

		for(int i = 0; i < nPoints; i++){	// Mean Shape
			float x = ModelAvgShape.getPoint(i).x;
			float y = ModelAvgShape.getPoint(i).y;
			fileStream.write((char*)&x, sizeof(x));
			fileStream.write((char*)&y, sizeof(y));
		}
		for(int i = 0; i < nPoints; i++){	// Detector Shape
			float x = DetectorAvgShape.getPoint(i).x;
			float y = DetectorAvgShape.getPoint(i).y;
			fileStream.write((char*)&x, sizeof(x));
			fileStream.write((char*)&y, sizeof(y));
		}

		int size = eigenValues.rows;			// Eigen Values and Vectors
		fileStream.write((char*)eigenValues.data, size*sizeof(float));
		size = eigenVectors.rows;
		int length = eigenVectors.cols;
		fileStream.write((char*)eigenVectors.data, size * length * sizeof(float));

		// 1D mean profiles
		for(int i = 0; i < nLevels1D; i++){
			for(int j = 0; j < nPoints; j++){
				for(int k = 0; k < ProfLen1d; k++){
					float f = ModelMean1DProfiles[i][j].profileGradients1D[k];
					fileStream.write((char*)&f,sizeof(f));
				}
			}
		}

		// 2D mean profiles
		for(int i= 0; i < nLevels2D; i++){
			for(int j = 0; j < nPoints; j++){
				Mat m;
				ModelMean2DProfiles[i][j].getProfile().copyTo(m);
				if(m.type()!=CV_32FC1)
					m.convertTo(m,CV_32FC1);
				fileStream.write((char*)m.data, ProfLen2d*ProfLen2d*sizeof(float));
			}
		}

		// 1D Covariance
		for(int i = 0; i < nLevels1D; i++){
			for(int j = 0; j < nPoints; j++){
				Mat m;
				Model1DCovarMatrices[i][j].copyTo(m);
				if(m.type()!=CV_32FC1)
					m.convertTo(m,CV_32FC1);
				fileStream.write((char*)m.data, ProfLen1d*ProfLen1d*sizeof(float));
			}
		}
		
		// 2D Covariance
		for(int i= 0; i < nLevels2D; i++){
			for(int j = 0; j < nPoints; j++){
				Mat m;
				Model2DCovarMatrices[i][j].copyTo(m);
				if(m.type()!=CV_32FC1)
					m.convertTo(m,CV_32FC1);
				fileStream.write((char*)m.data, ProfLen2d*ProfLen2d*ProfLen2d*ProfLen2d*sizeof(float));
			}
		}
		return SUCCESS;
	}
	else
		return FAILURE;
}

//-- Write the model out in text format
int Model::writeASMtoTxtFile(path outFileName){

	outFileName /= "asm.txt";
	ofstream fileStream (outFileName.string(), ios::out);
	pt::ptime now = pt::second_clock::local_time();
	/*
		DASM Model Produced:  date/time

		Model Attributes
		{
		nPoints 
		nLevels1dProfile
		nLevels2dProfile
		1dProfileLength
		2dProfileLength
		DetectorRefWidth
		DetectorTranslationX
		DetectorTranslationY
		nParts
		Parts data
		Model Avg Shape Points x y
		Detector Avg Shape Points x y
		Eigen Values
		Eigen Vectors
		Mean Profiles Per level 1D / 2D
		Covariance Matrices Per level 1D / 2D

	*/
	if( fileStream.is_open() ){

		fileStream << "DASM Model Produced: " << now.date() << " " << now.time_of_day() << endl << endl;
		fileStream << "---------- Model Attributes ----------" << endl;
		fileStream << "{" << endl;
		fileStream << "nPoints " << nPoints << endl;
		fileStream << "nLevels1dProfile " << nLevels1D << endl;
		fileStream << "nLevels2dProfile " << nLevels2D << endl;
		fileStream << "1dProfileLength "  << ProfLen1d << endl;
		fileStream << "2dProfileLength "  << ProfLen2d << endl;
		fileStream << "DetectorRefWidth " << DetRefWidth << endl;
		fileStream << "DetectorTranslation X " << transX << endl;
		fileStream << "DetectorTranslation Y " << transY << endl;
		fileStream << "nParts " << nParts << endl;
		fileStream << "}" << endl;
		fileStream << "---------- Parts Info ---------- " << endl;
		writeParts(fileStream,ModelParts);
		fileStream << "---------- Avg Model Shape ----------" << endl << "{" << endl;
		writePoints(fileStream, ModelAvgShape);
		fileStream << "}" << endl;
		fileStream << "---------- Avg Detector Shape ----------" << endl << "{" << endl;
		writePoints(fileStream,DetectorAvgShape);
		fileStream << "}" << endl;
		fileStream << "---------- Eigen Values ----------" << endl << "{ " << eigenValues.rows << endl;
		fileStream << eigenValues << endl;
		fileStream << "}" << endl;
		fileStream << "---------- Eigen Vectors ----------" << endl << "{ " << eigenVectors.rows << " " << eigenVectors.cols << endl;
		fileStream << eigenVectors << endl;
		fileStream << "}" << endl;
		// Mean Profiles
		fileStream << "---------- Mean Profiles 1D ----------" << endl;
		for(int i = 0; i < nLevels1D; i++){
			fileStream << "--! Level " << i << " !--"<< endl;
			writeMean1dProfs(fileStream, ModelMean1DProfiles[i], i);
		}
		fileStream << "---------- Mean Profiles 2D ----------" << endl;
		for(int i = 0; i < nLevels2D; i++){
			fileStream << "--! Level " << i << " !--" <<endl;
			writeMean2dProfs(fileStream, ModelMean2DProfiles[i], i);
		}
		// Covariance matrices
		fileStream << "---------- Covariance Matrices 1D ----------" <<endl;
		for(int i = 0; i < nLevels1D; i++){
			fileStream << "--! Level " << i << " !--"<< endl;
			writeCovars(fileStream, Model1DCovarMatrices[i], i);
		}
		fileStream << "---------- Covariance Matrices 2D ----------" <<endl;
		for(int i = 0; i < nLevels2D; i++){
			fileStream << "--! Level " << i << " !--" <<endl;
			writeCovars(fileStream, Model2DCovarMatrices[i], i);
		}
		fileStream.close();
	}
	else
		return FAILURE;

	return SUCCESS;
}

void Model::writePoints(ofstream &fileStream, Shape avgShape){

	if( fileStream.is_open() ){
		for(int i = 0; i < nPoints; i++){
			fileStream << avgShape.getPoint(i).x << " " << avgShape.getPoint(i).y << endl;
		}
	}
}
void Model::writeMean1dProfs(ofstream &fileStream, vector<Profile1D> meanProfs, int lev){
	const int profLength1d = Constants::instance()->getProfLength1d();
	if( fileStream.is_open() ){
		for(int i = 0; i < nPoints; i++){
			fileStream << "{" << " Pt " << i << endl;
			for(int j = 0; j < profLength1d; j++){
				fileStream << meanProfs[i].profileGradients1D[j] << " ";
			}
			fileStream << endl << "}" << endl;
		}
	}
}
void Model::writeMean2dProfs(ofstream &fileStream, vector<Profile2D> meanProfs, int lev){
	if( fileStream.is_open() ){
		for(int i = 0; i < nPoints; i++){
			Mat m = meanProfs[i].getProfile();
			if(m.type()!=CV_32F){
				m.convertTo(m,CV_32F);
			}
			fileStream << "{" << " Pt " << i << endl;
			fileStream << m;
			fileStream << endl << "}" << endl;
		}		
	}
}
void Model::writeCovars(ofstream &fileStream, vector<Mat> covars, int lev){
	if( fileStream.is_open() ){
		for(int i = 0; i < nPoints; i++){
			if(covars[i].type()!=CV_32F)
				covars[i].convertTo(covars[i],CV_32F);
			fileStream << "{" << " Pt " << i << endl;
			fileStream << covars[i];
			fileStream << endl << "}" << endl;
		}
	}
}
void Model::writeParts(ofstream &fileStream, vector<Parts> parts){
	if( fileStream.is_open() ){
		for(int i = 0; i < parts.size(); i++){
			fileStream << "{" << " Part " << i << endl;
			for(int j = 0; j < parts[i].partPts.size(); j++){
				fileStream << parts[i].partPts[j] << " ";
			}
			fileStream << endl << "Closed Boundary " << parts[i].getPartBoundary() << endl << "}" << endl;
		}
	}
}



