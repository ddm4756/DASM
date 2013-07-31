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

#include "main.h"

Constants *Constants::m_pInstance = 0;

//-- Entry Point
int main(int argc, char* argv[]){

	// Boost is used to parse the command line and config file
	options_description desc("Command Line Options");
	desc.add_options()("help,h", "Generate help message")
					("version,v", "Display the application version")
					("config,c", value<string>(), "Configuration File")
					("verbose,b", "Verbose Mode")
					("omp", value<int>()->default_value(1), "Enable OpenMP Support")
					;


	options_description cfgDesc("Config Options");
	cfgDesc.add_options()
					("DASM_Config.train", value<string>(),		"<Train>")
					("DASM_Config.search", value<string>(),		"<Search>")
					("DASM_Config.input-dir", value<string>(),	"<Train/Search> Input Directory Path")
					("DASM_Config.detector-vj", value<string>(), "<Train/Search> Path to VJ Object Detector Cascade File")
					("DASM_Config.detector-pp", value<string>(), "<Train/Search> Path to PittPatt Object Detector")
					("DASM_Config.model-dir", value<string>(),	"<Train> Output Directory Path for Model Files")
					("DASM_Config.parts-path", value<string>(), "<Train> Path to the Parts File")
					("DASM_Config.profLength1d", value<int>(),	"<Train> 1D Profile Length")
					("DASM_Config.profLength2d", value<int>(),	"<Train> 2D Profile Length")
					("DASM_Config.num1DLevels", value<int>(),	"<Train> Number of Pyramid Levels for 1D profiles")
					("DASM_Config.num2DLevels", value<int>(),	"<Train> Number of Pyramid Levels for 2D profiles")
					("DASM_Config.out-dir", value<string>(),	"<Search> Output Directory Path for Points")
					("DASM_Config.model-path", value<string>(), "<Search> Path to the Model File")
					("DASM_Config.eigPercent", value<int>(),	"<Search> % (1-100) of the energy to use (determines the # of eigenvectors)")
					("DASM_Config.searchParam", value<int>(),	"<Search> Search Parameter (1 = 1d profiles only, 2 = 2d and 1d profiles)")
					("DASM_Config.stacked", value<int>(),		"<Search> Stack the model to perform a second pass on the images")
					("DASM_Config.stdWidth", value<int>(),		"<Train> Standard Width of training images")
					("DASM_Config.stdHeight", value<int>(),		"<Train> Standard Height of training images")
					;

	options_description config_file_options;
	config_file_options.add(desc).add(cfgDesc);
	
	variables_map vm;
	try
	{
		store(parse_command_line(argc,argv,config_file_options), vm);
	}
	catch (boost::exception& e)
	{
		cout << boost::diagnostic_information(e) << endl << endl;
		cout << desc << endl;
		return FAILURE;
	}
	notify(vm);

	if(vm.count("help"))
	{
		cout << desc << endl;
		return SUCCESS;
	}
	if(vm.count("version"))
	{
		cout << "DASM v" << VERSION << endl;
		cout << "Author: David Macurak" << endl;
		return SUCCESS;
	}
	if(vm.count("verbose"))
	{
		Constants::instance()->setVb();
	}

	if(vm.count("omp")){ // Enable OpenMP support

		int num_cores = vm["omp"].as<int>();
		omp_set_num_threads(num_cores);	// Set the number of OpenMP threads (determined by the number of cores on the machine)
		int tid;
#pragma omp parallel private(tid) // Display to user how many threads were created
		{
			tid = omp_get_thread_num();
			if(tid==0 && Constants::instance()->isVerbose())
				cout << "OpenMP Enabled: " << omp_get_num_threads() << " threads spun up" << endl;
		}
	}
	else
		omp_set_num_threads(1);
	// Read the Config file
	ifstream s(vm["config"].as<string>());
	if(!s)
	{
		cerr<<"Failed to open Configuration File"<< endl;
		return FAILURE;
	}
	store(parse_config_file(s,cfgDesc),vm);
	notify(vm);

	path inputDir, modelsDir, outputDir, detectorPath, partsPath;
	pt::ptime now = pt::second_clock::local_time();

	// Train Option
	if(vm.count("DASM_Config.train") && (vm["DASM_Config.train"].as<string>().compare("true")) == 0 )
	{
		if(vm.count("DASM_Config.model-dir") && vm.count("DASM_Config.input-dir") && (vm.count("DASM_Config.detector-vj") || vm.count("DASM_Config.detector-pp")) && vm.count("DASM_Config.parts-path")){ // Required arguments

			inputDir = path(vm["DASM_Config.input-dir"].as<string>()); // User Input
			if(!is_directory(inputDir)){
				cerr << inputDir.string() << " is invalid Points/Image directory.";
				return FAILURE;
			}
			if(Constants::instance()->isVerbose())
				cout << "Images/Points Directory: " << inputDir.string() << endl;

			modelsDir = path(vm["DASM_Config.model-dir"].as<string>()); // User Input
			if(!is_directory(inputDir)){
				cerr << inputDir.string() << " is invalid Output directory for Model files.";
				return FAILURE;
			}
			if(Constants::instance()->isVerbose())
				cout << "Output Model Directory: " << modelsDir.string() << endl;

			if(vm.count("DASM_Config.detector-vj")){
				detectorPath = path(vm["DASM_Config.detector-vj"].as<string>()); // User Input
				if(!detectorPath.has_extension())
				{
					cerr << detectorPath.string() << " is not a Cascade File for the Object Detector.";
					return FAILURE;
				}
				Constants::instance()->setVJ();
				if(Constants::instance()->isVerbose())
					cout << "Using Viola-Jones Detector" << endl;
			}
			else if(vm.count("DASM_Config.detector-pp")){
				detectorPath = path(vm["DASM_Config.detector-pp"].as<string>()); // User Input
				if(!is_directory(detectorPath))
				{
					cerr << detectorPath.string() << " is not valid input for the PittPatt detector.";
					return FAILURE;
				}
				Constants::instance()->setPP();
				if(Constants::instance()->isVerbose())
					cout << "Using PittPatt Detector" << endl;
			}

			partsPath = path(vm["DASM_Config.parts-path"].as<string>()); // User Input
			if(!partsPath.has_extension()) 
			{
				cerr << detectorPath.string() << " is not a Parts File.";
				return FAILURE;
			}
			
			if(vm.count("DASM_Config.profLength2d")){	 // User Input
				int temp = vm["DASM_Config.profLength2d"].as<int>();
				if(temp >=3 && temp%2!=0){ // Error Checking
					Constants::instance()->setP2d(temp);
					Constants::instance()->setb(temp);
				}
			}
			if(vm.count("DASM_Config.profLength1d")){		// User Input		
				int temp = vm["DASM_Config.profLength1d"].as<int>();
				if(temp >=3 && temp%2!=0) // Error Checking
					Constants::instance()->setP1d(temp); 
			}
			if(vm.count("DASM_Config.num1DLevels")){		// User Input
				int temp = vm["DASM_Config.num1DLevels"].as<int>();
				if(temp >= 0 && temp < 6) // Error Checking
					Constants::instance()->setN1d(temp); 
			}
			if(vm.count("DASM_Config.num2DLevels")){		// User Input
				int temp = vm["DASM_Config.num2DLevels"].as<int>();
				if(temp >= 0 && temp < 6) // Error Checking
					Constants::instance()->setN2d(temp); 
			}
			if(vm.count("DASM_Config.stdWidth") && vm.count("DASM_Config.stdHeight")){
				int tempW = vm["DASM_Config.stdWidth"].as<int>();
				int tempH = vm["DASM_Config.stdHeight"].as<int>();
				if(tempW > 0 && tempH > 0)
					Constants::instance()->setSize(tempW, tempH);
			}

			
			cout << "DASM training started " << now.date() << " " << now.time_of_day() << endl;
			// Begin the Training
			s.close();
			int code = trainASM(inputDir, modelsDir, modelsDir, partsPath, detectorPath);

			return code; //Exit Program
		}
		else
		{
			cerr << "Too few arguments specified." << endl;
			return FAILURE;
		}
		s.close();
	}
	// Search Option
	else if(vm.count("DASM_Config.search") && (vm["DASM_Config.search"].as<string>().compare("true")) == 0 )
	{
		if(vm.count("DASM_Config.input-dir") && vm.count("DASM_Config.model-path") && (vm.count("DASM_Config.detector-vj") || vm.count("DASM_Config.detector-pp"))) // Required arguments
		{
			inputDir = path(vm["DASM_Config.input-dir"].as<string>()); // User Input
			if(!is_directory(inputDir)){
				cerr << inputDir.string() << " is invalid Image directory.";
				return FAILURE;
			}
			modelsDir = path(vm["DASM_Config.model-path"].as<string>()); // User Input
			if(!modelsDir.has_extension())
			{
				cerr << modelsDir.string() << " is invalid Model File.";
				return FAILURE;
			}
			if(vm.count("DASM_Config.detector-vj")){
				detectorPath = path(vm["DASM_Config.detector-vj"].as<string>()); // User Input
				if(!detectorPath.has_extension())
				{
					cerr << detectorPath.string() << " is not a valid Viola-Jones cascade file.";
					return FAILURE;
				}
				Constants::instance()->setVJ();
			}
			else if(vm.count("DASM_Config.detector-pp")){
				detectorPath = path(vm["DASM_Config.detector-pp"].as<string>()); // User Input
				if(!is_directory(detectorPath))
				{
					cerr << detectorPath.string() << " is not valid input for the PittPatt detector.";
					return FAILURE;
				}
				Constants::instance()->setPP();
			}

			if(vm.count("DASM_Config.out-dir"))
			{
				outputDir = path(vm["DASM_Config.out-dir"].as<string>()); // User Input
			}
			else{
				outputDir = current_path(); // Default Location
				outputDir /= "Points";
			}
			if(vm.count("DASM_Config.eigPercent")){	
				int temp = vm["DASM_Config.eigPercent"].as<int>(); // User Input
				if(temp > 0 && temp <= 100) // Error Checking
					Constants::instance()->setEP(temp/100.0f); // make it a percentage
			}
			if(vm.count("DASM_Config.searchParam")){
				int temp = vm["DASM_Config.searchParam"].as<int>(); // User Input
				if(temp == 1 || temp == 2) // Error Checking
					Constants::instance()->setSP(temp);
			}
			if(vm.count("DASM_Config.stacked")){
				int temp = vm["DASM_Config.stacked"].as<int>(); // User Input
				if(temp == 1 || temp == 2) // Error Checking
					Constants::instance()->setSt(temp);
			}

			s.close();
			cout << endl << "DASM search started " << now.date() << " " << now.time_of_day() << endl << endl;
			// Begin the Search
			int code = searchASM(inputDir, modelsDir, detectorPath, outputDir);
			
			return code; //Exit Program
		}
		else
		{
			cerr << "Too few arguments specified." << endl;
			return FAILURE;
		}
		s.close();
	}
	else
	{
		cerr << desc << endl;
		return FAILURE;
	}
}


