DASM
====
Dynamic Active Shape Models

Summary:
This application is designed with the sole purpose of automatically locating landmarks on objects in images. In particular, the application is based on Active Shape Models and the extensions to the original algorithm provided by STASM 3.1. In short, the software provides a means of training a "model" from a set of images with manually annotated landmarks. The model can then be used to identify landmarks on 

Building the project:
The software can be built in either x86 or x64 bit modes. The software has thus far only been tested on the Windows platform, but all of the necessary dependencies are supported cross-platform, so there it is therefore possible to build the software on Linux/OSX. On Windows, the easiest method is to grab the Visual Studio project file and build from there. However, if you do not have Visual Studio, you can compile the project using gcc. 

Dependencies:
The application requires two libraries to be installed on the user's machine: OpenCV and Boost. The software was built on OpenCV 2.4, but any version from 2.0 upwards is compatible, as well as Boost version 1.53. 

To build the software, you will need to include the header files located in the opencv/include and the boost/include directories.
Also, you will need to reference the opencv and boost lib folders. Make sure that if you are building in x64 bit mode, that you are referencing the x64 lib files. Link the following library files and their corresponding DLL's will need to placed in either the System Environment Path or in the same directory as the executable/library:
opencv_imgproc2**
opencv_core2**
opencv_highgui2**
opencv_objdetect2**
(When you download the opencv package, they are located within the build directory)

OpenMP is also supported by the program. Ensure that your compiler supports OpenMP in order to experience the gains from parallel processing. 

Running the program:
The application can perform 1 of 2 operations, train or search, depending on the configuration parameters passed to the program. Intuitively, training is a necessary prerequisite to search. The configuration files (search.cfg, train.cfg) contain all of the information required by the program to perform train/search. The parameters defined in these files will be explained in the user manual, however, they are for the most part self explainable.
