/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

/** 
 * @version 1.0
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QApplication>
#include <QFileInfo>
#include <QDir>

// Project
#include "GvxVoxelizerDialog.h"
#include "GvxVoxelizerEngine.h"
#include "GvxDataTypeHandler.h"
#include "GvxAssimpSceneVoxelizer.h"
#include "GvxWriter.h"

// STL
#include <string>
#include <iostream>
#include <cassert>
#include <sstream>

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

// TO DO
// This CImg dependency should be placed in an encapsulated class...
// ...
// CImg
#define cimg_use_magick	// Beware, this definition must be placed before including CImg.h
#include <CImg.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// Project
using namespace Gvx;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

// CImg
void printCImgLibraryInfo();

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char* pArgv[] )
{
	// Exit code
	int result = 0;

	// LOG
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "-------- GigaSpace Voxelizer ---------" << std::endl;
	std::cout << "--------------------------------------" << std::endl;
	std::cout << "\n" << std::endl;
	std::cin;
	// TO DO
	// - add a version
	// ...

#ifndef NDEBUG
	// LOG : print CImg settings
	printCImgLibraryInfo();
#endif

	// Qt main application
	QApplication application( pArgc, pArgv );
	
	// Show a USER-dialog to parameterize the voxelization process
	GvxVoxelizerDialog voxelizerDialog;
	if ( QDialog::Rejected == voxelizerDialog.exec() )
	{
		// Handle error
		return 1;
	}

	// Create a scene voxelizer
	GvxSceneVoxelizer* sceneVoxelizer = new GvxAssimpSceneVoxelizer();
	if ( sceneVoxelizer == NULL )
	{
		// TO DO
		// Handle error
		// ...
		return 2;
	}

	// Initialize the scene voxelizer
	QFileInfo fileInfo( voxelizerDialog._fileName );
	sceneVoxelizer->setFilePath( QString( fileInfo.absolutePath() + QDir::separator() ).toLatin1().constData() );
	sceneVoxelizer->setFileName( fileInfo.completeBaseName().toLatin1().constData() );
	sceneVoxelizer->setFileExtension( QString( "." + fileInfo.suffix() ).toLatin1().constData() );
	sceneVoxelizer->setMaxResolution( voxelizerDialog._maxResolution );
	sceneVoxelizer->setBrickWidth( voxelizerDialog._brickWidth );
	sceneVoxelizer->setDataType( static_cast< Gvx::GvxDataTypeHandler::VoxelDataType >( voxelizerDialog._dataType ) );
	sceneVoxelizer->setFilterType( voxelizerDialog._filterType );
	sceneVoxelizer->setFilterIterations( voxelizerDialog._nbFilterOperation );
	sceneVoxelizer->setNormals( voxelizerDialog._normals );

	// TO DO
	// Check input data here or in the voxelizer.
	// Because, in the future, the voxelizer could be called either by a GUI interface or by command line (with a configuration batch file)
	//...
	
	// LOG
	std::cout << "-------- BEGIN voxelization process --------" << std::endl;
	std::cout << "\n" << std::endl;

	// Launch the voxelization
	sceneVoxelizer->launchVoxelizationProcess();

	// LOG
	std::cout << "\n" << std::endl;
	std::cout << "-------- END voxelization process --------" << std::endl;

	// Enter Qt main event loop
	//result = application.exec();

	int nbChannels = 1;
	if ( voxelizerDialog._normals )
	{
		nbChannels = 2;
	}

	// Use GigaSpace Meta Data file writer
	GvxWriter* gigaSpaceWriter = new GvxWriter();
	// - exporter configuration
	gigaSpaceWriter->setModelDirectory( fileInfo.absolutePath().toLatin1().constData() );
	gigaSpaceWriter->setModelName( sceneVoxelizer->getFileName().c_str() );
	gigaSpaceWriter->setModelMaxResolution( sceneVoxelizer->getMaxResolution() );
	gigaSpaceWriter->setBrickWidth( sceneVoxelizer->getBrickWidth() );
	gigaSpaceWriter->setNbDataChannels( nbChannels );
	// - not used for the moment...
	vector< string > dataTypeNames;
	gigaSpaceWriter->setDataTypeNames( dataTypeNames );
	// - write Meta Data file
	bool statusOK = gigaSpaceWriter->write();
	// Handle error(s)
	assert( statusOK );
	if ( ! statusOK )
	{
		std::cout << "Error during writing GigaSpace's Meta Data file" << std::endl;
	}
	// Destroy GigaSpace Meta Data file writer
	delete gigaSpaceWriter;
	gigaSpaceWriter = NULL;

	// Return exit code
	return result;
}

/******************************************************************************
 * Print informations about the CImg library environement variables.
 *
 * TO DO : move this method in a CImg wrapper
 ******************************************************************************/
void printCImgLibraryInfo()
{
	cimg_library::cimg::info();
}
