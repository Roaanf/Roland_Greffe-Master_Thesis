/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Qt
#include <QApplication>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

// Simple Sphere
#include "SampleViewer.h"

/******************************************************************************
 * Main entry program
 *
 * @param argc ...
 * @param argv ...
 *
 * @return ...
 ******************************************************************************/
int main( int argc, char* argv[] )
{
	// Exit code
	int result = 0;

	// GLUT initialization
	glutInit( &argc, argv );

	// Qt main application
	QApplication app( argc, argv );

	// Initialize GigaSpace environment
	GvUtils::GsEnvironment::initialize( QCoreApplication::applicationDirPath().toLatin1().constData() );
	/*bool statusOK = */GvUtils::GsEnvironment::initializeSettings();
	Gs::initialize();

	// Create your QGLViewer custom widget
	SampleViewer* sampleViewer = new SampleViewer();
	sampleViewer->setWindowTitle( "Simple Triangles example" );
	sampleViewer->show();

	// Enter Qt main event loop
	result = app.exec();

	// Release memory
	delete sampleViewer;

	// Finalize GigaSpace
	Gs::finalize();

	// CUDA tip: clean up to ensure correct profiling
	cudaError_t error = cudaDeviceReset();

	// Return exit code
	return result;
}
