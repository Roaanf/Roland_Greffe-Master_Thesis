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
	// GLUT initialization
	glutInit( &argc, argv );

	// Qt main application
	QApplication app( argc, argv );

	// Create your QGLViewer custom widget
	SampleViewer* sampleViewer = new SampleViewer();
	sampleViewer->setWindowTitle( "Simple Triangles example" );
	sampleViewer->show();

	// Enter Qt main event loop
	return ( app.exec() );
}
