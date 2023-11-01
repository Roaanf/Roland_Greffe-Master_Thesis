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

#include "GvvApplication.h"

// OpenGL
//#include <GL/glew.h>
#include <GL/freeglut.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Main entry program
 *
 * @param pArgc number of arguments
 * @param pArgv list of arguments
 *
 * @return exit code
 ******************************************************************************/
int main( int pArgc, char** pArgv )
{
	// GLUT initialization
	glutInit( &pArgc, pArgv );

	// Qt main application
	GvViewerGui::GvvApplication::initialize( pArgc, pArgv );

	// Enter Qt main event loop
	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	int result = application.execute();

	// Return exit status
	return result;
}
