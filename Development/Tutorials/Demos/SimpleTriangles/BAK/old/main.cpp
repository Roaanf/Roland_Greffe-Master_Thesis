/*
 * Copyright (C) 2011 Fabrice Neyret <Fabrice.Neyret@imag.fr>
 * Copyright (C) 2011 Cyril Crassin <Cyril.Crassin@icare3d.org>
 * Copyright (C) 2011 Morgan Armand <morgan.armand@gmail.com>
 *
 * This file is part of Gigavoxels.
 *
 * Gigavoxels is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gigavoxels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gigavoxels.  If not, see <http://www.gnu.org/licenses/>.
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
