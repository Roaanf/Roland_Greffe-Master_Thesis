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

#include "SampleViewer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GsGraphics/GsGraphicsCore.h>
#include <GvUtils/GsEnvironment.h>

// System
#include <cstdio>
#include <cstdlib>

// OpenGL
#include <GL/freeglut.h>

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

// Qt
#include <QDir>

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
 * Constructor
 ******************************************************************************/
SampleViewer::SampleViewer()
:	QGLViewer()
,	mControlLight( false )
,	mMoveLight( false )
,	mSampleCore( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	delete mSampleCore;
}

/******************************************************************************
 * Initialize the viewer
 ******************************************************************************/
void SampleViewer::init()
{
	// GLEW initialization
	GLenum error = glewInit();
	if ( error != GLEW_OK )
	{
		// Problem : glewInit failed
		fprintf( stderr, "Error: %s\n", glewGetErrorString( error ) );

		// Exit program
		exit( 1 );
	}

	// LOG associated Graphics Core library properties/capabilities (i.e. OpenGL)
	GsGraphics::GsGraphicsCore::printInfo();

	// Initialize the GigaVoxels pipeline
	mSampleCore = new SampleCore();
	mSampleCore->init();

	// Modify QGLViewer state filename
	QString stateFilename = GvUtils::GsEnvironment::getUserProfilePath().c_str();
	stateFilename += QDir::separator();
	stateFilename += "qglviewer.xml";
	setStateFileName( stateFilename );

	// Read QGLViewer XML settings file if any
	restoreStateFromFile();

	// Viewer settings :
	// - sets the backgroundColor() of the viewer and calls qglClearColor()
	setBackgroundColor( QColor( 51, 51, 51 ) );
	// Update GigaVoxels clear color
	mSampleCore->setClearColor( 51, 51, 51, 255 );

	setLight(0.f, 0.f);

	// Viewer initialization
	setMouseTracking( false );
	// QGLViewer uses a timer to redarw scene, this enables the maximum refreshing rate.
	setAnimationPeriod( 0 );
	startAnimation();
}

/******************************************************************************
 * Draw function called each frame
 ******************************************************************************/
void SampleViewer::draw()
{
	// Clear default frame buffer
	// glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );					// already done by SampleViewr::setBackgroundColor()
	// glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	// already done in QGLViewr::preDraw() method

	glEnable( GL_DEPTH_TEST );
	glDisable( GL_LIGHTING );

	setLight( mLight[ 3 ], mLight[ 4 ] );

	// Render the GigaVoxels scene
	mSampleCore->draw();

	// Draw light if its manipulation is activated
	if ( mControlLight )
	{
		drawLight();
	}
}

/******************************************************************************
 * Resize GL event handler
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleViewer::resizeGL(int width, int height  )
{
	// Handle QGLViewer resize
	QGLViewer::resizeGL( width, height );

	// Handle GigaVoxels resize
	mSampleCore->resize( width, height );
}

/******************************************************************************
 * Get the viewer size hint
 *
 * @return the viewer size hint
 ******************************************************************************/
QSize SampleViewer::sizeHint() const
{
	// Default size
	return QSize( 512, 512 );
}

/******************************************************************************
  * Key press event handler
 *
 * @param pEvent the event
 ******************************************************************************/
void SampleViewer::keyPressEvent( QKeyEvent* e )
{
	switch ( e->key() )
	{
		case Qt::Key_Plus:
			mSampleCore->incMaxVolTreeDepth();
			break;

		case Qt::Key_Minus:
			mSampleCore->decMaxVolTreeDepth();
			break;

		case Qt::Key_C:
			// Tell GigaVoxels to clear its cache
			mSampleCore->clearCache();
			break;

		case Qt::Key_D:
			mSampleCore->toggleDynamicUpdate();
			break;

		case Qt::Key_I:
			// Toggle GigaVoxels performance monitor mechanism (if it has been activated during GigaVoxels compilation)
			mSampleCore->togglePerfmonDisplay( 1 );
			break;

		case Qt::Key_L:
			// Toggle light manipulation mechanism
			mControlLight = !mControlLight;
			mMoveLight = false;
			break;

		case Qt::Key_T:
			// Toggle the display of the GigaVoxels space partitioning structure
			mSampleCore->toggleDisplayOctree();
			break;

		case Qt::Key_U:
			// Toggle GigaVoxels performance monitor mechanism (if it has been activated during GigaVoxels compilation)
			mSampleCore->togglePerfmonDisplay( 2 );
			break;

		default:
			QGLViewer::keyPressEvent( e );
			break;
	}
}

/******************************************************************************
 * Mouse press event handler
 *
 * @param pEvent the event
 ******************************************************************************/
void SampleViewer::mousePressEvent( QMouseEvent* e )
{
	//if ( mLight1->grabsMouse() )
	if ( mControlLight )
	{
		mLight[ 5 ] = e->x();
		mLight[ 6 ] = e->y();
		mMoveLight = true;
	}
	else
	{
		QGLViewer::mousePressEvent( e );
	}
}

/******************************************************************************
 * Mouse move event handler
 *
 * @param pEvent the event
 ******************************************************************************/
void SampleViewer::mouseMoveEvent( QMouseEvent* e )
{
	//if ( mLight1->grabsMouse() )
	if ( mMoveLight )
	{
		int mx = e->x();
		int my = e->y();

		mLight[ 4 ] += ( mLight[ 5 ] - e->x() ) / 100.0f;
		mLight[ 5 ] = e->x();
		mLight[ 3 ] += -( mLight[ 6 ] - e->y() ) / 100.0f;
		mLight[ 6 ] = e->y();

		if ( mLight[ 3 ] < 0.0f )
		{
			mLight[ 3 ] = 0.0f;
		}

		if ( mLight[ 3 ] > (float)M_PI )
		{
			mLight[ 3 ] = (float)M_PI;
		}

		setLight( mLight[ 3 ], mLight[ 4 ] );
	}
	else
	{
		QGLViewer::mouseMoveEvent( e );
	}
}

/******************************************************************************
 * Mouse release event handler
 *
 * @param pEvent the event
 ******************************************************************************/
void SampleViewer::mouseReleaseEvent( QMouseEvent* e )
{
	QGLViewer::mouseReleaseEvent( e );
}

/******************************************************************************
* Draw light
 ******************************************************************************/
void SampleViewer::drawLight() const
{
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();
	gluPerspective( 90.0f, 1.0f, 0.1f, 10.0f );

	glMatrixMode( GL_MODELVIEW );   //changes the current matrix to the modelview matrix
	glPushMatrix();
	glLoadIdentity();
	glColor3f( 1.0f, 1.0f, 0.5f );
	glTranslatef( 0.f,0.f,-1.0f );
	glutSolidSphere( 0.05f, 20.f,20.f );
	glTranslatef( 0.5f * mLight[ 0 ], 0.5f * mLight[ 1 ], 0.5f * mLight[ 2 ] );
	glutSolidSphere( 0.05f, 20.f,20.f );
	glPopMatrix();

	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	glMatrixMode( GL_MODELVIEW );
}

/******************************************************************************
 * Set light
 *
 * @param theta ...
 * @param phi ...
 ******************************************************************************/
void SampleViewer::setLight( float theta, float phi )
{
	mLight[ 0 ] = sinf( theta ) * cosf( phi );
	mLight[ 1 ] = cosf( theta );
	mLight[ 2 ] = sinf( theta ) * sinf( phi ); 
	mLight[ 3 ] = theta;
	mLight[ 4 ] = phi;

	float3 lightDirInView = make_float3( mLight[ 0 ], mLight[ 1 ], mLight[ 2 ] );

	float4x4 modelViewMatrix;
	float4x4 invModelViewMatrix;

	glGetFloatv( GL_MODELVIEW_MATRIX, modelViewMatrix._array );
	modelViewMatrix._array[ 12 ] = 0.f;
	modelViewMatrix._array[ 13 ] = 0.f;
	modelViewMatrix._array[ 14 ] = 0.f;
	modelViewMatrix._array[ 15 ] = 1.f;
	invModelViewMatrix = transpose( modelViewMatrix );

	float3 lightDirInWorld = mulRot( invModelViewMatrix, lightDirInView );
	float3 lightDir = normalize( lightDirInWorld );

	// Update the GigaVoxels pipeline
	mSampleCore->setLightPosition( lightDir.x, lightDir.y, lightDir.z );
}
