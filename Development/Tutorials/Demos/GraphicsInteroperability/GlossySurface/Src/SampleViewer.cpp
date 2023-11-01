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
#include <GvCore/GvError.h>

// System
#include <cstdio>
#include <cstdlib>

// QGLViewer
#include <QGLViewer/manipulatedFrame.h>

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
:	QGLViewer( QGLFormat( QGL::DoubleBuffer | QGL::DepthBuffer | QGL::Rgba | QGL::AlphaChannel | QGL::StencilBuffer /*| QGL::DirectRendering*/ ) )
{
	mSampleCore = new SampleCore();
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleViewer::SampleViewer( QGLFormat& format )
:	QGLViewer( format )
{
	mSampleCore = new SampleCore();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleViewer::~SampleViewer()
{
	delete mSampleCore;
}

/******************************************************************************
 * ...
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

	mSampleCore->init(this);

	restoreStateFromFile();

	mLight1 = new qglviewer::ManipulatedFrame();
	mLight1->setPosition(1.0f, 1.0f, 1.0f);

	glEnable(GL_LIGHT1);

	const GLfloat ambient[]  = {0.2f, 0.2f, 2.0f, 1.0f};
	const GLfloat diffuse[]  = {0.8f, 0.8f, 1.0f, 1.0f};
	const GLfloat specular[] = {0.0f, 0.0f, 1.0f, 1.0f};

	glLightfv(GL_LIGHT1, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT1, GL_SPECULAR, specular);
	glLightfv(GL_LIGHT1, GL_DIFFUSE,  diffuse);

	setMouseTracking(true);
	setAnimationPeriod(0);
	startAnimation();
	setSceneRadius(1500.0);
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleViewer::draw()
{
	glClearColor(0.0f, 0.1f, 0.3f, 0.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);

	float pos[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
	mLight1->getPosition(pos[0], pos[1], pos[2]);

	float3 lightPos = make_float3( pos[ 0 ], pos[ 1 ], pos[ 2 ] );
	qglviewer::Vec light = qglviewer::Vec(lightPos.x, lightPos.y, lightPos.z);
	qglviewer::Vec lightV = camera()->cameraCoordinatesOf(light);
	qglviewer::Vec wlight = camera()->worldCoordinatesOf(lightV);
	mSampleCore->setLightPosition( lightV.x, lightV.y, lightV.z );
	mSampleCore->setWorldLight(wlight.x, wlight.y, wlight.z );

	qglviewer::Vec worldCamPos = camera()->position();
	mSampleCore->setWorldCamera(worldCamPos.x, worldCamPos.y, worldCamPos.z);
	
	// Render GigaVoxels
	mSampleCore->draw();

	glLightfv(GL_LIGHT1, GL_POSITION, pos);
	glEnable(GL_LIGHT1); // must be enabled for drawLight()

	if (mLight1->grabsMouse())
		drawLight(GL_LIGHT1, 1.2f);
	else
		drawLight(GL_LIGHT1);

	glDisable(GL_LIGHT1);
	glDisable(GL_DEPTH_TEST);
}

/******************************************************************************
 * ...
 *
 * @param width ...
 * @param height ...
 ******************************************************************************/
void SampleViewer::resizeGL( int width, int height )
{
	QGLViewer::resizeGL(width, height);
	mSampleCore->resize(width, height);
}

/******************************************************************************
 * ...
 *
 * @return
 ******************************************************************************/
QSize SampleViewer::sizeHint() const
{
	return QSize(512, 512);
}

/******************************************************************************
 * ...
 *
 * @param e ...
 ******************************************************************************/
void SampleViewer::keyPressEvent( QKeyEvent* e )
{
	QGLViewer::keyPressEvent(e);

	switch (e->key())
	{
	case Qt::Key_Plus:
		mSampleCore->incMaxVolTreeDepth();
		break;

	case Qt::Key_Minus:
		mSampleCore->decMaxVolTreeDepth();
		break;

	case Qt::Key_C:
		mSampleCore->clearCache();
		break;

	case Qt::Key_D:
		mSampleCore->toggleDynamicUpdate();
		break;

	case Qt::Key_I:
		mSampleCore->togglePerfmonDisplay(1);
		break;

	case Qt::Key_T:
		mSampleCore->toggleDisplayOctree();
		break;

	case Qt::Key_U:
		mSampleCore->togglePerfmonDisplay(2);
		break;
	}
}
