/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

#include "SampleViewer.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda SDK
#include <helper_math.h>

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
	if (glewInit() != GLEW_OK)
		exit(1);

	mSampleCore->init();

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

	glLightfv(GL_LIGHT1, GL_POSITION, pos);
	glEnable(GL_LIGHT1); // must be enabled for drawLight()

	if (mLight1->grabsMouse())
		drawLight(GL_LIGHT1, 1.2f);
	else
		drawLight(GL_LIGHT1);

	glDisable(GL_LIGHT1);
	glDisable(GL_DEPTH_TEST);

	//float3 lightPos = make_float3(pos[0], pos[1], pos[2]);
	//GV_CUDA_SAFE_CALL(cudaMemcpyToSymbol(cLightPosition, (&lightPos),
	//	sizeof(lightPos), 0, cudaMemcpyHostToDevice));

	mSampleCore->draw();
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
void SampleViewer::keyPressEvent( QKeyEvent *e )
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
