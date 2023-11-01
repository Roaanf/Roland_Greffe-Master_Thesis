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

#ifndef _SAMPLE_VIEWER_H_
#define _SAMPLE_VIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// QGLViewer
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

// Project
#include "SampleCore.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleViewer
 *
 * @brief The SampleViewer class provides a viewer widget for rendering.
 *
 * It holds a GigaVoxels pipeline.
 */
class SampleViewer : public QGLViewer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	SampleViewer();

	/**
	 * Destructor
	 */
	virtual ~SampleViewer();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Initialize the viewer
	 */
	virtual void init();

	/**
	 * Draw function called each frame
	 */
	virtual void draw();

	/**
	 * Resize GL event handler
	 *
	 * @param pWidth the new width
	 * @param pHeight the new height
	 */
	virtual void resizeGL( int pWidth, int pHeight );

	/**
	 * Get the viewer size hint
	 *
	 * @return the viewer size hint
	 */
	virtual QSize sizeHint() const;

	/**
	 * Key press event handler
	 *
	 * @param pEvent the event
	 */
	virtual void keyPressEvent( QKeyEvent* pEvent );

	/**
	 * Mouse press event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mousePressEvent( QMouseEvent* e );

	/**
	 * Mouse move event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mouseMoveEvent( QMouseEvent* e );

	/**
	 * Mouse release event handler
	 *
	 * @param pEvent the event
	 */
	virtual void mouseReleaseEvent( QMouseEvent* e );

	/**
	 * Set light
	 *
	 * @param theta ...
	 * @param phi ...
	 */
	void setLight( float theta, float phi );

	/**
	 * Draw light
	 */
	void drawLight() const;

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	bool mMoveLight;
	bool mControlLight;
	float mLight[7]; // (x,y,z,theta,phi, xpos, ypos) 
	SampleCore *mSampleCore;
	//qglviewer::ManipulatedFrame* mLight1;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_VIEWER_H_
