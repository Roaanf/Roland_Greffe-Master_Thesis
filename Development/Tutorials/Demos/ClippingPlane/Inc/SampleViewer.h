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

#ifndef _SAMPLEVIEWER_H_
#define _SAMPLEVIEWER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// Qt
#include <QGLViewer/qglviewer.h>
#include <QKeyEvent>

// Simple Sphere
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

// QGLViewer
namespace qglviewer
{
	class ManipulatedFrame;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleViewer
 *
 * @brief The SampleViewer class provides...
 *
 * ...
 */
class SampleViewer : public QGLViewer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

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

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Clipping plane
	 */
	qglviewer::ManipulatedFrame* _clippingPlane;

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
	 * @param width the new width
	 * @param height the new height
	 */
	virtual void resizeGL( int width, int height );

	/**
	 * Get the viewer size hint
	 *
	 * @return the viewer size hint
	 */
	virtual QSize sizeHint() const;

	/**
	 * Key press event handler
	 *
	 * @param e the event
	 */
	virtual void keyPressEvent( QKeyEvent* e );
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Core class containing the GigaVoxels pipeline
	 */
	SampleCore* _sampleCore;

	/**
	 * QGL viewer Manipulated Frame used to draw and manipulate a light in the 3D view
	 */
	qglviewer::ManipulatedFrame* _light1;

	/******************************** METHODS *********************************/

};

///**
// * Clipping plane manipulator
// */
//class ClippingPlaneManipulator : public qglviewer::ManipulatedFrame
//{
//
//	/**************************************************************************
//	 ***************************** PUBLIC SECTION *****************************
//	 **************************************************************************/
//
//public:
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
//	/**
//	 * Constructor
//	 */
//	ClippingPlaneManipulator();
//
//	/**
//	 * Destructor
//	 */
//	virtual ~ClippingPlaneManipulator();
//
//	/**
//	 * Implementation of the MouseGrabber main method.
//	 * 
//	 * The ManipulatedFrame grabsMouse() when the mouse is within a 10 pixels region around its Camera::projectedCoordinatesOf() position().
//	 *
//	 * @param pX ...
//	 * @param pY ...
//	 * @param pCamera ...
//	 */
//	virtual void checkIfGrabsMouse( int pX, int pY, const qglviewer::Camera* const pCamera );
//
//	/**************************************************************************
//	 **************************** PROTECTED SECTION ***************************
//	 **************************************************************************/
//
//protected:
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
//	/**************************************************************************
//	 ***************************** PRIVATE SECTION ****************************
//	 **************************************************************************/
//
//private:
//
//	/******************************* ATTRIBUTES *******************************/
//
//	/******************************** METHODS *********************************/
//
//};

#endif // !_SAMPLEVIEWER_H_
