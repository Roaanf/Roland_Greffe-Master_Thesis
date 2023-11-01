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

// Qtfe
class Qtfe;

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

	Q_OBJECT

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

	/**
	 * Transfer function editor
	 */
	Qtfe* _transferFunctionEditor;
	
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

	/**
	 * Mouse press event handler
	 *
	 * @param e the event
	 */
	virtual void mousePressEvent( QMouseEvent* e );

	/**
	 * Mouse move event handler
	 *
	 * @param e the event
	 */
	virtual void mouseMoveEvent( QMouseEvent* e );

	/**
	 * Mouse release event handler
	 *
	 * @param e the event
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

	/********************************* SLOTS **********************************/

protected slots:

	/**
	 * Slot called when at least one canal changed
	 */
	void onFunctionChanged();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	SampleCore* _sampleCore;

	///**
	// * Light
	// */
	//qglviewer::ManipulatedFrame* _light1;

	/**
	 * ...
	 */
	bool _moveLight;

	/**
	 * ...
	 */
	bool _controlLight;

	/**
	 * ...
	 */
	float _light[ 7 ]; // ( x, y, z, theta, phi, xpos, ypos )

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLEVIEWER_H_
