/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** 
 * @version 1.0
 */

#ifndef PIPELINEWINDOW_H
#define PIPELINEWINDOW_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// OpenGL
#include <GL/glew.h>

// Project
#include "PipelineGLUTWindowInterface.h"
#include "Pipeline.h"

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
 * @class PipelineWindowAdaptor
 *
 * @brief The PipelineWindowAdaptor class provides...
 *
 * ...
 */
class PipelineWindow : public PipelineGLUTWindowInterface
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
	PipelineWindow();

	/**
	 * Destructor
	 */
	virtual ~PipelineWindow();
	
	/**
	 * Initialize
	 *
	 * @return Flag to tell wheter or not it succeded
	 */
	virtual bool initialize();

	/**
	 * Finalize
	 *
	 * @return Flag to tell wheter or not it succeded
	 */
	virtual bool finalize();
	
	/**
	 * Display callback
	 */
	virtual void onDisplayFuncExecuted();

	/**
	 * Reshape callback
	 *
	 * @param pWidth The new window width in pixels
	 * @param pHeight The new window height in pixels
	 */
	virtual void onReshapeFuncExecuted( int pWidth, int pHeight );

	/**
	 * Keyboard callback
	 *
	 * @param pKey ASCII character of the pressed key
	 * @param pX Mouse location in window relative coordinates when the key was pressed
	 * @param pY Mouse location in window relative coordinates when the key was pressed
	 */
	virtual void onKeyboardFuncExecuted( unsigned char pKey, int pX, int pY );

	/**
	 * Mouse callback
	 *
	 * @param pButton The button parameter is one of left, middle or right.
	 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
	 * @param pX Mouse location in window relative coordinates when the mouse button state changed
	 * @param pY Mouse location in window relative coordinates when the mouse button state changed
	 */
	virtual void onMouseFuncExecuted( int pButton, int pState, int pX, int pY );

	/**
	 * Idle callback
	 */
	virtual void onIdleFuncExecuted();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
		
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * GigaVoxels pipeline
	 */
	Pipeline* mPipeline;

	///**
	// * ...
	// */
	//qglviewer::ManipulatedFrame* mLight1;

	/******************************** METHODS *********************************/

};

#endif // !PIPELINEWINDOW_H
