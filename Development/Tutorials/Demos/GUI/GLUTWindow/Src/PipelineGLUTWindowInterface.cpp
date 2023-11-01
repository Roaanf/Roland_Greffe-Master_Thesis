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

#include "PipelineGLUTWindowInterface.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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
PipelineGLUTWindowInterface::PipelineGLUTWindowInterface()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
PipelineGLUTWindowInterface::~PipelineGLUTWindowInterface()
{
}

/******************************************************************************
 * Initialize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineGLUTWindowInterface::initialize()
{
	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return Flag to tell wheter or not it succeded
 ******************************************************************************/
bool PipelineGLUTWindowInterface::finalize()
{
	return true;
}

/******************************************************************************
 * Display callback
 ******************************************************************************/
void PipelineGLUTWindowInterface::onDisplayFuncExecuted()
{
}

/******************************************************************************
 * Reshape callback
 *
 * @param pWidth The new window width in pixels
 * @param pHeight The new window height in pixels
 ******************************************************************************/
void PipelineGLUTWindowInterface::onReshapeFuncExecuted( int pWidth, int pHeight )
{
}

/******************************************************************************
 * Keyboard callback
 *
 * @param pKey ASCII character of the pressed key
 * @param pX Mouse location in window relative coordinates when the key was pressed
 * @param pY Mouse location in window relative coordinates when the key was pressed
 ******************************************************************************/
void PipelineGLUTWindowInterface::onKeyboardFuncExecuted( unsigned char pKey, int pX, int pY )
{
}

/******************************************************************************
 * Mouse callback
 *
 * @param pButton The button parameter is one of left, middle or right.
 * @param pState The state parameter indicates whether the callback was due to a release or press respectively.
 * @param pX Mouse location in window relative coordinates when the mouse button state changed
 * @param pY Mouse location in window relative coordinates when the mouse button state changed
 ******************************************************************************/
void PipelineGLUTWindowInterface::onMouseFuncExecuted( int pButton, int pState, int pX, int pY )
{
}

/******************************************************************************
 * Idle callback
 ******************************************************************************/
void PipelineGLUTWindowInterface::onIdleFuncExecuted()
{
}
