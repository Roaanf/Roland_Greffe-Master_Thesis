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

#include "GvvBrowsable.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

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
 * Default constructor
 ******************************************************************************/
GvvBrowsable::GvvBrowsable()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvBrowsable::~GvvBrowsable()
{
}

/******************************************************************************
 * Returns whether this browsable is checkable
 *
 * @return true if this browsable is checkable
 ******************************************************************************/
bool GvvBrowsable::isCheckable() const
{
	return false;
}

/******************************************************************************
 * Returns whether this browsable is enabled
 *
 * @return true if this browsable is enabled
 ******************************************************************************/
bool GvvBrowsable::isChecked() const
{
	assert( isCheckable() );
	return true;
}

/******************************************************************************
 * Sets this browsable has checked or not
 *
 * @param pFlag specifies whether this browsable is checked or not
 ******************************************************************************/
void GvvBrowsable::setChecked( bool pFlag )
{
	assert( isCheckable() );
}

/******************************************************************************
 * Returns whether this browsable is editable. Returns false by default.
 *
 * @return true if this browsable is editable
 ******************************************************************************/
bool GvvBrowsable::isEditable() const
{
	return false;
}

/******************************************************************************
 * Returns whether this browsable is read-only
 *
 * @return true if this browsable is read only
 ******************************************************************************/
bool GvvBrowsable::isReadOnly() const
{
	return false;
}

/******************************************************************************
 * Tell wheter or not the pipeline has a custom editor.
 *
 * @return the flag telling wheter or not the pipeline has a custom editor
 ******************************************************************************/
bool GvvBrowsable::hasCustomEditor() const
{
	return false;
}
