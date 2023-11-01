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

#include "CustomEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "CustomSectionEditor.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

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
 * Creator function
 *
 * @param pParent parent widget
 * @param pBrowsable pipeline element from which the editor will be associated
 *
 * @return the editor associated to the GigaVoxels pipeline
 ******************************************************************************/
GvvEditor* CustomEditor::create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable )
{
	return new CustomEditor( pParent );
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
CustomEditor::CustomEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvPipelineEditor( pParent, pFlags )
,	_customSectionEditor( NULL )
{
	// Create the user custom editor
	_customSectionEditor = new CustomSectionEditor( pParent, pFlags );
	assert( _customSectionEditor != NULL );
	if ( _customSectionEditor != NULL )
	{
		// Store the user custom editor
		_sectionEditors.push_back( _customSectionEditor );
	}
	else
	{
		// TO DO handle error
		// ...
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomEditor::~CustomEditor()
{
}
