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

#include "GvvEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSectionEditor.h"

// Qt
#include <QWidget>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor.
 ******************************************************************************/
GvvEditor::GvvEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	_sectionEditors()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvEditor::~GvvEditor()
{
	for ( unsigned int i = 0; i < getNbSections(); i++ )
	{
		GvvSectionEditor* sectionEditor = _sectionEditors[ i ];
		delete sectionEditor;
		sectionEditor = NULL;
	}
	_sectionEditors.clear();
}

/******************************************************************************
 * Get the number of sections.
 *
 * @return the number of sections
 ******************************************************************************/
unsigned int GvvEditor::getNbSections() const
{
	return _sectionEditors.size();
}

/******************************************************************************
 * ...
 ******************************************************************************/
GvvSectionEditor* GvvEditor::getSectionEditor( unsigned int pIndex )
{
	assert( pIndex < getNbSections() );
	
	GvvSectionEditor* sectionEditor = NULL;

	if ( pIndex < getNbSections() )
	{
		sectionEditor = _sectionEditors[ pIndex ];
	}

	return sectionEditor;
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void GvvEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	for ( unsigned int i = 0; i < getNbSections(); i++ )
	{
		GvvSectionEditor* sectionEditor = _sectionEditors[ i ];
		if ( sectionEditor != NULL )
		{
			sectionEditor->populate( pBrowsable );
		}
	}
}
