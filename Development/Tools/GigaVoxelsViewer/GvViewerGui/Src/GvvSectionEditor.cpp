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

#include "GvvSectionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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
GvvSectionEditor::GvvSectionEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_name()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvSectionEditor::~GvvSectionEditor()
{
}

/******************************************************************************
 * Get the name of the editor
 ******************************************************************************/
const QString& GvvSectionEditor::getName() const
{
	return _name;
}

/******************************************************************************
 * Set the name of the editor
 ******************************************************************************/
void GvvSectionEditor::setName( const QString& pName )
{
	_name = pName;
}
