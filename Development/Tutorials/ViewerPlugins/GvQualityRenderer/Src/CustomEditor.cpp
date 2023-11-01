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

// Qt
#include <QUrl>
#include <QFileDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QToolBar>

// GvViewer
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPluginManager.h"

#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

// Project
#include "CustomSectionEditor.h"

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// STL
using namespace std;

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
 * ...
 *
 * @param pParent ...
 * @param pBrowsable ...
 *
 * @return ...
 ******************************************************************************/
GvvEditor* CustomEditor::create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable )
{
	return new CustomEditor( pParent );
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
CustomEditor::CustomEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvPipelineEditor( parent, flags )
{
	_customSectionEditor = new CustomSectionEditor( parent, flags );

	_sectionEditors.push_back( _customSectionEditor );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomEditor::~CustomEditor()
{
}
