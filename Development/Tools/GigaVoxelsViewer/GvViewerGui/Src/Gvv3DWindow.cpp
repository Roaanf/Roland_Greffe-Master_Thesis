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

#include "Gvv3DWindow.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterfaceViewer.h"

//---------------
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include <QGroupBox>
//---------------

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor
 ******************************************************************************/
Gvv3DWindow::Gvv3DWindow( QWidget* parent, Qt::WindowFlags flags )
:	QObject( parent )
,	mPipelineViewer( NULL )
{
	//mPipelineViewer = new GvvPipelineInterfaceViewer( parent, NULL, flags );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
Gvv3DWindow::~Gvv3DWindow()
{
	delete mPipelineViewer;
	mPipelineViewer = NULL;
}

/******************************************************************************
 * Get the pipeline viewer
 *
 * return The pipeline viewer
 ******************************************************************************/
GvvPipelineInterfaceViewer* Gvv3DWindow::getPipelineViewer()
{
	return mPipelineViewer;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Gvv3DWindow::addViewer()
{
//	delete mPipelineViewer;
//	mPipelineViewer = NULL;

//	GvvApplication& application = GvvApplication::get();
//	GvvMainWindow* mainWindow = application.getMainWindow();

	//GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( mainWindow );
	GvvPipelineInterfaceViewer* viewer = new GvvPipelineInterfaceViewer( NULL );

	//mainWindow->mUi._3DViewGroupBox->layout()->addWidget( viewer );

	mPipelineViewer = viewer;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void Gvv3DWindow::removeViewer()
{
//	GvvApplication& application = GvvApplication::get();
	//GvvMainWindow* mainWindow = application.getMainWindow();

	//mainWindow->mUi._3DViewGroupBox->layout()->removeWidget( mPipelineViewer );

	delete mPipelineViewer;
	mPipelineViewer = NULL;
}
