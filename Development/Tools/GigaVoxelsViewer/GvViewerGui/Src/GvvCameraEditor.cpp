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

#include "GvvCameraEditor.h"

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

// STL
#include <iostream>

// System
#include <cassert>

// QGLViewer
#include <QGLViewer/qglviewer.h>

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
 * Default constructor
 ******************************************************************************/
GvvCameraEditor::GvvCameraEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Camera Editor" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvCameraEditor::~GvvCameraEditor()
{
}

/******************************************************************************
 * Slot called when camera field of view value has changed
 ******************************************************************************/
void GvvCameraEditor::on__fieldOfViewDoubleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		// camera() always returns the associated qglviewer::Camera, never NULL
		pipelineViewer->camera()->setFieldOfView( pValue );
	}
}

/******************************************************************************
 * Slot called when camera scene radius value has changed
 ******************************************************************************/
void GvvCameraEditor::on__sceneRadiusDoubleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		pipelineViewer->camera()->setSceneRadius( pValue );
	}
}

/******************************************************************************
 * Slot called when camera z near coefficient value has changed
 ******************************************************************************/
void GvvCameraEditor::on__zNearCoefficientDoubleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		pipelineViewer->camera()->setZNearCoefficient( pValue );
	}
}

/******************************************************************************
 * Slot called when camera z clipping coefficient value has changed
 ******************************************************************************/
void GvvCameraEditor::on__zClippingCoefficientDoubleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		pipelineViewer->camera()->setZClippingCoefficient( pValue );
	}
}
