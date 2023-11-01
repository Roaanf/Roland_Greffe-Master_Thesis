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

#include "GvvTransformationEditor.h"

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
GvvTransformationEditor::GvvTransformationEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
{
	setupUi( this );

	// Editor name
	setName( tr( "Transforms" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvTransformationEditor::~GvvTransformationEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void GvvTransformationEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		// -- [ Transform ] --
		float x;
		float y;
		float z;
		float w;
		// Translation
		pipeline->getTranslation( x, y, z );
		_xTranslationSpinBox->setValue( x );
		_yTranslationSpinBox->setValue( y );
		_zTranslationSpinBox->setValue( z );
		// Rotation
		pipeline->getRotation( x, y, z, w );
		_angleRotationSpinBox->setValue( x );
		_xRotationSpinBox->setValue( y );
		_yRotationSpinBox->setValue( z );
		_zRotationSpinBox->setValue( w );
		// Scale
		pipeline->getScale( x );
		_uniformScaleSpinBox->setValue( x );
	}
}


/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__xTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float x;
			float y;
			float z;
			pipeline->getTranslation( x, y, z );
			pipeline->setTranslation( pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__yTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float x;
			float y;
			float z;
			pipeline->getTranslation( x, y, z );
			pipeline->setTranslation( x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__zTranslationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float x;
			float y;
			float z;
			pipeline->getTranslation( x, y, z );
			pipeline->setTranslation( x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__xRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float angle;
			float x;
			float y;
			float z;
			pipeline->getRotation( angle, x, y, z );
			pipeline->setRotation( angle, pValue, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__yRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float angle;
			float x;
			float y;
			float z;
			pipeline->getRotation( angle, x, y, z );
			pipeline->setRotation( angle, x, pValue, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__zRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float angle;
			float x;
			float y;
			float z;
			pipeline->getRotation( angle, x, y, z );
			pipeline->setRotation( angle, x, y, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__angleRotationSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			float angle;
			float x;
			float y;
			float z;
			pipeline->getRotation( angle, x, y, z );
			pipeline->setRotation( pValue, x, y, z );
		}
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvTransformationEditor::on__uniformScaleSpinBox_valueChanged( double pValue )
{
	// Temporary, waiting for the global context listerner...
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->setScale( pValue );
		}
	}
}
