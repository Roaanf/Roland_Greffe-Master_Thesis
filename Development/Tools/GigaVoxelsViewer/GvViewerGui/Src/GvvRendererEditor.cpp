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

#include "GvvRendererEditor.h"

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
#include "GvvTimeBudgetMonitoringEditor.h"

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
GvvRendererEditor::GvvRendererEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
{
	setupUi( this );

	// Editor name
	setName( tr( "Renderer" ) );

	// Do connection(s)
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		QObject::connect( pipelineViewer, SIGNAL( resized( int, int ) ), SLOT( onViewerResized( int, int ) ) );
	}
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvRendererEditor::~GvvRendererEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void GvvRendererEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		_maxDepthSpinBox->setValue( pipeline->getRendererMaxDepth() );
		{
			unsigned int x;
			unsigned int y;
			unsigned int z;
			pipeline->getDataStructureNodeTileResolution( x, y, z );
			unsigned int nodeResolution = x;
			pipeline->getDataStructureBrickResolution( x, y, z );
			unsigned int brickResolution = x;
			unsigned int maxVoxelResolution = 0;
			if ( nodeResolution == 2 )
			{
				maxVoxelResolution = ( 1 << pipeline->getRendererMaxDepth() ) * brickResolution;
			}
			_maxResolutionLineEdit->setText( QString::number( maxVoxelResolution ) );
		}

		_dynamicUpdateCheckBox->setChecked( pipeline->hasDynamicUpdate() );

		_priorityOnBricksRadioButton->setChecked( pipeline->hasRendererPriorityOnBricks() );

		// -- [ Viewport ] --
		_viewportOffscreenSizeGroupBox->setChecked( pipeline->hasImageDownscaling() );
		unsigned int graphicsBufferWidth;
		unsigned int graphicsBufferHeight;
		pipeline->getGraphicsBufferSize( graphicsBufferWidth, graphicsBufferHeight );
		_graphicsBufferWidthSpinBox->setValue( graphicsBufferWidth );
		_graphicsBufferHeightSpinBox->setValue( graphicsBufferHeight );

		// -- Time budget monitoring
		_timeBudgetParametersGroupBox->setChecked( pipeline->hasTimeBudgetMonitoring() );
		_timeBudgetSpinBox->setValue( pipeline->getRenderingTimeBudget() );
	}
}

/******************************************************************************
 * Slot called when max depth value has changed
 ******************************************************************************/
void GvvRendererEditor::on__maxDepthSpinBox_valueChanged( int i )
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
			pipeline->setRendererMaxDepth( i );

			// Update cache info
			unsigned int x;
			unsigned int y;
			unsigned int z;
			pipeline->getDataStructureNodeTileResolution( x, y, z );
			unsigned int nodeResolution = x;
			pipeline->getDataStructureBrickResolution( x, y, z );
			unsigned int brickResolution = x;
			unsigned int maxVoxelResolution = 0;
			if ( nodeResolution == 2 )
			{
				maxVoxelResolution = ( 1 << pipeline->getRendererMaxDepth() ) * brickResolution;
			}
			_maxResolutionLineEdit->setText( QString::number( maxVoxelResolution ) );
		}
	}
}

/******************************************************************************
 * Slot called when cache policy value has changed (dynamic update)
 ******************************************************************************/
void GvvRendererEditor::on__dynamicUpdateCheckBox_toggled( bool pChecked )
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
		GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
		if ( pipeline != NULL )
		{
			pipeline->setDynamicUpdate( pChecked );
		}
	}
}

/******************************************************************************
 * Slot called when the renderer request priority strategy has changed
 ******************************************************************************/
void GvvRendererEditor::on__priorityOnBricksRadioButton_toggled( bool pChecked )
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
			pipeline->setRendererPriorityOnBricks( pChecked );
		}
	}
}

/******************************************************************************
 * Slot called when image downscaling mode value has changed
 ******************************************************************************/
void GvvRendererEditor::on__viewportOffscreenSizeGroupBox_toggled( bool pChecked )
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
			pipeline->setImageDownscaling( pChecked );
		}
	}
}

/******************************************************************************
 * Slot called when graphics buffer width value has changed
 ******************************************************************************/
void GvvRendererEditor::on__graphicsBufferWidthSpinBox_valueChanged( int pValue )
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
			unsigned int graphicsBufferWidth;
			unsigned int graphicsBufferHeight;
			pipeline->getGraphicsBufferSize( graphicsBufferWidth, graphicsBufferHeight );
			pipeline->setGraphicsBufferSize( pValue, graphicsBufferHeight );
		}
	}
}

/******************************************************************************
 * Slot called when graphics buffer height value has changed
 ******************************************************************************/
void GvvRendererEditor::on__graphicsBufferHeightSpinBox_valueChanged( int pValue )
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
			unsigned int graphicsBufferWidth;
			unsigned int graphicsBufferHeight;
			pipeline->getGraphicsBufferSize( graphicsBufferWidth, graphicsBufferHeight );
			pipeline->setGraphicsBufferSize( graphicsBufferWidth, pValue );
		}
	}
}

/******************************************************************************
 * Slot called when time budget monitoring state value has changed
 ******************************************************************************/
void GvvRendererEditor::on__timeBudgetParametersGroupBox_toggled( bool pChecked )
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
			pipeline->setTimeBudgetMonitoring( pChecked );
		}
	}
}

/******************************************************************************
 * Slot called when time budget value has changed
 ******************************************************************************/
void GvvRendererEditor::on__timeBudgetSpinBox_valueChanged( int pValue )
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
			pipeline->setRenderingTimeBudget( pValue );

			// Time budget monitoring view
			// - todo : change this...
			GvvTimeBudgetMonitoringEditor* timeBudgetMonitoringView = GvvApplication::get().getMainWindow()->getTimeBudgetMonitoringView();
			if ( timeBudgetMonitoringView != NULL )
			{
				timeBudgetMonitoringView->populate( pipeline );
			}
		}
	}
}

/******************************************************************************
 * Slot called when the viewer has been resized
 *
 * @param pWidth new viewer width
 * @param pHeight new viewr height
 ******************************************************************************/
void GvvRendererEditor::onViewerResized( int pWidth, int pHeight )
{
	_viewportWidthSpinBox->setValue( pWidth );
	_viewportHeightSpinBox->setValue( pHeight );
}
