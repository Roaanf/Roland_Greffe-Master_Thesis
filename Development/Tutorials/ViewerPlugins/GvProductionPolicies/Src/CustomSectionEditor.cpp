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

#include "CustomSectionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QtCore/QUrl>
#include <QtGui/QFileDialog>
#include <QtGui/QMessageBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QToolBar>

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
#include "SampleCore.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QColorDialog>

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
CustomSectionEditor::CustomSectionEditor( QWidget *parent, Qt::WindowFlags flags )
:	GvvSectionEditor( parent, flags )
,	_init( false )
{
	setupUi( this );

	// Editor name
	setName( tr( "Mandelbulb - Priority Management" ) );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
CustomSectionEditor::~CustomSectionEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void CustomSectionEditor::populate( GvViewerCore::GvvBrowsable* pBrowsable )
{
	assert( pBrowsable != NULL );
	SampleCore* pipeline = dynamic_cast< SampleCore* >( pBrowsable );
	assert( pipeline != NULL );
	if ( pipeline != NULL )
	{
		// Fractal parameters
		_powerSpinBox->setValue( pipeline->getFractalPower() );
		_nbIterationSpinBox->setValue( pipeline->getFractalNbIterations() );
		_useAdaptativeIterationCheckBox->setChecked( pipeline->hasFractalAdaptativeIterations() );

		// Priority policies parameters
		QString labelName;
		_priorityPolicyComboBox->addItem( tr( "No priority" ), QVariant( ePriorityPolicy_noPriority ) );
		_priorityPolicyComboBox->addItem( tr( "Default" ), QVariant( ePriorityPolicy_default ) );
		_priorityPolicyComboBox->addItem( tr( "Nearest" ), QVariant( ePriorityPolicy_nearest ) );
		_priorityPolicyComboBox->addItem( tr( "Farthest" ), QVariant( ePriorityPolicy_farthest ) );
		_priorityPolicyComboBox->addItem( tr( "Most detailed first" ), QVariant( ePriorityPolicy_mostDetailedFirst ) );
		_priorityPolicyComboBox->addItem( tr( "Least detailed first" ), QVariant( ePriorityPolicy_leastDetailedFirst ) );
		_priorityPolicyComboBox->addItem( tr( "Farthest from optimal size" ), QVariant( ePriorityPolicy_farthestFromOptimalSize ) );
		
		int index = _priorityPolicyComboBox->findData( pipeline->getProductionPriorityPolicy() );
		if ( index != -1 )
		{
			_priorityPolicyComboBox->setCurrentIndex( index );
		}
		else
		{
			_priorityPolicyComboBox->setCurrentIndex( 0 );
		}
	}

	// Update internal state
	_init = true;
}

/******************************************************************************
 * Helper function
 ******************************************************************************/
SampleCore* getSampleCore()
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	GvViewerCore::GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();

	SampleCore* sampleCore = dynamic_cast< SampleCore* >( pipeline );
	assert( sampleCore != NULL );

	return sampleCore;
}

/******************************************************************************
 * Slot called when the fractal's power value has changed
 ******************************************************************************/
void CustomSectionEditor::on__powerSpinBox_valueChanged( int pValue )
{
	getSampleCore()->setFractalPower( pValue );
}

/******************************************************************************
 * Slot called when the fractal's nb iterations value has changed
 ******************************************************************************/
void CustomSectionEditor::on__nbIterationSpinBox_valueChanged( int pValue )
{
	getSampleCore()->setFractalNbIterations( pValue );
}

/******************************************************************************
 * Slot called when the fractal's nb iterations mode has changed
 ******************************************************************************/
void CustomSectionEditor::on__useAdaptativeIterationCheckBox_toggled( bool pChecked )
{
	getSampleCore()->setFractalAdaptativeIterations( pChecked );
}

/******************************************************************************
 * Slot called when the priority policy changes
 ******************************************************************************/
void CustomSectionEditor::on__priorityPolicyComboBox_currentIndexChanged( int index )
{
	// The setter is called before the plugin is populated, but we mustn't change anything before.
	if ( _init )
	{
		PriorityPolicies policy = static_cast< PriorityPolicies >( _priorityPolicyComboBox->itemData( index ).toInt() );
		getSampleCore()->setProductionPriorityPolicy( static_cast< PriorityPolicies >( index ));
	}
}
