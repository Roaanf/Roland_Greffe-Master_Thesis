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

#include "GvvTimeBudgetMonitoringEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvvTimeBudgetPlotView.h"
#include "GvvPipelineInterface.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QString>

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
 * Constructor
 *
 * pParent ...
 * pFlags ...
 ******************************************************************************/
GvvTimeBudgetMonitoringEditor::GvvTimeBudgetMonitoringEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_plotView( NULL )
,	_pipeline( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Frame Time Budget Monitor" ) );

	// ------------------------------------------------------

	_plotView =  new GvvTimeBudgetPlotView( this, "Frame Time Budget Monitor" );
	
	QHBoxLayout* layout = new QHBoxLayout();
	_frameTimeViewGroupBox->setLayout( layout );
	assert( layout != NULL );
	if ( layout != NULL )
	{
		layout->addWidget( _plotView );
	}

	// ------------------------------------------------------
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvTimeBudgetMonitoringEditor::~GvvTimeBudgetMonitoringEditor()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pPipeline specifies the pipeline to be edited
 ******************************************************************************/
void GvvTimeBudgetMonitoringEditor::populate( GvvPipelineInterface* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		_pipeline = pPipeline;
	
		blockSignals( true );

//		_timeBudgetParametersGroupBox->setChecked( pPipeline->hasRenderingTimeBudget() );
//		_timeBudgetSpinBox->setValue( pPipeline->getRenderingTimeBudget() );
//		_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pPipeline->getRenderingTimeBudget() ) * 1000.f ) + QString( " ms" ) );
		_plotView->setTimeBudget( pPipeline->getRenderingTimeBudget() );

		blockSignals( false );
	}
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void GvvTimeBudgetMonitoringEditor::onCurveChanged( unsigned int pFrame, float pFrameTime )
{
	assert( _plotView != NULL );

	_plotView->onCurveChanged( pFrame, pFrameTime );
}

///******************************************************************************
// * Slot called when time budget parameters group box state has changed
// ******************************************************************************/
//void GvvTimeBudgetMonitoringEditor::on__timeBudgetParametersGroupBox_toggled( bool pChecked )
//{
//	assert( _pipeline  != NULL );
//
//	_pipeline->setTimeBudgetActivated( pChecked );
//}
//
///******************************************************************************
// * Slot called when user requested time budget value has changed
// ******************************************************************************/
//void GvvTimeBudgetMonitoringEditor::on__timeBudgetSpinBox_valueChanged( int pValue )
//{
//	assert( _pipeline  != NULL );
//
//	_pipeline->setRenderingTimeBudget( pValue );
//	_plotView->setTimeBudget( pValue );
//	_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pValue ) * 1000.f ) + QString( " ms" ) );
//}
