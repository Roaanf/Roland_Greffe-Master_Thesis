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

#include "TimeBudgetView.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "PlotView.h"
#include "SampleCore.h"

// STL
#include <iostream>

// System
#include <cassert>

// Qt
#include <QString>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
TimeBudgetView::TimeBudgetView( QWidget* pParent, Qt::WindowFlags pFlags )
:	QWidget( pParent, pFlags )
,	_plotView( NULL )
,	_pipeline( NULL )
{
	setupUi( this );

	// Editor name
	setObjectName( tr( "Frame Time Budget Monitor" ) );

	// ------------------------------------------------------

	_plotView =  new PlotView( this, "Frame Time Budget Monitor" );
	
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
TimeBudgetView::~TimeBudgetView()
{
}

/******************************************************************************
 * Populates this editor with the specified browsable
 *
 * @param pBrowsable specifies the browsable to be edited
 ******************************************************************************/
void TimeBudgetView::populate( SampleCore* pPipeline )
{
	assert( pPipeline != NULL );
	if ( pPipeline != NULL )
	{
		_pipeline = pPipeline;
	
		blockSignals( true );

		_timeBudgetParametersGroupBox->setChecked( pPipeline->hasTimeBudget() );
		_timeBudgetSpinBox->setValue( pPipeline->getTimeBudget() );
		_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pPipeline->getTimeBudget() ) * 1000.f ) + QString( " ms" ) );
		_plotView->setTimeBudget( pPipeline->getTimeBudget() );

		blockSignals( false );
	}
}

/******************************************************************************
 * Draw the specified curve
 *
 * @param pCurve specifies the curve to be drawn
 ******************************************************************************/
void TimeBudgetView::onCurveChanged( unsigned int pFrame, float pFrameTime )
{
	assert( _plotView != NULL );

	_plotView->onCurveChanged( pFrame, pFrameTime );
}

/******************************************************************************
 * Slot called when time budget parameters group box state has changed
 ******************************************************************************/
void TimeBudgetView::on__timeBudgetParametersGroupBox_toggled( bool pChecked )
{
	assert( _pipeline  != NULL );

	_pipeline->setTimeBudgetActivated( pChecked );
}

/******************************************************************************
 * Slot called when user requested time budget value has changed
 ******************************************************************************/
void TimeBudgetView::on__timeBudgetSpinBox_valueChanged( int pValue )
{
	assert( _pipeline  != NULL );

	_pipeline->setTimeBudget( pValue );
	_plotView->setTimeBudget( pValue );
	_timeBudgetLineEdit->setText( QString::number( 1.f / static_cast< float >( pValue ) * 1000.f ) + QString( " ms" ) );
}
