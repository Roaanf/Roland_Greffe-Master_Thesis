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

#ifndef _TIME_BUDGET_VIEW_H_
#define _TIME_BUDGET_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Qt
#include <QWidget>

// Project
#include "UI_QPlotView.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
class PlotView;
class SampleCore;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class TimeBudgetView : public QWidget, public Ui::QPlotView
{

	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * pParent ...
	 * pFlags ...
	 */
	TimeBudgetView( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~TimeBudgetView();

	/**
	 * Populates this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline ...
	 */
	void populate( SampleCore* pPipeline );

	/**
	 * Draw the specified curve
	 *
	 * @param pCurve specifies the curve to be drawn
	 */
	void onCurveChanged( unsigned int pFrame, float pFrameTime );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Plot view
	 */
	PlotView* _plotView;

	/**
	 * GigaVoxels pipeline
	 */
	SampleCore* _pipeline;

	/******************************** METHODS *********************************/
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	TimeBudgetView( const TimeBudgetView& );

	/**
	 * Copy operator forbidden.
	 */
	TimeBudgetView& operator=( const TimeBudgetView& );

	/********************************* SLOTS **********************************/

private slots:

	/**
	 * Slot called when time budget parameters group box state has changed
	 */
	void on__timeBudgetParametersGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when user requested time budget value has changed
	 */
	void on__timeBudgetSpinBox_valueChanged( int pValue );

};

#endif
