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

#ifndef _GVV_TIME_BUDGET_MONITORING_EDITOR_H_
#define _GVV_TIME_BUDGET_MONITORING_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQTimeBudgetView.h"

// Qt
#include <QWidget>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvPipelineInterface;
}
namespace GvViewerGui
{
	class GvvTimeBudgetPlotView;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvTimeBudgetMonitoringEditor : public QWidget, public Ui::GvvQTimeBudgetView
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
	GvvTimeBudgetMonitoringEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTimeBudgetMonitoringEditor();

	/**
	 * Populates this editor with the specified GigaVoxels pipeline
	 *
	 * @param pPipeline specifies the pipeline to be edited
	 */
	void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

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
	GvvTimeBudgetPlotView* _plotView;

	/**
	 * GigaVoxels pipeline
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

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
	GvvTimeBudgetMonitoringEditor( const GvvTimeBudgetMonitoringEditor& );

	/**
	 * Copy operator forbidden.
	 */
	GvvTimeBudgetMonitoringEditor& operator=( const GvvTimeBudgetMonitoringEditor& );

	/********************************* SLOTS **********************************/

private slots:

	/**
	 * Slot called when time budget parameters group box state has changed
	 */
//	void on__timeBudgetParametersGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when user requested time budget value has changed
	 */
//	void on__timeBudgetSpinBox_valueChanged( int pValue );

};

} // namespace GvViewerGui

#endif
