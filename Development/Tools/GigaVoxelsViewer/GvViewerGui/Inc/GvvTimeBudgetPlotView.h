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

#ifndef _GVV_TIME_BUDGET_PLOT_VIEW_H_
#define _GVV_TIME_BUDGET_PLOT_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qwt
#include <qwt_plot.h>

// Qt
#include <QColor>
#include <QObject>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Qwt
class QwtPlotCurve;
class QwtPlotMarker;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class PlotView
 *
 * @brief The PlotView class provides functionalities to display frame time plot.
 *
 * This class is used to ...
 */
class GVVIEWERGUI_EXPORT GvvTimeBudgetPlotView : public QwtPlot
{

	Q_OBJECT

	/**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
    /******************************** METHODS *********************************/

	/**
	 * Constructs a plot widget
	 */
	GvvTimeBudgetPlotView( QWidget* pParent, const char* pName = 0 );

	/**
	 * Default destructor
	 */
	virtual ~GvvTimeBudgetPlotView();

	/**
	 * Draw the specified curve
	 *
	 * @param pCurve specifies the curve to be drawn
	 */
	void onCurveChanged( unsigned int pFrame, float pFrameTime );

	/**
	 * Set the user requested time budget
	 *
	 * @param pValue the user requested time budget
	 */
	void setTimeBudget( unsigned int pValue );

	/********************************* SLOTS **********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Frame time curve
	 */
	QwtPlotCurve* _frameTimeCurve;
	double _xFrameTime[ 500 ];
	double _yFrameTime[ 500 ];
	QwtPlotMarker* _frameTimeMarker;

	/**
	 * User requested curve
	 */
	QwtPlotCurve* _timeBudgetCurve;
	double _xTimeBudget[ 500 ];
	double _yTimeBudget[ 500 ];
	QwtPlotMarker* _timeBudgetMarker;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvTimeBudgetPlotView( const GvvTimeBudgetPlotView& );

	/**
	 * Copy operator forbidden.
	 */
	GvvTimeBudgetPlotView& operator=( const GvvTimeBudgetPlotView& );

};

} // namespace GvViewerGui

#endif
