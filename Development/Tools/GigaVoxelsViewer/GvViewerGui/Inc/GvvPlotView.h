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

#ifndef _GVV_PLOT_VIEW_H_
#define _GVV_PLOT_VIEW_H_

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
 * @class GvvPlotView
 *
 * @brief The GvvPlotView class provides ...
 *
 * @ingroup GvViewerGui
 *
 * This class is used to ...
 */
class GVVIEWERGUI_EXPORT GvvPlotView : public QwtPlot
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
	GvvPlotView( QWidget* pParent, const char* pName = 0 );

	/**
	 * Default destructor
	 */
	virtual ~GvvPlotView();

	/**
	 * Draw the specified curve
	 *
	 * @param pCurve specifies the curve to be drawn
	 */
	void onCurveChanged( unsigned int pFrame, unsigned int pNodeValue, unsigned int pBrickValue, unsigned int pUnusedNodeValue, unsigned int pUnusedBrickValue );

	/********************************* SLOTS **********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node curve
	 */
	QwtPlotCurve* _nodeCurve;
	double _xDataNode[ 1000 ];
	double _yDataNode[ 1000 ];
	QwtPlotMarker* _nodeMarker;

	/**
	 * Brick curve
	 */
	QwtPlotCurve* _brickCurve;
	double _xDataBrick[ 1000 ];
	double _yDataBrick[ 1000 ];
	QwtPlotMarker* _brickMarker;

	///**
	// * Unused nodes
	// */
	//QwtPlotCurve* _unusedNodesCurve;
	//double _xDataUnusedNodes[ 1000 ];
	//double _yDataUnusedNodes[ 1000 ];
	//QwtPlotMarker* _unusedNodesMarker;

	///**
	// * Unused bricks
	// */
	//QwtPlotCurve* _unusedBricksCurve;
	//double _xDataUnusedBricks[ 1000 ];
	//double _yDataUnusedBricks[ 1000 ];
	//QwtPlotMarker* _unusedBricksMarker;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerGui

#endif
