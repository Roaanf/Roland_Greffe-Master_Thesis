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

#ifndef _GVV_CACHE_USAGE_VIEW_H_
#define _GVV_CACHE_USAGE_VIEW_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvvQCacheUsageWidget.h"

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

// Qwt
class QwtThermo;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * @class GvvCacheUsageView
 *
 * @brief The GvvCacheUsageView class provides ...
 *
 * @ingroup GvViewerGui
 *
 * This class is used to ...
 */
class GVVIEWERGUI_EXPORT GvvCacheUsageView : public QWidget, public Ui::GvvQCacheUsageWidget
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
	GvvCacheUsageView( QWidget* pParent, const char* pName = 0 );

	/**
	 * Default destructor
	 */
	virtual ~GvvCacheUsageView();

	/**
	 * Update view
	 *
	 * @param pNodeCacheUsage node cache usage (%)
	 * @param pBrickCacheUsage brick cache usage (%)
	 */
	void update( const GvViewerCore::GvvPipelineInterface* pPipeline );

	/********************************* SLOTS **********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Nodes cache usage
	 */
	QwtThermo* _nodesCacheUsage;

	/**
	 * Bricks cache usage
	 */
	QwtThermo* _bricksCacheUsage;

	/**
	 * Tree node sparseness
	 */
	QwtThermo* _treeNodeSparseness;

	/**
	 * Tree brick sparseness
	 */
	QwtThermo* _treeBrickSparseness;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when tree data structure monitoring's state has changed
	 */
	void on__treeMonitoringGroupBox_toggled( bool pChecked );

};

} // namespace GvViewerGui

#endif
