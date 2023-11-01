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

#ifndef _GVV_CACHE_EDITOR_H_
#define _GVV_CACHE_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQCacheEditor.h"

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
class GVVIEWERGUI_EXPORT GvvCacheEditor : public GvvSectionEditor, public Ui::GvvQCacheEditor
{
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	///**
	// * ...
	// *
	// * @param pParent ...
	// * @param pBrowsable ...
	// *
	// * @return ...
	// */
	//static GvvEditor* create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable );

	///**
	// * Populate the widget with a pipeline
	// *
	// * @param pPipeline the pipeline
	// */
	//void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Default constructor
	 */
	GvvCacheEditor( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvCacheEditor();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Populates this editor with the specified browsable
	 *
	 * @param pBrowsable specifies the browsable to be edited
	 */
	virtual void populate( GvViewerCore::GvvBrowsable* pBrowsable );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when custom cache policy value has changed
	 */
	void on__preventReplacingUsedElementsCachePolicyCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when custom cache policy value has changed
	 */
	void on__smoothLoadingCachePolicyGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when number of node subdivision value has changed
	 */
	void on__nbSubdivisionsSpinBox_valueChanged( int i );

	/**
	  * Slot called when number of brick loads value has changed
	 */
	void on__nbLoadsSpinBox_valueChanged( int i );

	/**
	 * Slot called when custom cache policy value has changed
	 */
	void on__timeLimitGroupBox_toggled( bool pChecked );

	/**
	  * Slot called when the time limit value has changed
	 */
	void on__timeLimitDoubleSpinBox_valueChanged( double pValue );

};

} // namespace GvViewerGui

#endif
