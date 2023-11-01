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

#ifndef _GVV_RENDERER_EDITOR_H_
#define _GVV_RENDERER_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQRendererEditor.h"

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
class GVVIEWERGUI_EXPORT GvvRendererEditor : public GvvSectionEditor, public Ui::GvvQRendererEditor
{
	// Qt Macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor
	 */
	GvvRendererEditor( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvRendererEditor();

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
	 * Slot called when max depth value has changed
	 */
	void on__maxDepthSpinBox_valueChanged( int i );

	/**
	 * Slot called when cache policy value has changed (dynamic update)
	 */
	void on__dynamicUpdateCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when the renderer request priority strategy has changed
	 */
	void on__priorityOnBricksRadioButton_toggled( bool pChecked );

	// ---- Viewport / Graphics buffer size ----

	/**
	 * Slot called when image downscaling mode value has changed
	 */
	void on__viewportOffscreenSizeGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when graphics buffer width value has changed
	 */
	void on__graphicsBufferWidthSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when graphics buffer height value has changed
	 */
	void on__graphicsBufferHeightSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when the viewer has been resized
	 *
	 * @param pWidth new viewer width
	 * @param pHeight new viewr height
	 */
	void onViewerResized( int pWidth, int pHeight );

	// ---- Time budget monitoring ----

	/**
	 * Slot called when time budget monitoring state value has changed
	 */
	void on__timeBudgetParametersGroupBox_toggled( bool pChecked );
	
	/**
	 * Slot called when time budget value has changed
	 */
	void on__timeBudgetSpinBox_valueChanged( int pValue );

};

} // namespace GvViewerGui

#endif
