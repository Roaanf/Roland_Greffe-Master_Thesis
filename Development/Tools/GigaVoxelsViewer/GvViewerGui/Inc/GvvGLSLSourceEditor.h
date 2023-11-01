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

#ifndef _GVV_SHADER_SOURCE_EDITOR_
#define _GVV_SHADER_SOURCE_EDITOR_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "UI_GvQGLSLSourceEditor.h"

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/**
 * ...
 */
class GVVIEWERGUI_EXPORT GvvGLSLSourceEditor : public QWidget
{
	// Qt macro
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
	GvvGLSLSourceEditor( QWidget* pParent );

	/**
	 * ...
	 *
	 * @param ...
	 */
	void populate( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Current pipeline
	 */
	GvViewerCore::GvvPipelineInterface* _pipeline;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Ui designer class
	 */
	Ui::GvQGLSLSourceEditor _ui;

	/******************************** METHODS *********************************/

private slots:

	/**
	 * Light action
	 */
	void onReload();

	/**
	 * Open action
	 */
	void onApply();

	/**
	 * Slot called when current page index has changed
	 *
	 * @param pIndex ...
	 */
	void on_tabWidget_currentChanged( int pIndex );

	/**
	 * Slot called when apply button has been released
	 */
	void on__applyButton_released();
	
};

} // namespace GvViewerGui

#endif
