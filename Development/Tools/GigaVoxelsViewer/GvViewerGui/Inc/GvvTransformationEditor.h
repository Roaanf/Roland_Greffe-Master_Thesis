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

#ifndef _GVV_TRANSFORMATION_EDITOR_H_
#define _GVV_TRANSFORMATION_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvSectionEditor.h"
#include "UI_GvvQTransformationEditor.h"

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
 * @class GvvTransformationEditor
 *
 * @brief The GvvTransformationEditor class provides ...
 *
 * ...
 */
class GVVIEWERGUI_EXPORT GvvTransformationEditor : public GvvSectionEditor, public Ui::GvvQTransformationEditor
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
	GvvTransformationEditor( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~GvvTransformationEditor();

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
	void on__xTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zTranslationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__xRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__angleRotationSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__uniformScaleSpinBox_valueChanged( double pValue );
		
};

} // namespace GvViewerGui

#endif
