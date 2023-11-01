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

#ifndef _CUSTOM_SECTION_EDITOR_H_
#define _CUSTOM_SECTION_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include <GvvSectionEditor.h>

// Project
#include "UI_GvQCustomEditor.h"

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
	class GvvBrowsable;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class CustomSectionEditor
 *
 * @brief The CustomSectionEditor class provides the user custom editor to
 * this GigaVoxels pipeline effect.
 *
 * This editor is a child of the parent CustomEditor. It holds a user interface
 * to edit the GigaVoxels pipeline parameters.
 * It is a GvvSectionEditor, so it is reprenseted by a separate page of parameters
 * in the main viewer user interface.
 */
class CustomSectionEditor : public GvViewerGui::GvvSectionEditor, public Ui::GvQCustomEditor
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
	 *
	 * @param pParent parent widget
	 * @param pFlags the window flags
	 */
	CustomSectionEditor( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );

	/**
	 * Destructor
	 */
	virtual ~CustomSectionEditor();

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
	 * Slot called when the 3D model file button has been clicked (released)
	 */
	void on__3DModelToolButton_released();

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

	// Shadow Caster

	/**
	 * Slot called when the 3D model file button has been clicked (released)
	 */
	void on__3DModelToolButton_2_released();

	/**
	 * Slot called when max depth value has changed
	 */
	void on__xTranslationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yTranslationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zTranslationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__xRotationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__yRotationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__zRotationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__angleRotationSpinBox_2_valueChanged( double pValue );

	/**
	 * Slot called when max depth value has changed
	 */
	void on__uniformScaleSpinBox_2_valueChanged( double pValue );

};

#endif
