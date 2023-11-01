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

#ifndef CUSTOMSECTIONEDITOR_H
#define CUSTOMSECTIONEDITOR_H

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
 * @class GvvCacheEditor
 *
 * @brief The GvvCacheEditor class provides ...
 *
 * ...
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
	 */
	CustomSectionEditor( QWidget *parent = 0, Qt::WindowFlags flags = 0 );

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
	 * Slot called when global nb spheres value has changed
	 */
	void on__nbPointsSpinBox_valueChanged( int value );

	/**
	 * Slot called when user defined min level of resolution to handle value has changed
	 */
	void on__minLevelToHandleUserDefinedSpinBox_valueChanged( int value );

	/**
	 * Slot called when intersection type value has changed
	 */
	void on__intersectionTypeComboBox_currentIndexChanged( int pIndex );

	/**
	 * Slot called when global sphere radius fader value has changed
	 */
	void on__pointSizeFaderDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when the geometric criteria group box state has changed
	 */
	void on__geometricCriteriaGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when min nb of spheres per brick value has changed
	 */
	void on__minNbPointsPerBrickSpinBox_valueChanged( int value );

	/**
	 * Slot called when Geometruc Criteria's global usage check box is toggled
	 */
	void on__geometricCriteriaGlobalUsageCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when the screen based criteria group box state has changed
	 */
	void on__apparentMinSizeCriteriaGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when global sphere radius fader value has changed
	 */
	void on__apparentMinSizeDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when the screen based criteria group box state has changed
	 */
	void on__apparentMaxSizeCriteriaGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when global sphere radius fader value has changed
	 */
	void on__apparentMaxSizeDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when shader's uniform color check box is toggled
	 */
	void on__shaderUseUniformColorCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when shader's uniform color tool button is released
	 */
	void on__shaderUniformColorToolButton_released();

	/**
	 * Slot called when shader's animation check box is toggled
	 */
	void on__shaderUseAnimationCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when shader's animation check box is toggled
	 */
	void on__shaderUseTextureCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when the 3D model file button has been clicked (released)
	 */
	void on__textureToolButton_released();

};

#endif
