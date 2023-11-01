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
     * Slot called when nb spheres by brick min level produced value has changed
	 */
    void on__nbSpheresSpinBox_valueChanged( int value );

    /**
     * Slot called when global nb spheres value has changed
     */
    void on__nbSpheresTotalSpinBox_valueChanged( int pValue );

    /**
     * Slot called when regenerate positions button is pressed
     */
    void on__regeneratePositions_pressed();
    void on__regeneratePositionsBis_pressed();
	
	/**
	 * Slot called when user defined min level of resolution to handle radio button is toggled
	 */
	void on__minLevelToHandleUserDefinedRadioButton_toggled( bool pChecked );

	/**
	 * Slot called when user defined min level of resolution to handle value has changed
	 */
	void on__minLevelToHandleUserDefinedSpinBox_valueChanged( int value );

	/**
	 * Slot called when automatic min level of resolution to handle radio button is toggled
	 */
	void on__minLevelToHandleAutomaticRadioButton_toggled( bool pChecked );

	/**
	 * Slot called when intersection type value has changed
	 */
	void on__intersectionTypeComboBox_currentIndexChanged( int pIndex );

	/**
	 * Slot called when global sphere radius fader value has changed
	 */
	void on__sphereRadiusFaderDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when the geometric criteria group box state has changed
	 */
	void on__geometricCriteriaGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when min nb of spheres per brick value has changed
	 */
	void on__minNbSpheresPerBrickSpinBox_valueChanged( int value );

	/**
	 * Slot called when the screen based criteria group box state has changed
	 */
	void on__screenBasedCriteriaGroupBox_toggled( bool pChecked );

	/**
	 * Slot called when the absolute size criteria group box state has changed
	 */
	void on__absoluteSizeCriteriaGroupBox_toggled( bool pChecked );

    /**
     * Slot called when sphere diameter coefficient stop criteria has changed
     */
    void on__sphereDiameterCoeffSpinBox_valueChanged( double pValue );

    /**
     * Slot called when screen space stop criteria has changed
     */
    void on__shaderScreenSpaceCoeffSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when fixed size of spheres radio button is toggled
	 */
	void on__fixedSizeSphereRadiusRadioButton_toggled( bool pChecked );

	/**
	 * Slot called when fixed size radius of spheres value has changed
	 */
	void on__fixedSizeSphereRadiusDoubleSpinBox_valueChanged( double pValue );
	
	/**
	 * Slot called when mean size of spheres radio button is toggled
	 */
	void on__meanSizeOfSpheresRadioButton_toggled( bool pChecked );

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
     * Slot called when shader's blur sphere check box is toggled
     */
    void on__shaderUseBlurChekBox_toggled( bool pChecked );

    /**
     * Slot called when shader's fog check box is toggled
     */
    void on__shaderUseFogChekBox_toggled( bool pChecked );

    /**
     * Slot called when fog density value has changed
     */
    void on__shaderFogDensitydoubleSpinBox_valueChanged( double pValue );

    /**
     * Slot called when shader's fog color tool button is released
     */
    void on__shaderFogColorToolButton_released();

    /**
     * Slot called when shader's light source type check box is toggled
     */
    void on__shaderUseLightSourceCheckBox_toggled( bool pChecked );

    /**
     * Slot called when shading check box is toggled
     */
    void on__shaderUseShadingCheckBox_toggled( bool pChecked );

    /**
     * Slot called when shader bug correction check box is toggled
     */
    void on__ShaderUseBugCorrectionCheckBox_toggled( bool pChecked );

    /**
     * Slot called when shader sphere illumination coeff value is changed
     */
    void on__shaderIlluminationCoeffDoubleSpinBox_valueChanged( double pValue );

	/**
     * Slot called when number of reflection value is changed
     */
	void on__numberOfReflectionsSpinBox_valueChanged( int pValue );

};

#endif
