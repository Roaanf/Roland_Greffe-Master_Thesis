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
	 * Slot called when the screen based criteria group box state has changed
	 */
	void on__screenBasedCriteriaGroupBox_toggled( bool pChecked );

	/**
     * Slot called when screen space stop criteria has changed
     */
    void on__screenBasedCriteriaCoeffDoubleSpinBox_valueChanged( double pValue );

	/**
     * Slot called when material's alpha correction coefficient has changed
     */
    void on__materialAlphaCorrectionCoefficientDoubleSpinBox_valueChanged( double pValue );

	/**
	 * Slot called when noise first frequency value has changed
	 */
	void on__noiseFirstFrequencySpinBox_valueChanged( double value );

	/**
	 * Slot called when noise strength value has changed
	 */
	void on__noiseStrengthSpinBox_valueChanged( double value );

	/**
     * Slot called when voxel size scale has changed
     */
    void on__voxelSizeScaleSpinBox_valueChanged( double pValue );

};

#endif
