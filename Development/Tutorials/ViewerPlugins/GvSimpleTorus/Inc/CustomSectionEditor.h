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

// Project
//#include "SampleCore.h"

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

	///**
	// * Return the SampleCore.
	// *
	// * @return SampleCore
	// */
	//SampleCore* getSampleCore();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when noise first frequency value has changed
	 */
	void on__noiseFirstFrequencySpinBox_valueChanged( int value );

	/**
	 * Slot called when noise strength value has changed
	 */
	void on__noiseStrengthSpinBox_valueChanged( double value );

	/**
	 * Slot called when noise type value has changed
	 */
	void on__noiseTypeComboBox_currentIndexChanged( int value );

	/**
	 * Slot called when light type has changed
	 */
	void on__lightTypeComboBox_currentIndexChanged( int value );

	/**
	 * Slot called when brightness value has changed
	 */
	void on__brightnessSpinBox_valueChanged( double value );

	/**
	 * Slot called when light position X value has changed
	 */
	void on__lightXDoubleSpinBox_valueChanged( double value );

	/**
	 * Slot called when light position Y value has changed
	 */
	void on__lightYDoubleSpinBox_valueChanged( double value );

	/**
	 * Slot called when light position Z value has changed
	 */
	void on__lightZDoubleSpinBox_valueChanged( double value );
};

#endif
