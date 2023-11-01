/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

	///**
	// * Slot called when noise first frequency value has changed
	// */
	//void on__noiseFirstFrequencySpinBox_valueChanged( double value );

	///**
	// * Slot called when noise strength value has changed
	// */
	//void on__noiseStrengthSpinBox_valueChanged( double value );
	
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

	///**
	// * Slot called when shader's voxel size multiplier value has changed
	// */
	//void on__voxelSizeMultiplierDoubleSpinBox_valueChanged( double value );

	/**
	 * Slot called when proxy geometry optimization value has changed
	 */
	void on__proxyGeometryOptimisationCheckBox_toggled( bool pChecked );

};

#endif
