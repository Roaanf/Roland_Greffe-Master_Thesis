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

	/**
	 * Boolean used to indicate if the editor was initialized or not.
	 */
	bool _init;
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

private slots:

	/**
	 * Slot called when the fractal's power value has changed
	 */
	void on__powerSpinBox_valueChanged( int pValue );

	/**
	 * Slot called when the fractal's nb iterations value has changed
	 */
	void on__nbIterationSpinBox_valueChanged( int pValue );
	
	/**
	 * Slot called when the fractal's nb iterations mode has changed
	 */
	void on__useAdaptativeIterationCheckBox_toggled( bool pChecked );

	/**
	 * Slot called when the priority policy changes
	 */
	void on__priorityPolicyComboBox_currentIndexChanged( int index );

};

#endif
