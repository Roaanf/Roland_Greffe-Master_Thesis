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

#ifndef _CUSTOM_EDITOR_H_
#define _CUSTOM_EDITOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include <GvvPipelineEditor.h>

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

// Project
class CustomSectionEditor;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class CustomEditor
 *
 * @brief The CustomEditor class provides a custom editor to this GigaVoxels
 * pipeline effect.
 *
 * This editor has a static creator function used by the factory class "GvvEditorWindow"
 * to create the associated editor (@see GvvEditorWindow::registerEditorFactory())
 */
class CustomEditor : public GvViewerGui::GvvPipelineEditor
{
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Creator function
	 *
	 * @param pParent parent widget
	 * @param pBrowsable pipeline element from which the editor will be associated
	 *
	 * @return the editor associated to the GigaVoxels pipeline
	 */
	static GvvEditor* create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Destructor
	 */
	virtual ~CustomEditor();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The custom editor
	 */
	CustomSectionEditor* _customSectionEditor;
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor
	 *
	 * @param pParent parent widget
	 * @param pFlags the window flags
	 */
	CustomEditor( QWidget* pParent = 0, Qt::WindowFlags pFlags = 0 );
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

};

#endif
