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

#ifndef GVVEDITOR_H
#define GVVEDITOR_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// STL
#include <vector>

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
	class GvvBrowsable;
}

namespace GvViewerGui
{
	class GvvSectionEditor;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvEditor
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Destructor.
	 */
	virtual ~GvvEditor();

	/**
	 * Populates this editor with the specified browsable
	 *
	 * @param pBrowsable specifies the browsable to be edited
	 */
	virtual void populate( GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Get the number of sections
	 *
	 * @return the number of sections
	 */
	unsigned int getNbSections() const;

	/**
	 * ...
	 */
	GvvSectionEditor* getSectionEditor( unsigned int pIndex );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The list of widgets
	 */ 
	std::vector< GvvSectionEditor* > _sectionEditors;
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvEditor( const GvvEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvEditor& operator=( const GvvEditor& );
	
};

} // namespace GvViewerGui

#endif
