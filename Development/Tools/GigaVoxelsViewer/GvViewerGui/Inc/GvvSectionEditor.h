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

#ifndef GVVSECTIONEDITOR_H
#define GVVSECTIONEDITOR_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// STL
#include <vector>

// Qt
#include <QWidget>
#include <QString>

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
	class GvvEditor;
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
class GVVIEWERGUI_EXPORT GvvSectionEditor : public QWidget
{

	friend class GvvEditor;

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Get the name of the editor
	 */
	const QString& getName() const;

	/**
	 * Set the name of the editor
	 */
	void setName( const QString& pName );

	/**
	 * Populates this editor with the specified browsable
	 *
	 * @param pBrowsable specifies the browsable to be edited
	 */
	virtual void populate( GvViewerCore::GvvBrowsable* pBrowsable ) = 0;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * Name of the editor
	 */
	QString _name;

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvSectionEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvSectionEditor();
	
	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvSectionEditor( const GvvSectionEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvSectionEditor& operator=( const GvvSectionEditor& );
	
};

} // namespace GvViewerGui

#endif
