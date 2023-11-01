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

#ifndef GVVBROWSERITEM_H
#define GVVBROWSERITEM_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QTreeWidgetItem>

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

namespace GvViewerGui
{

/**
 * This class represents the abstract base class for a browsable item.
 *
 * @ingroup	GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvBrowserItem : public QTreeWidgetItem
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvBrowserItem( GvViewerCore::GvvBrowsable* pBrowsable );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvBrowserItem();

	/**
	 * Returns the Browsable assigned to this item
	 *
	 * @return the Browsable assigned to this item
	 */
	const GvViewerCore::GvvBrowsable* getBrowsable() const;

	/**
	 * Returns the Browsable assigned to this item
	 *
	 * @return the Browsable assigned to this item
	 */
	GvViewerCore::GvvBrowsable* editBrowsable();

	/**
	 * Finds the item assigned to the given browsable
	 *
	 * @param pBrowsable specifies the browsable to be searched
	 *
	 * @return the corresponding item
	 */
	GvvBrowserItem* find( GvViewerCore::GvvBrowsable* pBrowsable );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The browsable contained by this item
	 */
	GvViewerCore::GvvBrowsable* mBrowsable;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden
	 */
	GvvBrowserItem( const GvvBrowserItem& );

	/**
	 * Copy operator forbidden
	 */
	GvvBrowserItem& operator=( const GvvBrowserItem& );

};

} // namespace GvViewerGui

#endif
