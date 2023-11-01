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

#ifndef GVVBROWSABLEITEM_H
#define GVVBROWSABLEITEM_H

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
class GVVIEWERGUI_EXPORT GvvBrowsableItem : public QTreeWidgetItem
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
	GvvBrowsableItem( GvViewerCore::GvvBrowsable* pBrowsable );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvBrowsableItem();

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
	GvvBrowsableItem* find( GvViewerCore::GvvBrowsable* pBrowsable );

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
	GvvBrowsableItem( const GvvBrowsableItem& );

	/**
	 * Copy operator forbidden
	 */
	GvvBrowsableItem& operator=( const GvvBrowsableItem& );

};

} // namespace GvViewerGui

#endif
