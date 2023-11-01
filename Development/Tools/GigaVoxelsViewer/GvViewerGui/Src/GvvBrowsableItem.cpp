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

#include "GvvBrowserItem.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowsable.h"

// Qt
#include <QHash>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an item with the given browsable
 *
 * @param pBrowsable specifies the browsable to be assigned to this item 
 ******************************************************************************/
GvvBrowserItem::GvvBrowserItem( GvvBrowsable* pBrowsable )
//:	QTreeWidgetItem( qHash( pBrowsable->getTypeName() ) )
:	QTreeWidgetItem( qHash( QString( pBrowsable->getTypeName() ) ) )
,	mBrowsable( pBrowsable )
{
	//-----------------------------------------------------------------------------
	// TO DO :
	// - for QTreeWidgetItem( qHash( QString( pBrowsable->getTypeName() ) ) )
	// - check if the problem comes from the int instead of the uint
	//-----------------------------------------------------------------------------
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvBrowserItem::~GvvBrowserItem()
{
	mBrowsable = NULL;
}

/******************************************************************************
 * Returns the Browsable assigned to this item
 *
 * @return the Browsable assigned to this item
 ******************************************************************************/
const GvvBrowsable* GvvBrowserItem::getBrowsable() const
{
	return mBrowsable;
}

/******************************************************************************
 * Returns the Browsable assigned to this item
 *
 * @return the Browsable assigned to this item
 ******************************************************************************/
GvvBrowsable* GvvBrowserItem::editBrowsable()
{
	return mBrowsable;
}

/******************************************************************************
 * Finds the item assigned to the given browsable
 *
 * @param pBrowsable specifies the browsable to be searched
 *
 * @return the corresponding item
 ******************************************************************************/
GvvBrowserItem* GvvBrowserItem::find( GvvBrowsable* pBrowsable )
{
	//** Checks whether this item holds the given element
	if ( mBrowsable == pBrowsable )
	{
		return this;
	}

	//** Iterates through the children and performs a recursive search
	for	( int i = 0; i < childCount(); ++i )
	{
		GvvBrowserItem* lChildItem = dynamic_cast< GvvBrowserItem* >( child( i ) );
		if ( lChildItem != NULL )
		{
			GvvBrowserItem* lFoundItem = lChildItem->find( pBrowsable );
			if ( lFoundItem != NULL )
			{
				return lFoundItem;
			}
		}
	}

	return NULL;
}
