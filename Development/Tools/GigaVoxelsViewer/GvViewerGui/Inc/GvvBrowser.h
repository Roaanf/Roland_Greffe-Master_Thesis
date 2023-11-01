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

#ifndef GVVBROWSER_H
#define GVVBROWSER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QTreeWidget>
#include <QHash>

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
	class GvvContextMenu;
	class GvvBrowserItem;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * This class represents the abstract base class for all browsers. It manages
 * a map of contextual menus.
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvBrowser : public QTreeWidget
{

	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 * 
	 * @param pParent the parent widget
	 */
	GvvBrowser( QWidget* pParent );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvBrowser();
	
	/**
	 * Returns the contextual menu with the given name. If the menu does not 
	 * exist then the method creates it and adds it within the map.
	 *
	 * @param pName specifies the name of the menu to be access
	 *
	 * @return the corresponding context menu
	 */
	GvvContextMenu* getContextMenu( const QString& pName );

	/**
	 * Returns the contextual menu with the given name. If the menu does not 
	 * exist then the method creates it and adds it within the map.
	 *
	 * @param pName specifies the name of the menu to be access
	 * @param pPath specifies the sub menu hierarchy
	 *
	 * @return the corresponding context menu
	 */
	GvvContextMenu* getContextSubMenu( const QString& pName, const QStringList& pPath );

	/**
	 * Finds the item assigned to the given browsable
	 *
	 * @param pBrowsable specifies the browsable to be searched
	 *
	 * @return the corresponding item
	 */
	GvvBrowserItem* find( GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Finds the item assigned to the browsable with the given name
	 *
	 * @param pName specifies the name of the browsable to be searched
	 *
	 * @return the corresponding item
	 */
	GvvBrowserItem* find( const QString& pName );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/

	/**
	 * Shortcut definition
	 */
	typedef QHash< unsigned int, GvvContextMenu* > GvvContextMenuHash;

	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * The contextual menus map
	 */
	GvvContextMenuHash mContextMenus;

	/******************************** METHODS *********************************/
	
	/**
	 * Creates a default item for the given browsable
	 *
	 * @param pBrowsable specifies the browsable for which an item is required
	 *
	 * @return the corresponding item
	 */
	virtual GvvBrowserItem* createItem( GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Handles the context menu event
	 *
	 * @param pEvent the context menu event
	 */
	virtual void contextMenuEvent( QContextMenuEvent* pEvent );

	/********************************** SLOTS **********************************/

protected slots:

	/**
	 * Handles the itemClicked signal
	 *
	 * @param pItem the clicked item
	 * @param pColumn the item's column
	 */
	virtual void onItemClicked( QTreeWidgetItem* pItem, int pColumn );

	/**
	 * Handles the itemChanged signal
	 *
	 * @param pItem the changed item
	 * @param pColumn the item's column
	 */
	virtual void onItemChanged( QTreeWidgetItem* pItem, int pColumn );

	/**
	* Handles the currentitemChanged signal
	*
	* @param pCurrent the Current item
	* @param pPrevious the  pPrevious current item
 	 */
	virtual void onCurrentItemChanged( QTreeWidgetItem* pCurrent , QTreeWidgetItem* pPrevious);

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvBrowser( const GvvBrowser& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvBrowser& operator=( const GvvBrowser& );
	
};

} // namespace GvViewerGui

#endif
