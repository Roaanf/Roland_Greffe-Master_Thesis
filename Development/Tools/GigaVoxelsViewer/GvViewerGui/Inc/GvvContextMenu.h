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

#ifndef GVVCONTEXTMENU_H
#define GVVCONTEXTMENU_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QMenu>

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
namespace GvViewerGui
{
	class GvvBrowser;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * This class specializes a QMenu in order to handle the aboutToShow signal.
 * This allows to iterate throught the actions and update them according
 * the context
 *
 * @ingroup XBrowser
 */
class GVVIEWERGUI_EXPORT GvvContextMenu : public QMenu
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
	 * @param pBrowser the parent widget
	 */
	GvvContextMenu( GvvBrowser* pBrowser );
	
	/**
	 * Contructs a menu with the given title and parent
	 * 
	 * @param pTitle the title menu
	 * @param pMenu the parent menu
	 */
	GvvContextMenu( const QString& pTitle, GvvContextMenu* pMenu );

	/**
	 * Destructor.
	 */
	virtual ~GvvContextMenu();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/
	
	/********************************** SLOTS **********************************/

protected slots:

	/**
	 * Handles the aboutToShow signal
	 */
	void onAboutToShow();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvContextMenu( const GvvContextMenu& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvContextMenu& operator=( const GvvContextMenu& );
	
};

} // namespace GvViewerGui

#endif
