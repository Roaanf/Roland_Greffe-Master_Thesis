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

#include "GvvContextMenu.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowser.h"
#include "GvvAction.h"

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 * Default constructor.
 ******************************************************************************/
GvvContextMenu::GvvContextMenu( GvvBrowser* pBrowser ) 
:	QMenu( pBrowser )
{
	//** Setups connections
	QObject::connect( this, SIGNAL( aboutToShow() ), this, SLOT( onAboutToShow() ) );
}

/******************************************************************************
 * Contructs a menu with the given title and parent
 * 
 * @param pTitle the title menu
 * @param pMenu the parent menu
 ******************************************************************************/
GvvContextMenu::GvvContextMenu( const QString& pTitle, GvvContextMenu* pMenu )
:	QMenu( pTitle, pMenu )
{
	//** Setups connections
	QObject::connect( this, SIGNAL( aboutToShow() ), this, SLOT( onAboutToShow() ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvContextMenu::~GvvContextMenu()
{
}

/******************************************************************************
 * Handles the aboutToShow signal
 ******************************************************************************/
void GvvContextMenu::onAboutToShow()
{
	//** Retrieves the list of actions within this context menu
	QList< QAction* > lActions = actions();

	//** Iterates though the actions and update them
	for ( int i = 0; i < lActions.size(); ++i )
	{
		GvvAction* lAction = dynamic_cast< GvvAction* >( lActions[ i ] );
		if
			( lAction != NULL )
		{
			lAction->onAboutToShow();
		}
	}
}
