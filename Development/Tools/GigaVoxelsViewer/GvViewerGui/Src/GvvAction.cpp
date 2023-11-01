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

#include "GvvAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvActionManager.h"

// Qt
#include <QObject>
#include <QDir>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

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
 * Constructs an action.
 * 
 * @param	pName		specifies the name of the action.
 * @param	pText		specifies the descriptive text of the action.
 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
 *							Does nothing if the string is empty. A full file path can also be given.
 * @param	pIsToggled	specifies if the action is toggled (Not toggled by default)
 ******************************************************************************/
GvvAction::GvvAction( QObject* pParent, const QString& pName, const QString& pText, const QString& pIconName, bool pIsToggled )
:	QAction( pText, pParent )
,	mName()
{
	//** Set the action name
	setName( pName );

	//** Create the default signal/slot connection
	connect( this, SIGNAL( triggered( bool ) ), this, SLOT( execute() ) );

	//** If the action is toggled
	setCheckable( pIsToggled );
	
	//** Define the path of the icon to be assigned
	QString iconName = pIconName;
	if ( iconName.isEmpty() )
	{
		iconName = pName + ".png";
	}

	//** Assign the icon
	QString iconRepository = GvUtils::GsEnvironment::getSystemDir( GvUtils::GsEnvironment::eResourcesDir ).c_str();
	QString iconfilename = iconRepository;
	iconfilename += QDir::separator();
	iconfilename += QString( "Icons" );
	iconfilename += QDir::separator();
	iconfilename += iconName;
	QIcon* icon = new QIcon( iconfilename );
	//QIcon* icon = NULL;
	if ( icon != NULL )
	{	
		setIcon( *icon );
	}

	//** Register this action 
	GvvActionManager::get().registerAction( this );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAction::~GvvAction()
{
	//** Unregisters it 
	GvvActionManager::get().unregisterAction( this );
}

/******************************************************************************
 * Returns the name of this action.
 *
 * @return the name of this action.
 ******************************************************************************/
const QString& GvvAction::getName() const
{
	return mName;
}

/******************************************************************************
 * Returns the name of this action.
 *
 * @return the name of this action.
 ******************************************************************************/
void GvvAction::setName( const QString& pName )
{
	setObjectName( pName );
	mName = pName;
}

/******************************************************************************
 * Updates this action before being shown
 ******************************************************************************/
void GvvAction::onAboutToShow()
{
}

/******************************************************************************
 * Executes this action.
 ******************************************************************************/
void GvvAction::execute()
{
}
