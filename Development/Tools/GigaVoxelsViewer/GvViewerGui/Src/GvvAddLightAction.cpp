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

#include "GvvAddLightAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"

// Qt
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/
/**
 * The unique name of the action
 */
const QString GvvAddLightAction::cName = "addLight";

/**
 * The default text assigned to the action
 */
const char* GvvAddLightAction::cDefaultText = QT_TR_NOOP( "Add Light" );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an action dependant of the applications project
 *
 * @param	pFileName	specifies the filename of the manual
 * @param	pText		specifies the descriptive text of this action
 * @param	pIconName	specifies the name of the icon for this action located in the icons application path
 *							Does nothing if the string is empty. A full file path can also be given.
 * @param	pIsToggled	specified if the action is toggled or not
 ******************************************************************************/
GvvAddLightAction::GvvAddLightAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
,	mFileName( pFileName )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvAddLightAction", "Add Light" ) );
	setShortcut( qApp->translate( "GvvAddLightAction", "L" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAddLightAction::~GvvAddLightAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvAddLightAction::execute()
{
	QMessageBox::information( GvvApplication::get().getMainWindow(), tr( "Add Light" ), tr( "Not yet implemented..." ) );

	// Open the GvvQLightEditor editor -> créer un dialog
	//GvvQLightEditor
}
