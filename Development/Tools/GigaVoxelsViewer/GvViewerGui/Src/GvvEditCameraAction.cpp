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

#include "GvvEditCameraAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"
#include "GvvCameraEditor.h"

// Qt
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;
	
/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/
/**
 * The unique name of the action
 */
const QString GvvEditCameraAction::cName = "editCamera";

/**
 * The default text assigned to the action
 */
const char* GvvEditCameraAction::cDefaultText = QT_TR_NOOP( "Edit Camera" );

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
GvvEditCameraAction::GvvEditCameraAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvEditCameraAction", "Edit Camera" ) );
	//setShortcut( qApp->translate( "GvvEditCameraAction", "C" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvEditCameraAction::~GvvEditCameraAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvEditCameraAction::execute()
{
	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	
	GvvCameraEditor* editor = mainWindow->getCameraEditor();
	if ( editor != NULL )
	{
		editor->show();
	}
}
