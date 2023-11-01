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

#include "GvvAddSceneAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvGLSceneManager.h"
#include "GvvGLSceneInterface.h"

// Qt
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

// GigaSpace
#include <GvUtils/GsEnvironment.h>

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
const QString GvvAddSceneAction::cName = "addScene";

/**
 * The default text assigned to the action
 */
const char* GvvAddSceneAction::cDefaultText = QT_TR_NOOP( "Add Scene" );

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
GvvAddSceneAction::GvvAddSceneAction( const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvAddSceneAction", "Add Scene" ) );
	//setShortcut( qApp->translate( "GvvAddSceneAction", "P" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAddSceneAction::~GvvAddSceneAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvAddSceneAction::execute()
{
	QString defaultDirectory = GvUtils::GsEnvironment::getDataDir( GvUtils::GsEnvironment::e3DModelsDir ).c_str();
	QString filename = QFileDialog::getOpenFileName( GvvApplication::get().getMainWindow(), "Choose a file", defaultDirectory, tr( "GL Scene Files (*.*)" ) );
	if ( ! filename.isEmpty() )
	{
		GvvGLSceneInterface* scene = new GvvGLSceneInterface();
		if ( scene != NULL )
		{
			scene->setScene( GvvGLSceneManager::get().load( filename.toStdString() ) );

			GvvGLSceneManager::get().addGLScene( scene );
		}
	}
}
