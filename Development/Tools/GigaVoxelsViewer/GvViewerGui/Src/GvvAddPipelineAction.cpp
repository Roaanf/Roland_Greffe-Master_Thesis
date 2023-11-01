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

#include "GvvAddPipelineAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvPluginManager.h"
#include "GvvEnvironment.h"

// Qt
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QProcess>
#include <QDesktopServices>
#include <QUrl>
#include <QMessageBox>

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
const QString GvvAddPipelineAction::cName = "addPipeline";

/**
 * The default text assigned to the action
 */
const char* GvvAddPipelineAction::cDefaultText = QT_TR_NOOP( "Add Pipeline" );

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
GvvAddPipelineAction::GvvAddPipelineAction( const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvAddPipelineAction", "Add Pipeline" ) );
	setShortcut( qApp->translate( "GvvAddPipelineAction", "P" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvAddPipelineAction::~GvvAddPipelineAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvAddPipelineAction::execute()
{
	QString defaultDirectory = GvvEnvironment::getDemoPath().c_str();
	QString fileName = QFileDialog::getOpenFileName( GvvApplication::get().getMainWindow(), "Choose a file", defaultDirectory, tr( "GigaVoxels Pipeline Files (*.gvp)" ) );
	if ( ! fileName.isEmpty() )
	{
		GvvPluginManager::get().unloadAll();
		GvvPluginManager::get().loadPlugin( fileName.toStdString() );
	}
}
