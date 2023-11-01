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

#include "GvvDisplayPerformanceCountersAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"

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
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/
/**
 * The unique name of the action
 */
const QString GvvDisplayPerformanceCountersAction::cName = "displayPerformanceCounters";

/**
 * The default text assigned to the action
 */
const char* GvvDisplayPerformanceCountersAction::cDefaultText = QT_TR_NOOP( "Display Performance Counters" );

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
GvvDisplayPerformanceCountersAction::GvvDisplayPerformanceCountersAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvDisplayPerformanceCountersAction", "Display Performance Counters" ) );
	setShortcut( qApp->translate( "GvvDisplayPerformanceCountersAction", "P" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvDisplayPerformanceCountersAction::~GvvDisplayPerformanceCountersAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvDisplayPerformanceCountersAction::execute()
{
	//QMessageBox::information( GvvApplication::get().getMainWindow(), tr( "Display Performance Counters" ), tr( "Not yet implemented..." ) );

	GvvApplication& application = GvvApplication::get();
	GvvMainWindow* mainWindow = application.getMainWindow();
	Gvv3DWindow* window3D = mainWindow->get3DWindow();
	GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
	if ( pipelineViewer != NULL )
	{
	GvvPipelineInterface* pipeline = pipelineViewer->editPipeline();
	if ( pipeline != NULL )
	{
		unsigned int performanceCountersMode = 0;
		if ( isChecked() )
		{
			performanceCountersMode = 1;	// Device mode
		}
		pipeline->togglePerfmonDisplay( performanceCountersMode );
	}
	}
}
