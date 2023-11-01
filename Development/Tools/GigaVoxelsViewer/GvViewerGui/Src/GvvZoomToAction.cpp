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

#include "GvvZoomToAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "Gvv3DWindow.h"
#include "GvvPipelineInterfaceViewer.h"
#include "GvvPipelineInterface.h"
#include "GvvGLSceneInterface.h"
#include "GvvContextManager.h"
#include "GvvBrowsable.h"

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
const QString GvvZoomToAction::cName = "zoomTo";

/**
 * The default text assigned to the action
 */
const char* GvvZoomToAction::cDefaultText = QT_TR_NOOP( "Zoom To" );

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
GvvZoomToAction::GvvZoomToAction( const QString& pFileName, 
										const QString& pText, 
										const QString& pIconName,
										bool pIsToggled )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName, pIsToggled )
{
	//** Sets the status tip
	setStatusTip( qApp->translate( "GvvZoomToAction", "Zoom To" ) );
	//setShortcut( qApp->translate( "GvvZoomToAction", "Z" ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvZoomToAction::~GvvZoomToAction()
{
}

/******************************************************************************
 * Overwrites the execute action
 ******************************************************************************/
void GvvZoomToAction::execute()
{
	GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
	if ( browsable != NULL )
	{
		//** Zoom to element
		GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
		if ( pipeline != NULL )
		{
			// TO DO
			// ...
		}

		//** Zoom to element
		GvvGLSceneInterface* scene = dynamic_cast< GvvGLSceneInterface* >( browsable );
		if ( scene != NULL )
		{
			GvvApplication& application = GvvApplication::get();
			GvvMainWindow* mainWindow = application.getMainWindow();
			Gvv3DWindow* window3D = mainWindow->get3DWindow();
			GvvPipelineInterfaceViewer* pipelineViewer = window3D->getPipelineViewer();
			if ( pipelineViewer != NULL )
			{
				// Scene bounding box
				qglviewer::Vec bboxMin( scene->_minX, scene->_minY, scene->_minZ );
				qglviewer::Vec bboxMax( scene->_maxX, scene->_maxY, scene->_maxZ );

				// Modify scene radius
				const float sceneRadius = qglviewer::Vec( bboxMax - bboxMin ).norm();
				pipelineViewer->setSceneRadius( sceneRadius );
				
				// Fit to bounding box
				pipelineViewer->camera()->fitBoundingBox( bboxMin, bboxMax );
			}
		}
	}
}
