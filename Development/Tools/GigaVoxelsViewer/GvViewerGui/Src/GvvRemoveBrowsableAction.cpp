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

#include "GvvRemoveBrowsableAction.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvApplication.h"
#include "GvvMainWindow.h"
#include "GvvContextManager.h"
#include "GvvBrowsable.h"
#include "GvvPipelineInterface.h"
#include "GvvPipelineManager.h"
#include "GvvPluginManager.h"
#include "GvvGLSceneInterface.h"
#include "GvvGLSceneManager.h"

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
const QString GvvRemoveBrowsableAction::cName = "removeBrowsable";

/**
 * The default text assigned to the action
 */
const char* GvvRemoveBrowsableAction::cDefaultText = QT_TR_NOOP( "Remove" );

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 *******'***********************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructs an action.
 *
 * @param	pText specifies the descriptive text of this action
 * @param	pIconName specifies the name of the icon for this action located in the icons application path
 *					Does nothing if the string is empty. A full file path can also be given.
 ******************************************************************************/
GvvRemoveBrowsableAction::GvvRemoveBrowsableAction( const QString& pText, const QString& pIconName )
:	GvvAction( GvvApplication::get().getMainWindow(), cName, pText, pIconName )
{
	setStatusTip( qApp->translate("GvvRemoveBrowsableAction", cDefaultText ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvRemoveBrowsableAction::~GvvRemoveBrowsableAction()
{
}

/******************************************************************************
 * Overwrites the execute method
 ******************************************************************************/
void GvvRemoveBrowsableAction::execute()
{
	GvvBrowsable* browsable = GvvContextManager::get()->editCurrentBrowsable();
	if ( browsable != NULL )
	{
		//** Updates the context
		GvvContextManager::get()->setCurrentBrowsable( NULL );

		//** Remove the element
		GvvPipelineInterface* pipeline = dynamic_cast< GvvPipelineInterface* >( browsable );
		if ( pipeline != NULL )
		{
			GvvPipelineManager::get().removePipeline( pipeline );

			// Destroy the pipeline
			// TO DO ----------------------------
			//delete pipeline;
			//pipeline = NULL;
			GvvPluginManager::get().unloadAll();
			// ----------------------------------
		}
		else
		{
			//** Remove the element
			GvvGLSceneInterface* scene = dynamic_cast< GvvGLSceneInterface* >( browsable );
			if ( scene != NULL )
			{
				GvvGLSceneManager::get().removeGLScene( scene );

				// Destroy object
				delete scene;
				scene = NULL;
			}
		}
	}
}
