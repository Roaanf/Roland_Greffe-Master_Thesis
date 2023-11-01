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

#include "GvvGLSceneBrowser.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvBrowserItem.h"
#include "GvvBrowsable.h"
#include "GvvContextMenu.h"
#include "GvvGLSceneInterface.h"

// Qt
#include <QContextMenuEvent>
#include <QTreeWidget>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
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
GvvGLSceneBrowser::GvvGLSceneBrowser( QWidget* pParent ) 
:	GvvBrowser( pParent )
,	GvvGLSceneManagerListener()
{
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvGLSceneBrowser::~GvvGLSceneBrowser()
{
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvGLSceneBrowser::onGLSceneAdded( GvvGLSceneInterface* pScene )
{
	assert( pScene != NULL);
	if ( pScene != NULL )
	{
		GvvBrowserItem* sceneItem = createItem( pScene );
		addTopLevelItem( sceneItem );

		// Expand item
		expandItem( sceneItem );
	}
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvGLSceneBrowser::onGLSceneRemoved( GvvGLSceneInterface* pScene )
{
	// Finds the item assigned to the given browsable
	GvvBrowserItem* item = find( pScene );
	if ( item != NULL )
	{
		int index = indexOfTopLevelItem( item );
		takeTopLevelItem( index );
	}
}
