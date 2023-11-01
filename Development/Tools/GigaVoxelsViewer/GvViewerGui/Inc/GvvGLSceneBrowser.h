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
#ifndef _GVV_GL_SCENE_BROWSER_H_
#define _GVV_GL_SCENE_BROWSER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvBrowser.h"
#include "GvvGLSceneManagerListener.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GvViewer
namespace GvViewerCore
{
	class GvvBrowsable;
	class GvvGLSceneInterface;
}

namespace GvViewerGui
{
	class GvvContextMenu;
	class GvvBrowserItem;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * This class represents the abstract base class for all browsers. It manages
 * a map of contextual menus.
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvGLSceneBrowser : public GvvBrowser, public GvViewerCore::GvvGLSceneManagerListener
{
	// Qt macro
	Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Default constructor.
	 * 
	 * @param pParent the parent widget
	 */
	GvvGLSceneBrowser( QWidget* pParent );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvGLSceneBrowser();
		
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** TYPEDEFS ********************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	virtual void onGLSceneAdded( GvViewerCore::GvvGLSceneInterface* pScene );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	virtual void onGLSceneRemoved( GvViewerCore::GvvGLSceneInterface* pScene );
	
	/********************************** SLOTS **********************************/

protected slots:

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvGLSceneBrowser( const GvvGLSceneBrowser& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvGLSceneBrowser& operator=( const GvvGLSceneBrowser& );
	
};

} // namespace GvViewerGui

#endif
