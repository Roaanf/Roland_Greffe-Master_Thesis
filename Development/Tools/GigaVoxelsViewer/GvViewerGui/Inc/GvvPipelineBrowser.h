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

#ifndef GVVPIPELINEBROWSER_H
#define GVVPIPELINEBROWSER_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvBrowser.h"
#include "GvvPipelineManagerListener.h"

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
	class GvvPipelineInterface;
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
class GVVIEWERGUI_EXPORT GvvPipelineBrowser : public GvvBrowser, public GvViewerCore::GvvPipelineManagerListener
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
	GvvPipelineBrowser( QWidget* pParent );
	
	/**
	 * Destructor.
	 */
	virtual ~GvvPipelineBrowser();
		
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
	virtual void onPipelineAdded( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	virtual void onPipelineRemoved( GvViewerCore::GvvPipelineInterface* pPipeline );
	
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
	GvvPipelineBrowser( const GvvPipelineBrowser& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvPipelineBrowser& operator=( const GvvPipelineBrowser& );
	
};

} // namespace GvViewerGui

#endif
