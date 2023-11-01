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

#ifndef GVVPIPELINEEDITOR_H
#define GVVPIPELINEEDITOR_H

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"
#include "GvvEditor.h"
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
}

namespace GvViewerGui
{
	class GvvCacheEditor;
	class GvvTransformationEditor;
	class GvvRendererEditor;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/** 
 * ...
 *
 * @ingroup GvViewerGui
 */
class GVVIEWERGUI_EXPORT GvvPipelineEditor : public GvvEditor, public GvViewerCore::GvvPipelineManagerListener
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * @param pParent ...
	 * @param pBrowsable ...
	 *
	 * @return ...
	 */
	static GvvEditor* create( QWidget* pParent, GvViewerCore::GvvBrowsable* pBrowsable );

	/**
	 * Destructor.
	 */
	virtual ~GvvPipelineEditor();
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The cache editor
	 */
	GvvCacheEditor* _cacheEditor;

	/**
	 * The transformation editor
	 */
	GvvTransformationEditor* _transformationEditor;

	/**
	 * The renderer editor
	 */
	GvvRendererEditor* _rendererEditor;

	/******************************** METHODS *********************************/

	/**
	 * Default constructor.
	 */
	GvvPipelineEditor( QWidget* pParent = NULL, Qt::WindowFlags pFlags = 0 );
	
	/**
	 * Tell that a pipeline has been modified.
	 *
	 * @param the modified pipeline
	 */
	virtual void onPipelineModified( GvViewerCore::GvvPipelineInterface* pPipeline );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/
	
	/**
	 * Copy constructor forbidden.
	 */
	GvvPipelineEditor( const GvvPipelineEditor& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvPipelineEditor& operator=( const GvvPipelineEditor& );
	
};

} // namespace GvViewerGui

#endif
