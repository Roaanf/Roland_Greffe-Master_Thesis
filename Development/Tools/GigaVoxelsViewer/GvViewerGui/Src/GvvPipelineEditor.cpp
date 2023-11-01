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

#include "GvvPipelineEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterface.h"
#include "GvvCacheEditor.h"
#include "GvvTransformationEditor.h"
#include "GvvRendererEditor.h"

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
 * ...
 *
 * @param pParent ...
 * @param pBrowsable ...
 *
 * @return ...
 ******************************************************************************/
GvvEditor* GvvPipelineEditor::create( QWidget* pParent, GvvBrowsable* pBrowsable )
{
	return new GvvPipelineEditor( pParent );
}

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvPipelineEditor::GvvPipelineEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	GvvEditor( pParent, pFlags )
,	GvvPipelineManagerListener()
,	_cacheEditor( NULL )
,	_transformationEditor( NULL )
,	_rendererEditor( NULL )
{
	// Data Structure / Cache editor
	_cacheEditor = new GvvCacheEditor( pParent, pFlags );
	_sectionEditors.push_back( _cacheEditor );

	// Renderer editor
	_rendererEditor = new GvvRendererEditor( pParent, pFlags );
	_sectionEditors.push_back( _rendererEditor );

	// Transformation editor
	_transformationEditor = new GvvTransformationEditor( pParent, pFlags );
	_sectionEditors.push_back( _transformationEditor );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvPipelineEditor::~GvvPipelineEditor()
{
}

/******************************************************************************
 * Remove a pipeline has been modified.
 *
 * @param the modified pipeline
 ******************************************************************************/
void GvvPipelineEditor::onPipelineModified( GvvPipelineInterface* pPipeline )
{
	populate( pPipeline );
}
