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

#include "Plugin.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "PluginConfig.h"
#include "CustomEditor.h"

// System
#include <cassert>
#include <iostream>

#include <GvvPluginManager.h>

// STL
//#include <sstream>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Project
#include "SampleCore.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineManager.h>
#include <GvvEditorWindow.h>
#include <GvvPipelineInterfaceViewer.h>

// GigaVoxels
#include <GsCompute/GsDeviceManager.h>
#include <GvCore/GsError.h>

// Cuda SDK
#include <helper_cuda.h>
	
/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// VtViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

// STL
using namespace std;

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
 * 
 ******************************************************************************/
extern "C" GVCOLLISION_EXPORT GvvPluginInterface* createPlugin( GvvPluginManager& pManager )
{
    //return new Plugin( pManager );
	Plugin* plugin = new Plugin( pManager );
	assert( plugin != NULL );

	return plugin;
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
Plugin::Plugin( GvvPluginManager& pManager )
:	_manager( pManager )
,	_name( "GvCollisionPlugin" )
,	_exportName( "Format A" )
,	_pipeline( NULL )
{
	initialize();
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
Plugin::~Plugin()
{
	finalize();
}

/******************************************************************************
 *
 ******************************************************************************/
void Plugin::initialize()
{
	// Initialize GigaSpace
	Gs::initialize();
	GsCompute::GsDeviceManager::get().initialize();

	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->registerEditorFactory( SampleCore::cTypeName, &CustomEditor::create );

	// Initialize CUDA with OpenGL Interoperability
	if ( ! GvViewerGui::GvvApplication::get().isGPUComputingInitialized() )
	{
		//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
		//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
		GV_CHECK_CUDA_ERROR( "cudaSetDevice" );

		GvViewerGui::GvvApplication::get().setGPUComputingInitialized( true );
	}

	// Create the GigaVoxels pipeline
	_pipeline = new SampleCore();
	assert( _pipeline != NULL );

	// Pipeline BEGIN
	if ( _pipeline != NULL )
	{
		assert( _pipeline != NULL );
		_pipeline->init();

		GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
		GvvPipelineInterfaceViewer* viewer = window3D->getPipelineViewer();
		_pipeline->resize( viewer->size().width(), viewer->size().height() );
	}

	// Tell the viewer that a new pipeline has been added
	GvvPipelineManager::get().addPipeline( _pipeline );
}

/******************************************************************************
 *
 ******************************************************************************/
void Plugin::finalize()
{
	// Tell the viewer that a pipeline is about to be removed
	GvvPipelineManager::get().removePipeline( _pipeline );

	// Destroy the pipeline
	delete _pipeline;
	_pipeline = NULL;

	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();
	
	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->unregisterEditorFactory( SampleCore::cTypeName );

	// Finalize GigaSpace
	GsCompute::GsDeviceManager::get().finalize();
	Gs::finalize();

	// CUDA tip: clean up to ensure correct profiling
	cudaError_t error = cudaDeviceReset();
	assert( error == cudaSuccess );
}

/******************************************************************************
 * Get the plugin name
 *
 * @return the plugin name
 ******************************************************************************/
const string& Plugin::getName()
{
    return _name;
}
