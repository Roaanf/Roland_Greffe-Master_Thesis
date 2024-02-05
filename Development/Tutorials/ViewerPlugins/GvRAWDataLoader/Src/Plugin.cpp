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

// System
#include <cassert>
#include <iostream>

// GigaSpace
#include <GvUtils/GsPluginManager.h>

// STL
#include <sstream>

// Project
#include "SampleCore.h"
#include "CustomEditor.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineManager.h>
#include <GvvEditorWindow.h>
#include <GvvRawDataLoaderDialog.h>
#include <GvvPipelineInterfaceViewer.h>

// Qt
#include <QFileDialog>
#include <qdebug.h>

// GigaVoxels
#include <GsCompute/GsDeviceManager.h>
#include <GvCore/GsError.h>

// Cuda SDK
#include <helper_cuda.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigqVoxels
using namespace GvUtils;

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
extern "C" GVRAWDATALOADER_EXPORT GsPluginInterface* createPlugin( GsPluginManager& pManager )
{
    //return new GvMyPlugin( pManager );
	GvMyPlugin* plugin = new GvMyPlugin( pManager );
	assert( plugin != NULL );

	return plugin;
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvMyPlugin::GvMyPlugin( GsPluginManager& pManager )
:	mManager( pManager )
,	mName( "GvRAWDataLoaderPlugin" )
,	mExportName( "Format A" )
,	_pipeline( NULL )
{
	initialize();
}

/******************************************************************************
 * Default constructor
 ******************************************************************************/
GvMyPlugin::~GvMyPlugin()
{
	finalize();
}

/******************************************************************************
 *
 ******************************************************************************/
void GvMyPlugin::initialize()
{
	// Initialize GigaSpace
	Gs::initialize();
	GsCompute::GsDeviceManager::get().initialize();

	// Initialize CUDA with OpenGL Interoperability
	if ( ! GvViewerGui::GvvApplication::get().isGPUComputingInitialized() )
	{
		//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
		//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
		cudaSetDevice( gpuGetMaxGflopsDeviceId() );
		GV_CHECK_CUDA_ERROR( "cudaSetDevice" );
		
		GvViewerGui::GvvApplication::get().setGPUComputingInitialized( true );
	}

	// Open the RAW data loader dialog
	//
	// PROBLEM :
	// - le dialog semble provoquer un draw() sans qu'il y ait eu un resize donc un init du pipeline => crash OpenGL...
	GvvRawDataLoaderDialog* dataLoaderDialog = new GvvRawDataLoaderDialog( NULL );
	QString modelFilename;
	unsigned int modelResolution;
	unsigned int brickSize;
select: // Awful solution imo
	if ( dataLoaderDialog != NULL )
	{
		// Retrieve USER selections
		dataLoaderDialog->exec();
		if ( dataLoaderDialog->result() == QDialog::Accepted )
		{
			// Retrieve data from loader
			modelFilename = dataLoaderDialog->get3DModelFilename();
			
			if (false) {
				// TODO parsing of the .mhd
			}
			else {
				modelResolution = dataLoaderDialog->get3DModelResolution();
				brickSize = dataLoaderDialog->getBrickSize();
				
				std::cout << "Checking the values..." << std::endl;
				std::cout << "modelResolution: " << modelResolution << std::endl;
				std::cout << "brickSize: " << brickSize << std::endl;
			}
		}
		
		if (modelFilename == "") {
			goto select;
		}

		//dataLoaderDialog->hide();

		// Check if the selected file is an XML or a RAW
		qDebug() << modelFilename;

		// Clean resources
		delete dataLoaderDialog;
		dataLoaderDialog = NULL;
	}
	
	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->registerEditorFactory( SampleCore::cTypeName, &CustomEditor::create );

	// Create the GigaVoxels pipeline
	_pipeline = new SampleCore<GvCore::GsVec1D< 2 >,GvCore::GsVec1D< 16 >>();
	if ( _pipeline != NULL )
	{
		_pipeline->set3DModelFilename( modelFilename.toLatin1().constData() );
		_pipeline->set3DModelResolution( modelResolution );

		// TO DO
		// add true resolution too !!!
	}
	
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
void GvMyPlugin::finalize()
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
 * 
 ******************************************************************************/
const string& GvMyPlugin::getName()
{
    return mName;
}
