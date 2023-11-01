/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include <GvvPluginManager.h>

// STL
#include <sstream>

// OpenGL
#include <GL/glew.h>
#include <GL/freeglut.h>

// Project
#include "SampleCore.h"
#include "CustomEditor.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>
#include <Gvv3DWindow.h>
#include <GvvPipelineManager.h>
#include <GvvEditorWindow.h>
#include <GvvPipelineEditor.h>
#include <GvvDeviceInterface.h>
	
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
extern "C" GVRAYMAPGENERATOR_EXPORT GvvPluginInterface* createPlugin( GvvPluginManager& pManager )
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
:	mManager( pManager )
,	mName( "GvRayMapGeneratorPlugin" )
,	mExportName( "Format A" )
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
	GvViewerGui::GvvApplication& application = GvViewerGui::GvvApplication::get();
	GvViewerGui::GvvMainWindow* mainWindow = application.getMainWindow();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->registerEditorFactory( SampleCore::cTypeName, &CustomEditor::create );

	// Add the GigaVoxels pipeline in 3D view
	GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
	window3D->addViewer();	// deplacer cet appel via le "GvvPipelineManager::get().addPipeline( _pipeline )" !!!!!!!!!!!!!!!!

	// Create the GigaVoxels pipeline
	_pipeline = new SampleCore();

	// Tell the viewer that a new pipeline has been added
	GvvPipelineManager::get().addPipeline( _pipeline );

	//// TEST -----------------------------------------
	//GvCore::GvDeviceManager::get().initialize();
	//for ( int i = 0; i < GvCore::GvDeviceManager::get().getNbDevices(); i++ )
	//{
	//	const GvCore::GvDevice* device = GvCore::GvDeviceManager::get().getDevice( i );
	//	assert( device != NULL );
	//	if ( device != NULL )
	//	{
	//		GvvDeviceInterface* deviceInterface = new GvvDeviceInterface();

	//		deviceInterface->setDeviceName( device->_name );
	//		deviceInterface->setComputeCapabilityMajor( device->mProperties._computeCapabilityMajor );
	//		deviceInterface->setComputeCapabilityMinor( device->mProperties._computeCapabilityMinor );
	//	}
	//}
	//// TEST -----------------------------------------
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
	
	// deplacer cet appel via le "GvvPipelineManager::get().removePipeline( _pipeline )" !!!!!!!!!!!!!!!!
	GvViewerGui::Gvv3DWindow* window3D = mainWindow->get3DWindow();
	window3D->removeViewer();

	// Register custom editor's factory method
	GvViewerGui::GvvEditorWindow* editorWindow = mainWindow->getEditorWindow();
	editorWindow->unregisterEditorFactory( SampleCore::cTypeName );
}

/******************************************************************************
 * 
 ******************************************************************************/
const string& Plugin::getName()
{
    return mName;
}
