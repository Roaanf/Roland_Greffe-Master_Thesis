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

#include "GvvGLSceneManager.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGLSceneInterface.h"
#include "GvvGLSceneManagerListener.h"

// System
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>

// STL
#include <algorithm>

// Assimp
#include <assimp/postprocess.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * The unique instance of the singleton.
 */
GvvGLSceneManager* GvvGLSceneManager::msInstance = NULL;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Get the unique instance.
 *
 * @return the unique instance
 ******************************************************************************/
GvvGLSceneManager& GvvGLSceneManager::get()
{
    if ( msInstance == NULL )
    {
        msInstance = new GvvGLSceneManager();
    }

    return *msInstance;
}

/******************************************************************************
 * Constructor
 ******************************************************************************/
GvvGLSceneManager::GvvGLSceneManager()
:	_scenes()
,	_listeners()
,	_scene( NULL )
{
	// Attach stdout to the logging system.
	// Get one of the predefine log streams. This is the quick'n'easy solution to 
	// access Assimp's log system. Attaching a log stream can slightly reduce Assimp's
	// overall import performance.
	_logStream = aiGetPredefinedLogStream( aiDefaultLogStream_STDOUT, NULL );

	// Attach a custom log stream to the libraries' logging system.
	aiAttachLogStream( &_logStream );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvGLSceneManager::~GvvGLSceneManager()
{
	//	If the call to aiImportFile() succeeds, the imported data is returned in an aiScene structure. 
	// The data is intended to be read-only, it stays property of the ASSIMP 
	// library and will be stable until aiReleaseImport() is called. After you're 
	// done with it, call aiReleaseImport() to free the resources associated with 
	// this file.
	aiReleaseImport( _scene );
	
	// Detach a custom log stream from the libraries' logging system.
	aiDetachLogStream( &_logStream );
}

/******************************************************************************
 * Load 3D scene from file
 *
 * @param pFilename file to load
 *
 * @return flag telling wheter or not loading has succeded
 ******************************************************************************/
//bool GvvGLSceneManager::load( const std::string& pFilename )
const aiScene* GvvGLSceneManager::load( const std::string& pFilename )
{
	//bool result = false;

	// Load the scene from the 3D model file.
	// Read the given file and returns its content.
	// Return a pointer to the imported data or NULL if the import failed.
	//string filename = string( getFilePath() + getFileName() + getFileExtension() );
	//_scene = const_cast< aiScene* >( aiImportFile( pFilename.data(), aiProcessPreset_TargetRealtime_Fast ) );
	_scene = aiImportFile( pFilename.data(), aiProcessPreset_TargetRealtime_Fast );
	// NOTE
	// The aiProcessPreset_TargetRealtime_Fast macro is an OR combination of the following flags :
	// - aiProcess_CalcTangentSpace
	// - aiProcess_GenNormals
	// - aiProcess_JoinIdenticalVertices
	// - aiProcess_Triangulate
	// - aiProcess_GenUVCoords
	// - aiProcess_SortByPType
				
	// Check import status
//	assert( _scene != NULL );
//	if ( _scene != NULL )
//	{
//		result = true;
//	}

	//return result;
	return _scene;
}

/******************************************************************************
 * Add a pipeline.
 *
 * @param the pipeline to add
 ******************************************************************************/
void GvvGLSceneManager::addGLScene( GvvGLSceneInterface* pScene )
{
	assert( pScene != NULL );
	if ( pScene != NULL )
	{
		// Add pipeline
		_scenes.push_back( pScene );

		// Inform listeners that a pipeline has been added
		vector< GvvGLSceneManagerListener* >::iterator it = _listeners.begin();
		for ( ; it != _listeners.end(); ++it )
		{
			GvvGLSceneManagerListener* listener = *it;
			if ( listener != NULL )
			{
				listener->onGLSceneAdded( pScene );
			}
		}
	}
}

/******************************************************************************
 * Remove a pipeline.
 *
 * @param the pipeline to remove
 ******************************************************************************/
void GvvGLSceneManager::removeGLScene( GvvGLSceneInterface* pScene )
{
	assert( pScene != NULL );
	if ( pScene != NULL )
	{
		vector< GvvGLSceneInterface * >::iterator itGLScene;
		itGLScene = find( _scenes.begin(), _scenes.end(), pScene );
		if ( itGLScene !=_scenes.end() )
		{
			// Remove pipeline
			_scenes.erase( itGLScene );

			// Inform listeners that a pipeline has been removed
			vector< GvvGLSceneManagerListener* >::iterator itListener = _listeners.begin();
			for ( ; itListener != _listeners.end(); ++itListener )
			{
				GvvGLSceneManagerListener* listener = *itListener;
				if ( listener != NULL )
				{
					listener->onGLSceneRemoved( pScene );
				}
			}
		}
	}
}

/******************************************************************************
 * Tell that a pipeline has been modified.
 *
 * @param the modified pipeline
 ******************************************************************************/
void GvvGLSceneManager::setModified( GvvGLSceneInterface* pScene )
{
	assert( pScene != NULL );
	if ( pScene != NULL )
	{
		// Add pipeline
		_scenes.push_back( pScene );

		// Inform listeners that a pipeline has been added
		vector< GvvGLSceneManagerListener* >::iterator it = _listeners.begin();
		for ( ; it != _listeners.end(); ++it )
		{
			GvvGLSceneManagerListener* listener = *it;
			if ( listener != NULL )
			{
				listener->onGLSceneModified( pScene );
			}
		}
	}
}

/******************************************************************************
 * Register a listener.
 *
 * @param pListener the listener to register
 ******************************************************************************/
void GvvGLSceneManager::registerListener( GvvGLSceneManagerListener* pListener )
{
	assert( pListener != NULL );
	if ( pListener != NULL )
	{
		// Add listener
		_listeners.push_back( pListener );
	}
}

/******************************************************************************
 * Unregister a listener.
 *
 * @param pListener the listener to unregister
 ******************************************************************************/
void GvvGLSceneManager::unregisterListener( GvvGLSceneManagerListener* pListener )
{
	assert( pListener != NULL );
	if ( pListener != NULL )
	{
		vector< GvvGLSceneManagerListener * >::iterator it;
		it = find( _listeners.begin(), _listeners.end(), pListener );
		if ( it != _listeners.end() )
		{
			// Remove pipeline
			_listeners.erase( it );
		}
	}
}
