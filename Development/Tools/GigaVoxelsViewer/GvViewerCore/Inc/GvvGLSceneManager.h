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

#ifndef _GVV_GL_SCENE_MANAGER_H_
#define _GVV_GL_SCENE_MANAGER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"

// STL
#include <vector>
#include <string>

// Assimp
#include <assimp/cimport.h>
#include <assimp/scene.h>

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
	class GvvGLSceneInterface;
	class GvvGLSceneManagerListener;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerCore
{

/**
 * GvPluginManager
 */
class GVVIEWERCORE_EXPORT GvvGLSceneManager
{

	/**************************************************************************
     ***************************** PUBLIC SECTION *****************************
     **************************************************************************/

public:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /******************************** METHODS *********************************/

    /**
     * Get the unique instance.
     *
     * @return the unique instance
     */
    static GvvGLSceneManager& get();

	/**
	 * Load 3D scene from file
	 *
	 * @param pFilename file to load
	 *
	 * @return flag telling wheter or not loading has succeded
	 */
	//bool load( const std::string& pFilename );
	const aiScene* load( const std::string& pFilename );

	/**
	 * Add a pipeline.
	 *
	 * @param the pipeline to add
	 */
	void addGLScene( GvvGLSceneInterface* pGLScene );

	/**
	 * Remove a pipeline.
	 *
	 * @param the pipeline to remove
	 */
	void removeGLScene( GvvGLSceneInterface* pGLScene );

	/**
	 * Tell that a pipeline has been modified.
	 *
	 * @param the modified pipeline
	 */
	void setModified( GvvGLSceneInterface* pGLScene );

	/**
	 * Register a listener.
	 *
	 * @param pListener the listener to register
	 */
	void registerListener( GvvGLSceneManagerListener* pListener );

	/**
	 * Unregister a listener.
	 *
	 * @param pListener the listener to unregister
	 */
	void unregisterListener( GvvGLSceneManagerListener* pListener );

   /**************************************************************************
    **************************** PROTECTED SECTION ***************************
    **************************************************************************/

protected:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

	/**
     * List of pipelines
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< GvvGLSceneInterface* >_scenes;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/**
     * List of listeners
     */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
    std::vector< GvvGLSceneManagerListener* > _listeners;
#if defined _MSC_VER
#pragma warning( pop )
#endif
		
	/**
	 * The root structure of the imported data. 
	 * 
	 *  Everything that was imported from the given file can be accessed from here.
	 *  Objects of this class are generally maintained and owned by Assimp, not
	 *  by the caller. You shouldn't want to instance it, nor should you ever try to
	 *  delete a given scene on your own.
	 */
	const aiScene* _scene;
	
	/**
	 * Represents a log stream.
	 * A log stream receives all log messages and streams them _somewhere_.
	 */
	aiLogStream _logStream;

    /******************************** METHODS *********************************/

    /**************************************************************************
     ***************************** PRIVATE SECTION ****************************
     **************************************************************************/

private:

    /******************************* INNER TYPES *******************************/

    /******************************* ATTRIBUTES *******************************/

    /**
     * The unique instance
     */
    static GvvGLSceneManager* msInstance;

    /******************************** METHODS *********************************/

    /**
     * Constructor
     */
    GvvGLSceneManager();

	/**
     * Constructor
     */
    ~GvvGLSceneManager();

};

} // namespace GvViewerCore

#endif
