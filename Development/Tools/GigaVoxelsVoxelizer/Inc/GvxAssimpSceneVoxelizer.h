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

#ifndef _GVX_ASSIMP_SCENE_VOXELIZER_H_
#define _GVX_ASSIMP_SCENE_VOXELIZER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxSceneVoxelizer.h"

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace Gvx
{

/** 
 * @class GvxAssimpSceneVoxelizer
 *
 * @brief The GvxAssimpSceneVoxelizer class provides an implementation
 * of a scene voxelizer with ASSIMP, the Open Asset Import Library.
 *
 * It is used to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxAssimpSceneVoxelizer : public GvxSceneVoxelizer
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvxAssimpSceneVoxelizer();

	/**
	 * Destructor
	 */
	virtual ~GvxAssimpSceneVoxelizer();

	/**
	 * Normalize the scene.
	 * It determines the whole scene bounding box and then modifies vertices
	 * to scale the scene.
	 */
	virtual bool normalizeScene();

	/**
	 * Voxelize the scene
	 */
	virtual bool voxelizeScene();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * The root structure of the imported data. 
	 * 
	 *  Everything that was imported from the given file can be accessed from here.
	 *  Objects of this class are generally maintained and owned by Assimp, not
	 *  by the caller. You shouldn't want to instance it, nor should you ever try to
	 *  delete a given scene on your own.
	 */
	aiScene* _scene;
	
	/**
	 * Represents a log stream.
	 * A log stream receives all log messages and streams them _somewhere_.
	 */
	aiLogStream _logStream;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool loadScene();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxAssimpSceneVoxelizer( const GvxAssimpSceneVoxelizer& );

	/**
	 * Copy operator forbidden.
	 */
	GvxAssimpSceneVoxelizer& operator=( const GvxAssimpSceneVoxelizer& );

};

}

#endif
