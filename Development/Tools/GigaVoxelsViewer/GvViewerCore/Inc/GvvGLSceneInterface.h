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

#ifndef _GVV_GL_SCENE_INTERFACE_H_
#define _GVV_GL_SCENE_INTERFACE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvCoreConfig.h"
#include "GvvBrowsable.h"

// Assimp
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

namespace GvViewerCore
{

/** 
 * @class GvvGLSceneInterface
 *
 * @brief The GvvGLSceneInterface class provides...
 *
 * ...
 */
class GVVIEWERCORE_EXPORT GvvGLSceneInterface : public GvvBrowsable
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Type name
	 */
	static const char* cTypeName;

	// Mesh bounds
	//
	// @todo add accessors and move to protected section
	float _minX;
	float _minY;
	float _minZ;
	float _maxX;
	float _maxY;
	float _maxZ;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	GvvGLSceneInterface();

	/**
	 * Destructor
	 */
	virtual ~GvvGLSceneInterface();

	/**
	 * ...
	 *
	 * @param pScene ...
	 */
	void setScene( const aiScene* pScene );

	/**
	 * Returns the type of this browsable. The type is used for retrieving
	 * the context menu or when requested or assigning an icon to the
	 * corresponding item
	 *
	 * @return the type name of this browsable
	 */
	virtual const char* getTypeName() const;

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Initialize the scene
	 */
	virtual void initialize();

	/**
	 * Finalize the scene
	 */
	virtual void finalize();

	/**
	 * Draw the scene
	 */
	virtual void draw();
	
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
	const aiScene* _scene;
	
	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	void recursive_render( const aiScene* pScene, const aiNode* pNode );

	/**
	 * ...
	 */
	void apply_material( const aiMaterial* mtl );

	/**
	 * ...
	 */
	static void color4_to_float4( const aiColor4D* c, float f[4] );

	/**
	 * ...
	 */
	static void set_float4( float f[4], float a, float b, float c, float d );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

} // namespace GvViewerCore

#endif // !GVVPIPELINEINTERFACE_H
