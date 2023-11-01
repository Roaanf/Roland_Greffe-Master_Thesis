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

#ifndef _GVV_MESH_H_
#define _GVV_MESH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSceneConfig.h"
#include "GvvMeshInterface.h"

// STL
#include <vector>

// glm
#include <glm/glm.hpp>

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
	class GvvProgrammableShaderInterface;
}
namespace GvViewerScene
{
	class GvvGraphicsObject;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerScene
{

/** 
 * @class GvvMesh
 *
 * @brief The GvvMesh class provides info on a device.
 *
 * ...
 */
class GVVIEWERSCENE_EXPORT GvvMesh : public GvViewerCore::GvvMeshInterface
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
	GvvMesh();

	/**
	 * Destructor
	 */
	virtual ~GvvMesh();

	/**
	 * Get the flag telling wheter or not it has programmable shaders
	 *
	 * @return the flag telling wheter or not it has programmable shaders
	 */
	virtual bool hasProgrammableShader() const;

	/**
	 * Add a programmable shader
	 */
	virtual void addProgrammableShader( GvViewerCore::GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual void removeProgrammableShader( GvViewerCore::GvvProgrammableShaderInterface* pShader );

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual const GvViewerCore::GvvProgrammableShaderInterface* getProgrammableShader( unsigned int pIndex ) const;

	/**
	 * Get the associated programmable shader
	 *
	 * @param pIndex shader index
	 *
	 * @return the associated programmable shader
	 */
	virtual GvViewerCore::GvvProgrammableShaderInterface* editProgrammableShader( unsigned int pIndex );

	/**
	 * Load 3D object/scene
	 *
	 * @param pFilename filename
	 *
	 * @return a flag telling wheter or not it succeeds
	 */
	bool load( const char* pFilename );

	/**
	 * This function is the specific implementation method called
	 * by the parent GvIRenderer::render() method during rendering.
	 *
	 * @param pModelMatrix the current model matrix
	 * @param pViewMatrix the current view matrix
	 * @param pProjectionMatrix the current projection matrix
	 * @param pViewport the viewport configuration
	 */
	virtual void render( const glm::mat4& pModelViewMatrix, const glm::mat4& pProjectionMatrix, const glm::uvec4& pViewport );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Graphics object
	 */
	GvViewerScene::GvvGraphicsObject* _graphicsObject;

	/**
	 * Programmable shader
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvViewerCore::GvvProgrammableShaderInterface* > _programmableShaders;
#if defined _MSC_VER
#pragma warning( pop )
#endif

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvvMesh( const GvvMesh& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvMesh& operator=( const GvvMesh& );

};

} // namespace GvViewerScene

#endif // !_GVV_MESH_H_
