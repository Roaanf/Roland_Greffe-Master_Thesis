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

#include "GvvMesh.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvProgrammableShaderInterface.h"
#include "GvvGraphicsObject.h"

// STL
#include <algorithm>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerScene;

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
 * Constructor
 ******************************************************************************/
GvvMesh::GvvMesh()
:	GvvMeshInterface()
,	_graphicsObject( NULL )
,	_programmableShaders()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
GvvMesh::~GvvMesh()
{
	// TO DO

	// Release resources
	delete _graphicsObject;
	_graphicsObject = NULL;
}

/******************************************************************************
 * Get the flag telling wheter or not it has programmable shaders
 *
 * @return the flag telling wheter or not it has programmable shaders
 ******************************************************************************/
bool GvvMesh::hasProgrammableShader() const
{
	return ( _programmableShaders.size() > 0 );
}

/******************************************************************************
 * Add a programmable shader
 ******************************************************************************/
void GvvMesh::addProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
	// TO DO : check in already there ?

	_programmableShaders.push_back( pShader );
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
void GvvMesh::removeProgrammableShader( GvvProgrammableShaderInterface* pShader )
{
	vector< GvvProgrammableShaderInterface* >::iterator itShader;
	itShader = find( _programmableShaders.begin(), _programmableShaders.end(), pShader );
	if ( itShader != _programmableShaders.end() )
	{
		// Remove pipeline
		_programmableShaders.erase( itShader );
	}
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
const GvvProgrammableShaderInterface* GvvMesh::getProgrammableShader( unsigned int pIndex ) const
{
	assert( pIndex < _programmableShaders.size() );
	if ( pIndex < _programmableShaders.size() )
	{
		return _programmableShaders[ pIndex ];
	}

	return NULL;
}

/******************************************************************************
 * Get the associated programmable shader
 *
 * @param pIndex shader index
 *
 * @return the associated programmable shader
 ******************************************************************************/
GvvProgrammableShaderInterface* GvvMesh::editProgrammableShader( unsigned int pIndex )
{
	assert( pIndex < _programmableShaders.size() );
	if ( pIndex < _programmableShaders.size() )
	{
		return _programmableShaders[ pIndex ];
	}

	return NULL;
}

/******************************************************************************
 * Load 3D object/scene
 *
 * @param pFilename filename
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool GvvMesh::load( const char* pFilename )
{
	assert( pFilename != NULL );

	// TODO : use ResourceManager::get()

	// Release resources
	delete _graphicsObject;
	_graphicsObject = NULL;

	// Create graphics object
	_graphicsObject = new GvvGraphicsObject();
	_graphicsObject->initialize();
	_graphicsObject->load( pFilename );

	return false;
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GvIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
void GvvMesh::render( const glm::mat4& pModelViewMatrix, const glm::mat4& pProjectionMatrix, const glm::uvec4& pViewport )
{
	_graphicsObject->render( pModelViewMatrix, pProjectionMatrix, pViewport );
}
