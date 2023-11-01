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

#ifndef _GVV_PIPELINE_H_
#define _GVV_PIPELINE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvSceneConfig.h"
#include "GvvPipelineInterface.h"

// STL
#include <vector>

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
	class GvvMeshInterface;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerScene
{

/** 
 * @class GvvPipeline
 *
 * @brief The GvvPipeline class provides info on a device.
 *
 * ...
 */
class GVVIEWERSCENE_EXPORT GvvPipeline : public GvViewerCore::GvvPipelineInterface
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
	GvvPipeline();

	/**
	 * Destructor
	 */
	virtual ~GvvPipeline();

	/**
	 * Get the flag telling wheter or not it has meshes
	 *
	 * @return the flag telling wheter or not it has meshes
	 */
	virtual bool hasMesh() const;

	/**
	 * Add a mesh
	 *
	 * @param pMesh a mesh
	 */
	virtual void addMesh( GvViewerCore::GvvMeshInterface* pMesh );

	/**
	 * Remove a mesh
	 *
	 * @param pMesh a mesh
	 */
	virtual void removeMesh( GvViewerCore::GvvMeshInterface* pMesh );

	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual const GvViewerCore::GvvMeshInterface* getMesh( unsigned int pIndex ) const;
	
	/**
	 * Get the i-th mesh
	 *
	 * @param pIndex index of the mesh
	 *
	 * @return the i-th mesh
	 */
	virtual GvViewerCore::GvvMeshInterface* editMesh( unsigned int pIndex );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Programmable shader
	 */
#if defined _MSC_VER
#pragma warning( push )
#pragma warning( disable:4251 )
#endif
	std::vector< GvViewerCore::GvvMeshInterface* > _meshes;
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
	GvvPipeline( const GvvPipeline& );
	
	/**
	 * Copy operator forbidden.
	 */
	GvvPipeline& operator=( const GvvPipeline& );

};

} // namespace GvViewerScene

#endif // !_GVV_PIPELINE_H_
