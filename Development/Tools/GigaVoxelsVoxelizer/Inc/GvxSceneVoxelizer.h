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

#ifndef _GVX_SCENE_VOXELIZER_H_
#define _GVX_SCENE_VOXELIZER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "GvxDataTypeHandler.h"
#include "GvxVoxelizerEngine.h"

// System
#include <string>

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
 * @class GvxSceneVoxelizer
 *
 * @brief The GvxSceneVoxelizer class provides an interface to a voxelize a scene.
 *
 * The main idea is to load a 3D scene in memory and traverse it to retrieve
 * useful data needed during voxelization (i.e. vertices, faces, normals,
 * materials, textures, etc...)
 */
class GvxSceneVoxelizer
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
	GvxSceneVoxelizer();

	/**
	 * Destructor
	 */
	virtual ~GvxSceneVoxelizer();

	/**
	 * The main method called to voxelize a scene.
	 * All settings must have been done previously (i.e. filename, path, etc...)
	 */
	virtual bool launchVoxelizationProcess();

	/**
	 * Get the data file path
	 *
	 * @return the data file path
	 */
	const std::string& getFilePath() const;

	/**
	 * Set the data file path
	 *
	 * @param pFilePath the data file path
	 */
	void setFilePath( const std::string& pFilePath );

	/**
	 * Get the data file name
	 *
	 * @return the data file name
	 */
	const std::string& getFileName() const;

	/**
	 * Set the data file name
	 *
	 * @param pFileName the data file name
	 */
	void setFileName( const std::string& pFileName );

	/**
	 * Get the data file extension
	 *
	 * @return the data file extension
	 */
	const std::string& getFileExtension() const;

	/**
	 * Set the data file extension
	 *
	 * @param pFileExtension the data file extension
	 */
	void setFileExtension( const std::string& pFileExtension );

	/**
	 * Get the max level of resolution
	 *
	 * @return the max level of resolution
	 */
	unsigned int getMaxResolution() const;

	/**
	 * Set the max level of resolution
	 *
	 * @param pValue the max level of resolution
	 */
	void setMaxResolution( unsigned int pValue );

	/**
	 * Tell wheter or not normals generation is activated
	 *
	 * @return a flag telling wheter or not normals generation is activated
	 */
	bool isGenerateNormalsOn() const;

	/**
	 * Set the flag telling wheter or not normals generation is activated
	 *
	 * @param pFlag the flag telling wheter or not normals generation is activated
	 */
	void setGenerateNormalsOn( bool pFlag );

	/**
	 * Get the brick width
	 *
	 * @return the brick width
	 */
	unsigned int getBrickWidth() const;

	/**
	 * Set the brick width
	 *
	 * @param pValue the brick width
	 */
	void setBrickWidth( unsigned int pValue );

	/**
	 * Get the data type of voxels
	 *
	 * @return the data type of voxels
	 */
	GvxDataTypeHandler::VoxelDataType getDataType() const;

	/**
	 * Set the data type of voxels
	 *
	 * @param pType the data type of voxels
	 */
	void setDataType( GvxDataTypeHandler::VoxelDataType pType );

	/**
	 * Set the filter type
	 *
	 * 0 = mean
	 * 1 = gaussian
	 * 2 = laplacian
	 */
	void setFilterType(int filterType);

	/**
	 * Set the number of application of the filter
	 */
	void setFilterIterations(int nbFilterOperation);

	/**
	 * Set whether or not we generate the normal field
	 */
	void setNormals ( bool normals);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * 3D model file path.
	 * Path must be terminated by the specific Operationg System directory seperator (/, \, //, etc...).
	 */
	std::string _filePath;
	
	/**
	 * 3D model file name
	 */
	std::string _fileName;
	
	/**
	 * 3D model file extension
	 */
	std::string _fileExtension;

	/**
	 * Max scene resolution
	 */
	unsigned int _maxResolution;

	/**
	 * Flag to tell wheter or not to generate normals
	 */
	bool _isGenerateNormalsOn;

	/**
	 * Brick width
	 */
	unsigned int _brickWidth;

	/**
	 * Data type
	 */
	GvxDataTypeHandler::VoxelDataType _dataType;

	/**
	 * Voxelizer engine
	 */
	GvxVoxelizerEngine _voxelizerEngine;
	
	/******************************** METHODS *********************************/

	/**
	 * Load/import the scene
	 */
	virtual bool loadScene();

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

	/**
	 * Apply the mip-mapping algorithmn.
	 * Given a pre-filtered voxel scene at a given level of resolution,
	 * it generates a mip-map pyramid hierarchy of coarser levels (until 0).
	 */
	virtual bool mipmap();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Copy constructor forbidden.
	 */
	GvxSceneVoxelizer( const GvxSceneVoxelizer& );

	/**
	 * Copy operator forbidden.
	 */
	GvxSceneVoxelizer& operator=( const GvxSceneVoxelizer& );

};

}

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "GvxSceneVoxelizer.inl"

#endif
