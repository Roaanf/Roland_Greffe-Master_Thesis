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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace Gvx
{

/******************************************************************************
 * Get the data file path
 *
 * @return the data file path
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFilePath() const
{
	return _filePath;
}

/******************************************************************************
 * Set the data file path
 *
 * @param pFilePath the data file path
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFilePath( const std::string& pFilePath )
{
	_filePath = pFilePath;
}

/******************************************************************************
 * Get the data file name
 *
 * @return the data file name
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFileName() const
{
	return _fileName;
}

/******************************************************************************
 * Set the data file name
 *
 * @param pFileName the data file name
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFileName( const std::string& pFileName )
{
	_fileName = pFileName;
}

/******************************************************************************
 * Get the data file extension
 *
 * @return the data file extension
 ******************************************************************************/
inline const std::string& GvxSceneVoxelizer::getFileExtension() const
{
	return _fileExtension;
}

/******************************************************************************
 * Set the data file extension
 *
 * @param pFileExtension the data file extension
 ******************************************************************************/
inline void GvxSceneVoxelizer::setFileExtension( const std::string& pFileExtension )
{
	_fileExtension = pFileExtension;
}

/******************************************************************************
 * Get the max level of resolution
 *
 * @return the max level of resolution
 ******************************************************************************/
inline unsigned int GvxSceneVoxelizer::getMaxResolution() const
{
	return _maxResolution;
}

/******************************************************************************
 * Set the max level of resolution
 *
 * @param pValue the max level of resolution
 ******************************************************************************/
inline void GvxSceneVoxelizer::setMaxResolution( unsigned int pValue )
{
	_maxResolution = pValue;
}

/******************************************************************************
 * Tell wheter or not normals generation is activated
 *
 * @return a flag telling wheter or not normals generation is activated
 ******************************************************************************/
inline bool GvxSceneVoxelizer::isGenerateNormalsOn() const
{
	return _isGenerateNormalsOn;
}

/******************************************************************************
 * Set the flag telling wheter or not normals generation is activated
 *
 * @param pFlag the flag telling wheter or not normals generation is activated
 ******************************************************************************/
inline void GvxSceneVoxelizer::setGenerateNormalsOn( bool pFlag )
{
	_isGenerateNormalsOn = pFlag;
}

/******************************************************************************
 * Get the brick width
 *
 * @return the brick width
 ******************************************************************************/
inline unsigned int GvxSceneVoxelizer::getBrickWidth() const
{
	return _brickWidth;
}

/******************************************************************************
 * Set the brick width
 *
 * @param pValue the brick width
 ******************************************************************************/
inline void GvxSceneVoxelizer::setBrickWidth( unsigned int pValue )
{
	_brickWidth = pValue;
}

/******************************************************************************
 * Get the data type of voxels
 *
 * @return the data type of voxels
 ******************************************************************************/
inline GvxDataTypeHandler::VoxelDataType GvxSceneVoxelizer::getDataType() const
{
	return _dataType;
}

/******************************************************************************
 * Set the data type of voxels
 *
 * @param pType the data type of voxels
 ******************************************************************************/
inline void GvxSceneVoxelizer::setDataType( GvxDataTypeHandler::VoxelDataType pType )
{
	_dataType = pType;
}

}
