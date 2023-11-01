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

#ifndef _HELPER_MATRIX_H_
#define _HELPER_MATRIX_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

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

/**
 * @class MatrixHelper
 *
 * @brief The MatrixHelper class provides helper functions to manipulate matrices.
 *
 * ...
 *
 */
class MatrixHelper
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	/******************************** METHODS *********************************/

	/**
	 * Define a viewing transformation
	 *
	 * @param eyeX specifies the position of the eye point
	 * @param eyeY specifies the position of the eye point
	 * @param eyeZ specifies the position of the eye point
	 * @param centerX specifies the position of the reference point
	 * @param centerY specifies the position of the reference point
	 * @param centerZ specifies the position of the reference point
	 * @param upX specifies the direction of the up vector
	 * @param upY specifies the direction of the up vector
	 * @param upZ specifies the direction of the up vector
	 *
	 * @return the resulting viewing transformation
	 */
	static inline float4x4 lookAt( float eyeX, float eyeY, float eyeZ,
								float centerX, float centerY, float centerZ,
								float upX, float upY, float upZ );

	/**
	 * Define a translation matrix
	 *
	 * @param x specify the x, y, and z coordinates of a translation vector
	 * @param y specify the x, y, and z coordinates of a translation vector
	 * @param z specify the x, y, and z coordinates of a translation vector
	 *
	 * @return the resulting translation matrix
	 */
	static inline float4x4 translate( float x, float y, float z);

	/**
	 * Multiply two matrices
	 *
	 * @param a first matrix
	 * @param b second matrix
	 *
	 * @return the resulting matrix
	 */
	static inline float4x4 mul( const float4x4& a, const float4x4& b );

	/**
	 * Copy data from input matrix to output matrix
	 *
	 * @param pInputMatrix input matrix
	 * @param pOutpuMatrix outpu matrix
	 */
	static inline void copy( const float4x4& pInputMatrix, float4x4& pOutpuMatrix );

	/**
	 * Define a transformation that produces a parallel projection (i.e. orthographic)
	 *
	 * @param left specify the coordinates for the left and right vertical clipping planes
	 * @param right specify the coordinates for the left and right vertical clipping planes
	 * @param bottom specify the coordinates for the bottom and top horizontal clipping planes
	 * @param top specify the coordinates for the bottom and top horizontal clipping planes
	 * @param nearVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
	 * @param farVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
	 *
	 * @return the resulting orthographic matrix
	 */
	static inline float4x4 ortho( float left, float right, float bottom, float top,	float nearVal, float farVal );

	/**
	 * Create matrix used to change of reference frame matrix associated to a brick
	 *
	 * ...
	 *
	 * @param pBrickPos position of the brick (same as the position of the node minus the border)
	 * @param pXSize x size of the brick ( same as the size of the node plus the border )
	 * @param pYSize y size of the brick ( same as the size of the node plus the border )
	 * @param pZSize z size of the brick ( same as the size of the node plus the border )
	 *
	 * @return ...
	 */
	static inline float4x4 brickBaseMatrix( const float3& brickPos, float xSize, float ySize, float zSize );

	/**
	 * Methods that compute Change-of-basis matrices to project along the 3 axis.
	 * Those matrices are the multiplication of the openGL modelViewMatrix with projectionMatrix
	 * after a call to gluLookAt(...) and glortho(...).
	 *
	 * @param brickPos Origin of the brick's base
	 * @param xSize Size of the brick along x axe
	 * @param ySize Size of the brick along y axe
	 * @param zSize Size of the brick along z axe
	 * @param projectionMatX Change-of-basis matrix to project along X
	 * @param projectionMatY Change-of-basis matrix to project along y
	 * @param projectionMatZ Change-of-basis matrix to project along Z
	 */
	static inline void projectionMatrix( const float3& brickPos, float xSize, float ySize, float zSize,
										float4x4& projectionMatX, float4x4& projectionMatY, float4x4& projectionMatZ );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "helper_matrix.inl"

#endif // !_HELPER_MATRIX_H_
