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

#ifndef _RLE_COMPRESSOR_H_
#define _RLE_COMPRESSOR_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

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
 * @class GvCommonGraphicsPass
 *
 * @brief The GvCommonGraphicsPass class provides interface to
 *
 * Some resources from OpenGL may be mapped into the address space of CUDA,
 * either to enable CUDA to read data written by OpenGL, or to enable CUDA
 * to write data for consumption by OpenGL.
 */
class RLECompressor
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	RLECompressor();

	/**
	 * Destructor
	 */
	virtual ~RLECompressor();

	/**
	 * Initiliaze
	 */
	void initialize();

	/**
	 * Finalize
	 */
	void finalize();

	/**
	 * ...
	 *
	 * @param pInput ...
	 * @param pOutput ...
	 */
	static void sameArray( const unsigned char* pInput, const unsigned char* pOutput );

	/**
	 * Given an input data array, write output array with RLE encoding (compression)
	 *
	 * @param pInput input data
	 * @param pOutput output data (RLE encoding - compression)
	 */
	//static void RLEcomp( const unsigned int* pInput, unsigned int* pOutput );
	//static void RLEcompBis( const unsigned char* pInput, unsigned char* pOutput );

	/**
	 * Given an input data array, write output arrays with RLE encoding (compression)
	 *
	 * @param input initial array
	 * @param nBricks the number of bricks in the input array
	 * @param bricksEnds array containing the ends of each bricks in the compressed arrays 
	 * @param plateausValues values of the plateaus in the bricks
	 * @param plateausStarts beginnings of each plateaus in the bricks
	 */
	void compressionPrefixSum( const unsigned int* input, 
			const unsigned int nBricks,
			unsigned int* bricksEnds, 
			unsigned int* plateausValues,
			unsigned char* plateausStarts );

	/**
	 * Given an input data array, write output array with RLE encoding (compression)
	 * - data
	 * - and offsets
	 *
	 * @param pInput input data
	 * @param pOutputData output data (RLE encoding - compression)
	 * @param pOutputOffset output offsets (RLE encoding - compression)
	 */
	static bool RLEcompOffset( const unsigned char* pInput, unsigned char* pOutputData, unsigned int* pOutputOffset );

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

	/**
	 * Copy constructor forbidden.
	 */
	RLECompressor( const RLECompressor& );

	/**
	 * Copy operator forbidden.
	 */
	RLECompressor& operator=( const RLECompressor& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "RLECompressor.inl"

#endif
