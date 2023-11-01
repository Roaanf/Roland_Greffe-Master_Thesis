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

#ifndef _SAMPLECORE_H_
#define _SAMPLECORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// Loki
#include <loki/Typelist.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvUtils/GsForwardDeclarationHelper.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Forward references
namespace GvCore
{
	//template<
	//	class DataTList, template<typename> class DataArrayType,
	//	class NodeTileRes, class BrickRes, uint BorderSize >
	//struct VolumeTree;

	//template< uint r >
	//struct GsVec1D;

	template< typename T >
	class GsLinearMemory;

	//template< typename T >
	//class GsDeviceTexturingMemory;
}

template< class DataTList >
struct BvhTree;

template < typename BvhTreeType >
class BvhTreeCache;

// Producer
template< typename TDataStructureType, uint DataPageSize, typename TDataProductionManager >
class GPUTriangleProducerBVH;

//template < typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType >
//class SphereProducer;

// Shaders
//class SphereShader;

// Renderers
template< typename BvhTreeType, typename BvhTreeCacheType, typename ProducerType >
class BvhTreeRenderer;

//template<
//	class VolTreeType, class NodeResolution,
//	class BrickResolution, uint BorderSize,
//	class GPUProducer, class SampleShader >
//class RendererCUDA;

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< float4, uchar4 >::Result DataType;

// Defines the size of a node tile
//typedef gigavoxels::GsVec1D< 2 > NodeRes;

// Defines the size of a brick
//typedef gigavoxels::GsVec1D< 8 > BrickRes;

// Defines the size of the border around a brick
//enum { BrickBorderSize = 1 };

// Defines the total size of a brick
//typedef gigavoxels::GsVec1D<8 + 2 * BrickBorderSize> RealBrickRes;

// Defines the type of structure we want to use. GsDeviceTexturingMemory is the type of array used 
// to store the bricks.
typedef BvhTree< DataType > BvhTreeType;

// Defines the type of the producer
//typedef SphereShader ShaderType;

// Defines the type of the cache.
typedef BvhTreeCache< BvhTreeType > BvhTreeCacheType;

// Defines the type of the shader
typedef GPUTriangleProducerBVH<	BvhTreeType, 32, BvhTreeCacheType >	ProducerType;

// Defines the type of the renderer we want to use.
typedef BvhTreeRenderer< BvhTreeType, BvhTreeCacheType, ProducerType > RendererType;

//typedef RendererCUDA< VolumeTreeType,
//	NodeRes, BrickRes, BrickBorderSize,
//	ProducerType, ShaderType >					RendererType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides...
 *
 * ...
 */
class SampleCore
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
	SampleCore();

	/**
	 * Destructor
	 */
	~SampleCore();

	/**
	 * ...
	 */
	void init();
	/**
	 * ...
	 */
	void draw();
	/**
	 * ...
	 *
	 * @param width ...
	 * @param height ...
	 */
	void resize( int width, int height );

	/**
	 * ...
	 */
	void clearCache();

	/**
	 * ...
	 */
	void toggleDisplayOctree();
	/**
	 * ...
	 */
	void toggleDynamicUpdate();
	/**
	 * ...
	 *
	 * @param mode ...
	 */
	void togglePerfmonDisplay( uint mode );

	/**
	 * ...
	 */
	void incMaxVolTreeDepth();
	/**
	 * ...
	 */
	void decMaxVolTreeDepth();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	BvhTreeType			*mBvhTree;
	/**
	 * ...
	 */
	BvhTreeCacheType	*mBvhTreeCache;
	/**
	 * ...
	 */
	RendererType		*mBvhTreeRenderer;
	/**
	 * ...
	 */
	ProducerType		*mProducer;

	/**
	 * ...
	 */
	GLuint				mColorBuffer;
	/**
	 * ...
	 */
	GLuint				mDepthBuffer;

	/**
	 * ...
	 */
	GLuint				mColorTex;
	/**
	 * ...
	 */
	GLuint				mDepthTex;

	/**
	 * ...
	 */
	GLuint				mFrameBuffer;

	/**
	 * ...
	 */
	struct cudaGraphicsResource	*mColorResource;
	/**
	 * ...
	 */
	struct cudaGraphicsResource	*mDepthResource;

	/**
	 * ...
	 */
	int					mWidth;
	/**
	 * ...
	 */
	int					mHeight;

	/**
	 * ...
	 */
	bool				mDisplayOctree;
	/**
	 * ...
	 */
	uint				mDisplayPerfmon;
	/**
	 * ...
	 */
	uint				mMaxVolTreeDepth;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLECORE_H_
