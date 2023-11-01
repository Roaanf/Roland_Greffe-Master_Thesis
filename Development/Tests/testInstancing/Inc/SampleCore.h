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

// Cutil
#include <cutil_math.h>

// GL
#include <GL/glew.h>

// Loki
#include <loki/Typelist.h>

// Gigavoxels
#include <GvCore/vector_types_ext.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvCore
{
	template< uint r >
	struct StaticRes1D;
}
namespace GvStructure
{
	template
	<
		class DataTList, class NodeTileRes,
		class BrickRes, uint BorderSize
	>
	struct GvVolumeTree;

	template
	<
		typename VolumeTreeType, typename ProducerType,
		typename NodeTileRes, typename BrickFullRes
	>
	class GvVolumeTreeCache;
}
namespace GvRenderer
{
	template
	<
		typename VolumeTreeType, typename VolumeTreeCacheType,
		typename ProducerType, typename SampleShader
	>
    class VolumeTreeRendererCUDA_instancing;
}

// Custom Producer
template< typename NodeRes, typename BrickRes, uint BorderSize, typename VolumeTreeType >
class ToreProducer;

// Custom Shader
class ToreShader;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;

// Defines the size of a node tile
typedef GvCore::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::StaticRes1D< 8 > BrickRes;

// Defines the size of the border around a brick
enum { BrickBorderSize = 1 };

// Defines the total size of a brick
typedef GvCore::StaticRes1D<8 + 2 * BrickBorderSize> RealBrickRes;

// Defines the type of the shader
typedef ToreShader ShaderType;

// Defines the type of structure we want to use.
typedef GvStructure::GvVolumeTree
<
	DataType,
	NodeRes, BrickRes, BrickBorderSize
>
VolumeTreeType;

// Defines the type of the producer
typedef ToreProducer
<	NodeRes, BrickRes,
	BrickBorderSize, VolumeTreeType
>
ProducerType;

// Defines the type of the cache we want to use.
typedef GvStructure::GvVolumeTreeCache
<
	VolumeTreeType, ProducerType,
	NodeRes, RealBrickRes
>
VolumeTreeCacheType;

// Defines the type of the renderer we want to use.
typedef GvRenderer::VolumeTreeRendererCUDA_instancing
<
	VolumeTreeType, VolumeTreeCacheType,
	ProducerType, ShaderType
>
VolumeTreeRendererType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides a helper class containing a	GifgaVoxels pipeline.
 *
 * A simple GigaVoxels pipeline consists of :
 * - a data structure
 * - a cache
 * - a custom producer
 * - a renderer
 *
 * The custom shader is pass as a template argument.
 *
 * Besides, this class enables the interoperability with OpenGL graphics library.
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
	 * Initialize the GigaVoxels pipeline
	 */
	void init();

	/**
	 * Draw function called of frame
	 */
	void draw(uchar4 color, float3 pos = make_float3(0.f,0.f,0.f));

	/**
	 * Resize the frame
	 *
	 * @param width the new width
	 * @param height the new height
	 */
	void resize( int width, int height );

	/**
	 * Clear the GigaVoxels cache
	 */
	void clearCache();

	/**
	 * Toggle the display of the N-tree (octree) of the data structure
	 */
	void toggleDisplayOctree();

	/**
	 * Toggle the GigaVoxels dynamic update mode
	 */
	void toggleDynamicUpdate();

	/**
	 * Toggle the display of the performance monitor utility if
	 * GigaVoxels has been compiled with the Performance Monitor option
	 *
	 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
	 */
	void togglePerfmonDisplay( uint mode );

	/**
	 * Increment the max resolution of the data structure
	 */
	void incMaxVolTreeDepth();

	/**
	 * Decrement the max resolution of the data structure
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
	 * Data structure
	 */
	VolumeTreeType* _volumeTree;

	/**
	 * Cache
	 */
	VolumeTreeCacheType* _volumeTreeCache;

	/**
	 * Renderer
	 */
	VolumeTreeRendererType* _volumeTreeRenderer;

	/**
	 * Producer
	 */
	ProducerType* _producer;

	/**
     * Color buffer
	 */
    GLuint _colorBuffer;

	/**
     * Depth buffer
	 */
    GLuint _depthBuffer;

	/**
	 * Color texture
	 */
    GLuint _colorTex;

	/**
	 * Depth texture
	 */
    GLuint _depthTex;

	/**
	 * Frame buffer
	 */
    GLuint _frameBuffer;

	/**
	 * CUDA graphics resource associated with color buffer
	 */
    struct cudaGraphicsResource* _colorResource;

	/**
	 * CUDA graphics resource associated with depth buffer
	 */
    struct cudaGraphicsResource* _depthResource;

	/**
	 * Frame width
	 */
	int _width;

	/**
	 * Frame height
	 */
	int _height;

	/**
	 * Flag to tell wheter or not to display the N-tree (octree) of the data structure
	 */
	bool _displayOctree;

	/**
	 * Flag to tell wheter or not to display the performance monitor utility
	 */
	uint _displayPerfmon;

	/**
	 * Max resolution of the data structure
	 */
	uint _maxVolTreeDepth;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLECORE_H_
