/*
 * GigaVoxels is a ray-guided streaming library used for efficient
 * 3D real-time rendering of highly detailed volumetric scenes.
 *
 * Copyright (C) 2011-2012 INRIA <http://www.inria.fr/>
 *
 * Authors : GigaVoxels Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/** 
 * @version 1.0
 */

#ifndef _SAMPLECORE_H_
#define _SAMPLECORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// Cuda GPU Computing SDK
#include <helper_math.h>

// Loki
#include <loki/Typelist.h>

// OpenGL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvUtils/GsForwardDeclarationHelper.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Custom Producer
template< typename TDataStructureType, typename TDataProductionManager >
class Producer;

// Custom Shader
class ShaderKernel;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;

// Defines the size of a node tile
typedef GvCore::GsVec1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::GsVec1D< 8 > BrickRes;

// Defines the type of structure we want to use
typedef GvStructure::GsVolumeTree
<
	DataType,
	NodeRes, BrickRes
>
DataStructureType;

// Defines the type of the producer
typedef GvStructure::GsDataProductionManager< DataStructureType > DataProductionManagerType;
typedef Producer< DataStructureType, DataProductionManagerType > ProducerType;

// Defines the type of the shader
typedef GvUtils::GsSimpleHostShader
<
	ShaderKernel
>
ShaderType;

// Define the type of renderer
typedef GvRendering::GsRendererCUDA< DataStructureType, DataProductionManagerType, ShaderType > RendererType;

// Defines the GigaVoxels Pipeline
typedef GvUtils::GsSimplePipeline
<
	ShaderType,
	DataStructureType,
	DataProductionManagerType
>
PipelineType;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class SampleCore
 *
 * @brief The SampleCore class provides a helper class containing a	GigaVoxels pipeline.
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
	 * Draw function called every frame
	 */
	void draw();

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

	/**
	 * Specify color to clear the color buffer
	 *
	 * @param pRed red component
	 * @param pGreen green component
	 * @param pBlue blue component
	 * @param pAlpha alpha component
	 */
	void setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha );

	/**
	 * Set the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	void setLightPosition( float pX, float pY, float pZ );

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
	 * GigaVoxels pipeline
	 */
	PipelineType* _pipeline;

	/**
	 * GigaSpace producer
	 */
	ProducerType* _producer;

	/**
	 * GigaSpace renderer
	 */
	RendererType* _renderer;
	
	/**
	 * ...
	 */
	GLuint _inputDepthBuffer;

	// ...
	// UTILISER 2 FBO et deux textures pour ne pas faire un read/write de color �mais s�per en 2 �l�ments !!!!!!!!!!!!!
	// ...

	/**
	 * ...
	 */
	GLuint mColorTex;

	/**
	 * ...
	 */
	GLuint mDepthTex;

	/**
	 * ...
	 */
	GLuint _frameBuffer;

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
