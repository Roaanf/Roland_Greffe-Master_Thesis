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

// OpenGL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/vector_types_ext.h>
#include <GvUtils/GvForwardDeclarationHelper.h>

// Loki
#include <loki/Typelist.h>

// GvViewer
#include <GvvPipelineInterface.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Producers
template< typename DataTList, typename NodeRes, typename BrickRes, uint BorderSize >
class ProducerLoad;

// Shaders
class ShaderLoad;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;

// Defines the size of a node tile
typedef GvCore::StaticRes1D< 2 > NodeRes;

// Defines the size of a brick
typedef GvCore::StaticRes1D< 8 > BrickRes;

// Defines the size of the border around a brick
enum { BrickBorderSize = 1 };

// Defines the total size of a brick
typedef GvCore::StaticRes1D< 8 + 2 * BrickBorderSize > RealBrickRes;

// Defines the type of structure we want to use. Array3DGPUTex is the type of array used 
// to store the bricks.
typedef GvStructure::GvVolumeTree
<
	DataType,
	NodeRes, BrickRes
> VolumeTreeType;

// Defines the type of the shader
typedef ShaderLoad ShaderType;

// Defines the type of the producer
typedef ProducerLoad
<
	DataType,
	NodeRes, BrickRes, BrickBorderSize
> ProducerType;

// Defines the type of the cache we want to use.
typedef GvStructure::GvDataProductionManager
<
	VolumeTreeType, ProducerType
> VolumeTreeCacheType;

// Defines the type of the renderer we want to use.
typedef GvRenderer::GvVolumeTreeRendererCUDA
<
	VolumeTreeType, VolumeTreeCacheType,
	ProducerType, ShaderType
> RendererType;

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
class SampleCore : public GvViewerCore::GvvPipelineInterface
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
	SampleCore();

	/**
	 * Destructor
	 */
	virtual ~SampleCore();

	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * Initialize the GigaVoxels pipeline
	 */
	virtual void init();

	/**
	 * Draw function called of frame
	 */
	virtual void draw();

	/**
	 * Resize the frame
	 *
	 * @param width the new width
	 * @param height the new height
	 */
	virtual void resize( int width, int height );

	/**
	 * Clear the GigaVoxels cache
	 */
	virtual void clearCache();

	/**
	 * Toggle the display of the N-tree (octree) of the data structure
	 */
	virtual void toggleDisplayOctree();

	/**
	 * Toggle the GigaVoxels dynamic update mode
	 */
	virtual void toggleDynamicUpdate();

	/**
	 * Toggle the display of the performance monitor utility if
	 * GigaVoxels has been compiled with the Performance Monitor option
	 *
	 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
	 */
	virtual void togglePerfmonDisplay( uint mode );

	/**
	 * Increment the max resolution of the data structure
	 */
	virtual void incMaxVolTreeDepth();

	/**
	 * Decrement the max resolution of the data structure
	 */
	virtual void decMaxVolTreeDepth();

	/**
	 * Get the node tile resolution of the data structure.
	 *
	 * @param pX the X node tile resolution
	 * @param pY the Y node tile resolution
	 * @param pZ the Z node tile resolution
	 */
	virtual void getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const;

		/**
	 * Get the brick resolution of the data structure (voxels).
	 *
	 * @param pX the X brick resolution
	 * @param pY the Y brick resolution
	 * @param pZ the Z brick resolution
	 */
	virtual void getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const;
	
	/**
	 * Get the max depth.
	 *
	 * @return the max depth
	 */
	virtual unsigned int getRendererMaxDepth() const;
	
	/**
	 * Set the max depth.
	 *
	 * @param pValue the max depth
	 */
	virtual void setRendererMaxDepth( unsigned int pValue );

	/**
	 * Get the max number of requests of node subdivisions the cache has to handle.
	 *
	 * @return the max number of requests
	 */
	virtual unsigned int getCacheMaxNbNodeSubdivisions() const;

	/**
	 * Set the max number of requests of node subdivisions.
	 *
	 * @param pValue the max number of requests
	 */
	virtual void setCacheMaxNbNodeSubdivisions( unsigned int pValue );

	/**
	 * Get the max number of requests of brick of voxel loads.
	 *
	 * @return the max number of requests
	 */
	virtual unsigned int getCacheMaxNbBrickLoads() const;
	
	/**
	 * Set the max number of requests of brick of voxel loads.
	 *
	 * @param pValue the max number of requests
	 */
	virtual void setCacheMaxNbBrickLoads( unsigned int pValue );
	
	/**
	 * Set the request strategy indicating if, during data structure traversal,
	 * priority of requests is set on brick loads or on node subdivisions first.
	 *
	 * @param pFlag the flag indicating the request strategy
	 */
	virtual void setRendererPriorityOnBricks( bool pFlag );

	/**
	 * Specify color to clear the color buffer
	 *
	 * @param pRed red component
	 * @param pGreen green component
	 * @param pBlue blue component
	 * @param pAlpha alpha component
	 */
	virtual void setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha );

	/**
	 * Get the aperture
	 *
	 * @return the aperture
	 */
	float getAperture() const;

	/**
	 * Set the aperture
	 *
	 * @param pValue the new value
	 */
	void setAperture( float pValue );

	/**
	 * Get the focal length
	 *
	 * @return the focal length
	 */
	float getFocalLength() const;

	/**
	 * Set the focal length
	 *
	 * @param pValue the new value
	 */
	void setFocalLength( float pValue );

	/**
	 * Get the plane in focus
	 *
	 * @return the plane in focus
	 */
	float getPlaneInFocus() const;

	/**
	 * Set the plane in focus
	 *
	 * @param pValue the new value
	 */
	void setPlaneInFocus( float pValue );

	/**
	 * Tell wheter or not the pipeline has a light.
	 *
	 * @return the flag telling wheter or not the pipeline has a light
	 */
	virtual bool hasLight() const;

	/**
	 * Get the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	virtual void getLightPosition( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the light position
	 *
	 * @param pX the X light position
	 * @param pY the Y light position
	 * @param pZ the Z light position
	 */
	virtual void setLightPosition( float pX, float pY, float pZ );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Depth of field parameter
	 */
	float3 _dofParameters;

	/**
	 * Light position
	 */
	float3 _lightPosition;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/
	
	VolumeTreeType* mVolumeTree;
	VolumeTreeCacheType* mVolumeTreeCache;
	RendererType* mRenderer;
	ProducerType* mProducer;

	GLuint mDepthBuffer;

	GLuint mColorTex;
	GLuint mDepthTex;

	GLuint mFrameBuffer;

	int mWidth;
	int mHeight;

	bool mDisplayOctree;
	uint mDisplayPerfmon;
	uint mMaxVolTreeDepth;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLECORE_H_
