/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

/** 
 * @version 1.0
 */

#ifndef _SAMPLE_CORE_H_
#define _SAMPLE_CORE_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Cuda
#include <cuda_runtime.h>
#include <vector_types.h>

// GL
#include <GL/glew.h>

// Loki
#include <loki/Typelist.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvUtils/GsForwardDeclarationHelper.h>

// GvViewer
#include <GvvPipelineInterface.h>

// Project
#include "CustomPriorityPolicies.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvUtils
{
	// Transfer function
	class GsTransferFunction;
}
namespace GsGraphics
{
	class GsShaderProgram;
}

// Custom Producer
template< typename TDataStructureType >
class ProducerKernel;

// Custom Shader
class ShaderFractal;

// Custom Priority Policy Manager
class CustomPriorityPoliciesManager;

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

// Defines the type of the data manager
typedef GvStructure::GsDataProductionManager< DataStructureType, CustomPriorityPoliciesManager > DataProductionManagerType;

// Defines the type of the producer
typedef GvUtils::GsSimpleHostProducer
<
	ProducerKernel< DataStructureType >,
	DataStructureType,
	DataProductionManagerType
>
ProducerType;

// Defines the type of the shader
typedef ShaderFractal ShaderType;

// Define the type of renderer
typedef GvRendering::GsRendererCUDA< DataStructureType, DataProductionManagerType, ShaderType > RendererType;

// Simple Pipeline
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
class SampleCore : public GvViewerCore::GvvPipelineInterface
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
	 * Draw function called every frame
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
	 * Tell whether or not the pipeline has a transfer function.
	 *
	 * @return the flag telling whether or not the pipeline has a transfer function
	 */
	virtual bool hasTransferFunction() const;

	/**
	 * Get the transfer function filename if it has one.
	 *
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual const char* getTransferFunctionFilename( unsigned int pIndex = 0 ) const;

	/**
	 * Set the transfer function filename if it has one.
	 *
	 * @param pFilename the transfer function's filename
	 * @param pIndex the index of the transfer function
	 *
	 * @return the transfer function
	 */
	virtual void setTransferFunctionFilename( const char* pFilename, unsigned int pIndex = 0 );

	/**
	 * Update the associated transfer function
	 *
	 * @param pData the new transfer function data
	 * @param pSize the size of the transfer function
	 */
	virtual void updateTransferFunction( float* pData, unsigned int pSize );

	/**
	 * Get the translation used to position the GigaVoxels data structure
	 *
	 * @param pX the x componenet of the translation
	 * @param pX the y componenet of the translation
	 * @param pX the z componenet of the translation
	 */
	void getTranslation( float& pX, float& pY, float& pZ ) const;

	/**
	 * Get the number of requests of node subdivisions the cache has handled.
	 *
	 * @return the number of requests
	 */
	virtual unsigned int getCacheNbNodeSubdivisionRequests() const;

	/**
	 * Get the number of requests of brick of voxel loads the cache has handled.
	 *
	 * @return the number of requests
	 */
	virtual unsigned int getCacheNbBrickLoadRequests() const;

	/**
	 * Get the cache policy
	 *
	 * @return the cache policy
	 */
	virtual unsigned int getCachePolicy() const;

	/**
	 * Set the cache policy
	 *
	 * @param pValue the cache policy
	 */
	virtual void setCachePolicy( unsigned int pValue );

	/**
	 * Get the node cache memory
	 *
	 * @return the node cache memory
	 */
	virtual unsigned int getNodeCacheMemory() const;

	/**
	 * Set the node cache memory
	 *
	 * @param pValue the node cache memory
	 */
	virtual void setNodeCacheMemory( unsigned int pValue );

	/**
	 * Get the brick cache memory
	 *
	 * @return the brick cache memory
	 */
	virtual unsigned int getBrickCacheMemory() const;

	/**
	 * Set the brick cache memory
	 *
	 * @param pValue the brick cache memory
	 */
	virtual void setBrickCacheMemory( unsigned int pValue );

	/**
	 * Get the node cache capacity
	 *
	 * @return the node cache capacity
	 */
	virtual unsigned int getNodeCacheCapacity() const;

	/**
	 * Set the node cache capacity
	 *
	 * @param pValue the node cache capacity
	 */
	virtual void setNodeCacheCapacity( unsigned int pValue );

	/**
	 * Get the brick cache capacity
	 *
	 * @return the brick cache capacity
	 */
	virtual unsigned int getBrickCacheCapacity() const;

	/**
	 * Set the brick cache capacity
	 *
	 * @param pValue the brick cache capacity
	 */
	virtual void setBrickCacheCapacity( unsigned int pValue );

	/**
	 * Get the number of unused nodes in cache
	 *
	 * @return the number of unused nodes in cache
	 */
	virtual unsigned int getCacheNbUnusedNodes() const;

	/**
	 * Get the number of unused bricks in cache
	 *
	 * @return the number of unused bricks in cache
	 */
	virtual unsigned int getCacheNbUnusedBricks() const;

	/**
	 * Get the fractal's power
	 *
	 * @return the fractal's power
	 */
	int getFractalPower() const;

	/**
	 * Set the fractal's power
	 *
	 * @param pValue the fractal's power
	 */
	void setFractalPower( int pValue );

	/**
	 * Get the fractal's nb iterations
	 *
	 * @return the fractal's nb iterations
	 */
	unsigned int getFractalNbIterations() const;

	/**
	 * Set the fractal's nb iterations
	 *
	 * @param pValue the fractal's nb iterations
	 */
	void setFractalNbIterations( unsigned int pValue );

	/**
	 * Tell whether or not the fractal's adaptative iteration mode is activated
	 *
	 * @return a flag telling whether or not the fractal's adaptative iteration mode is activated
	 */
	bool hasFractalAdaptativeIterations() const;

	/**
	 * Set the flag telling whether or not the fractal's adaptative iteration mode is activated
	 *
	 * @param pFlags the flag telling whether or not the fractal's adaptative iteration mode is activated
	 */
	void setFractalAdaptativeIterations( bool pFlag );

	/**
	 * Set the priority policy to use.
	 */
	void setProductionPriorityPolicy( PriorityPolicies policy );

	/**
	 * Get the priority policy in use.
	 */
	PriorityPolicies getProductionPriorityPolicy();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Translation used to position the GigaVoxels data structure
	 */
	float _translation[ 3 ];

	/**
	 * Get the fractal's power
	 *
	 * @return the fractal's power
	 */
	int _fractalPower;

	/**
	 * Get the fractal's nb iterations
	 *
	 * @return the fractal's nb iterations
	 */
	unsigned int _fractalNbIterations;

	/**
	 * Tell whether or not the fractal's adaptative iteration mode is activated
	 *
	 * @return a flag telling whether or not the fractal's adaptative iteration mode is activated
	 */
	bool _hasFactalAdaptativeIterations;

	/******************************** METHODS *********************************/

	/**
	 * Initialize the GigaVoxels pipeline
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool initializePipeline();

	/**
	 * Finalize the GigaVoxels pipeline
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool finalizePipeline();

	/**
	 * Initialize the transfer function
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool initializeTransferFunction();

	/**
	 * Finalize the transfer function
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool finalizeTransferFunction();

	/**
	 * Finalize graphics resources
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool finalizeGraphicsResources();

	/**
	 * Initialize elements for post-processing HUD display
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool initializeHUDDisplay();

	/**
	 * Finalize elements for post-processing HUD display
	 *
	 * @return flag to tell whether or not it succeeded
	 */
	bool finalizeHUDDisplay();

	/**
	 * Render the HUD display
	 */
	void renderHUDDisplay();

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
	 * Frame width
	 */
	int _width;

	/**
	 * Frame height
	 */
	int _height;

	/**
	 * Flag to tell whether or not to display the N-tree (octree) of the data structure
	 */
	bool _displayOctree;

	/**
	 * Flag to tell whether or not to display the performance monitor utility
	 */
	uint _displayPerfmon;

	/**
	 * Max resolution of the data structure
	 */
	uint _maxVolTreeDepth;

	/**
	 * Transfer function
	 */
	GvUtils::GsTransferFunction* _transferFunction;

	/**
	 * Shader program for post-processing
	 */
	GsGraphics::GsShaderProgram* _postProcessShaderProgram;

};

#endif // !_SAMPLE_CORE_H_
