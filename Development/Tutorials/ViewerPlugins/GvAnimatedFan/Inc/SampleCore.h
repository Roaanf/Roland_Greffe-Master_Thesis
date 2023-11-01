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

// Loki
#include <loki/Typelist.h>

// OpenGL
#include <GL/glew.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvUtils/GsForwardDeclarationHelper.h>

// GvViewer
#include <GvvPipelineInterface.h>

// STL
#include <string>

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
typedef Loki::TL::MakeTypelist< uchar4, half4 >::Result DataType;

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
class SampleCore : public GvViewerCore::GvvPipelineInterface
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/******************************* ATTRIBUTES *******************************/

	///**
	// * Type name
	// */
	//static const char* cTypeName;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	SampleCore();

	/**
	 * Destructor
	 */
	virtual ~SampleCore();

	///**
	// * Returns the type of this browsable. The type is used for retrieving
	// * the context menu or when requested or assigning an icon to the
	// * corresponding item
	// *
	// * @return the type name of this browsable
	// */
	//virtual const char* getTypeName() const;
		
	/**
     * Gets the name of this browsable
     *
     * @return the name of this browsable
     */
	virtual const char* getName() const;

	/**
	 * ...
	 */
	virtual void init();
	/**
	 * ...
	 */
	virtual void draw();
	/**
	 * ...
	 *
	 * @param width ...
	 * @param height ...
	 */
	virtual void resize( int width, int height );

	/**
	 * ...
	 */
	virtual void clearCache();

	/**
	 * ...
	 */
	virtual void toggleDisplayOctree();
	/**
	 * ...
	 */
	virtual void toggleDynamicUpdate();
	/**
	 * ...
	 *
	 * @param mode ...
	 */
	virtual void togglePerfmonDisplay( unsigned int mode );

	/**
	 * ...
	 */
	virtual void incMaxVolTreeDepth();
	/**
	 * ...
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
	 * Get the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	virtual void getTranslation( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	virtual void setTranslation( float pX, float pY, float pZ );

	/**
	 * Get the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	virtual void getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	virtual void setRotation( float pAngle, float pX, float pY, float pZ );

	/**
	 * Get the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	virtual void getScale( float& pValue ) const;

	/**
	 * Set the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	virtual void setScale( float pValue );

	/**
	 * Tell wheter or not the pipeline has a transfer function.
	 *
	 * @return the flag telling wheter or not the pipeline has a transfer function
	 */
	virtual bool hasTransferFunction() const;

	/**
	 * Update the associated transfer function
	 *
	 * @param the new transfer function data
	 * @param the size of the transfer function
	 */
	virtual void updateTransferFunction( float* pData, unsigned int pSize );

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

	/**
	 * Tell wheter or not the pipeline has a 3D model to load.
	 *
	 * @return the flag telling wheter or not the pipeline has a 3D model to load
	 */
	virtual bool has3DModel() const;

	/**
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	virtual std::string get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	virtual void set3DModelFilename( const std::string& pFilename );

	/**
	 * Get the 3D model resolution
	 *
	 * @return the 3D model resolution
	 */
	unsigned int get3DModelResolution() const;

	/**
	 * Set the 3D model resolution
	 *
	 * @param pValue the 3D model resolution
	 */
	void set3DModelResolution( unsigned int pValue );

	void setThreshold (double value);

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Translation used to position the GigaVoxels data structure
	 */
	float _translation[ 3 ];

	/**
	 * Rotation used to position the GigaVoxels data structure
	 */
	float _rotation[ 4 ];

	/**
	 * Scale used to transform the GigaVoxels data structure
	 */
	float _scale;

	/**
	 * Light position
	 */
	float3 _lightPosition;

	/**
	 * 3D model filename
	 */
	std::string _filename;

	/**
	 * 3D model resolution
	 */
	unsigned int _resolution;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * GigaSpace producer
	 */
	ProducerType* _producer;

	/**
	 * GigaVoxels pipeline
	 */
	PipelineType* _pipeline;

	/**
	 * GigaSpace renderer
	 */
	RendererType* _renderer;

	/**
	 * ...
	 */
	GLuint _depthBuffer;

	/**
	 * ...
	 */
	GLuint _colorTex;

	/**
	 * ...
	 */
	GLuint _depthTex;

	/**
	 * ...
	 */
	GLuint _frameBuffer;

	/**
	 * ...
	 */
	int _width;

	/**
	 * ...
	 */
	int _height;

	/**
	 * ...
	 */
	bool _displayOctree;

	/**
	 * ...
	 */
	uint _displayPerfmon;

	/**
	 * ...
	 */
	uint _maxVolTreeDepth;

	float threshold;


	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_CORE_H_
