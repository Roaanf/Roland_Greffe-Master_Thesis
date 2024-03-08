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

// GigaVoxels
namespace GvUtils
{
	// Transfer function
	class GsTransferFunction;
}

// Custom Producer
template< typename TDataStructureType, typename TDataProductionManager >
class Producer;

// Custom Shader
class ShaderKernel;

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

// Defines the type list representing the content of one voxel
// - handled types:
typedef Loki::TL::MakeTypelist< ushort >::Result DataType;
//typedef Loki::TL::MakeTypelist< ushort >::Result DataType;
//typedef Loki::TL::MakeTypelist< float >::Result DataType;
// - not yet handled types:
//typedef Loki::TL::MakeTypelist< char >::Result DataType;
//typedef Loki::TL::MakeTypelist< short >::Result DataType;
//typedef Loki::TL::MakeTypelist< int >::Result DataType;
//typedef Loki::TL::MakeTypelist< uint >::Result DataType;

// Defines the size of a node tile
typedef GvCore::GsVec1D< 2 > NodeRes; // The GsVec1D represent a 3D resolution where each dimension has the same value s

// Defines the size of a brick
typedef GvCore::GsVec1D< 64 > BrickRes;

// Defines the type of structure we want to use
//typedef GvStructure::GsVolumeTree< DataType, NodeRes, BrickRes, 0 > DataStructureType;
typedef GvStructure::GsVolumeTree< DataType, NodeRes, BrickRes > DataStructureType;

// Defines the type of the producer
typedef GvStructure::GsDataProductionManager< DataStructureType > DataProductionManagerType;
typedef Producer< DataStructureType, DataProductionManagerType > ProducerType;

// Defines the type of the shader
typedef GvUtils::GsSimpleHostShader< ShaderKernel > ShaderType;

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

	/**
	 * Set the 3D model resolution
	 *
	 * @param pValue the 3D model resolution
	 */
	void setTrueResolution(unsigned int trueX, unsigned int trueY, unsigned int trueZ);

	void setRadius(unsigned int radius);

	/**
	 * Get the producer's threshold
	 *
	 * @return the threshold
	 */
	float getProducerThresholdLow() const;

	/**
	 * Get the producer's threshold
	 *
	 * @return the threshold
	 */
	float getProducerThresholdHigh() const;

	/**
	 * Set the producer's threshold
	 *
	 * @param pValue the threshold
	 */
	void setProducerThresholdLow( float pValue );

	/**
	 * Set the producer's threshold
	 *
	 * @param pValue the threshold
	 */
	void setProducerThresholdHigh( float pValue );

	/**
	 * Get the shader's threshold
	 *
	 * @return the threshold
	 */
	float getShaderThresholdLow() const;

	/**
	 * Get the shader's threshold
	 *
	 * @return the threshold
	 */
	float getShaderThresholdHigh() const;

	/**
	 * Set the shader's threshold
	 *
	 * @param pValue the threshold
	 */
	void setShaderThresholdLow( float pValue );

	/**
	 * Set the shader's threshold
	 *
	 * @param pValue the threshold
	 */
	void setShaderThresholdHigh( float pValue );

	/**
	 * Get the full opacity distance
	 *
	 * @return the full opacity distance
	 */
	float getFullOpacityDistance() const;

	/**
	 * Set the full opacity distance
	 *
	 * @param pValue the full opacity distance
	 */
	void setFullOpacityDistance( float pValue );

	/**
	 * Get the gradient step
	 *
	 * @return the gradient step
	  */
	float getGradientStep() const;

	/**
	 * Set the gradient step
	 *
	 * @param pValue the gradient step
	 */
	void setGradientStep( float pValue );

	float getXRayConst() const;

	void setXRayConst(float pValue);

	/**
	 * Get the min data value
	 *
	 * @return the min data value
	 */
	float getMinDataValue() const;

	/**
	 * Get the max data value
	 *
	 * @return the max data value
	 */
	float getMaxDataValue() const;

	/*
	* Get the value of the boolean gradient rendering
	*/
	bool getGradientRenderingBool() const;

	/*
	* Set the value of the boolean gradient rendering
	*/
	void setGradientRenderingBool(bool pValue);

	void setRenderMode(int index);

	/**
	 * Get the nodes cache usage
	 *
	 * @return the nodes cache usage
	 */
	virtual unsigned int getNodeCacheUsage() const;
	
	/**
	 * Get the bricks cache usage
	 *
	 * @return the bricks cache usage
	 */
	virtual unsigned int getBrickCacheUsage() const;

	/**
	 * Get the number of tree leaf nodes
	 *
	 * @return the number of tree leaf nodes
	 */
	virtual unsigned int getNbTreeLeafNodes() const;
	
	/**
	 * Get the number of tree nodes
	 *
	 * @return the number of tree nodes
	 */
	virtual unsigned int getNbTreeNodes() const;

	/**
	 * Get the brick cache memory
	 *
	 * @return the brick cache memory
	 */
	virtual unsigned int getBrickCacheMemory() const;

	/**
	 * Get the node cache memory
	 *
	 * @return the node cache memory
	 */
	virtual unsigned int getNodeCacheMemory() const;
	
	/**
	 * Get the node cache capacity
	 *
	 * @return the node cache capacity
	 */
	virtual unsigned int getNodeCacheCapacity() const;

	/**
	 * Get the brick cache capacity
	 *
	 * @return the brick cache capacity
	 */
	virtual unsigned int getBrickCacheCapacity() const;

	/**
	 * Get the flag indicating wheter or not cache monitoring is activated
	 *
	 * @return the flag indicating wheter or not cache monitoring is activated
	 */
	virtual bool hasCacheMonitoring() const;

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

	/**
	 * Producer's threshold
	 */
	float _producerThresholdLow;

	/**
	 * Producer's threshold
	 */
	float _producerThresholdHigh;

	/**
	 * Shader's threshold
	 */
	float _shaderThresholdLow;

	/**
	 * Shader's threshold
	 */
	float _shaderThresholdHigh;

	/**
	 * Full opacity distance
	 */
	float _fullOpacityDistance;

	/**
	 * Gradient step used to compute local normal with symetric gradient from data
	 */
	float _gradientStep;

	/**
	 * Minimum data value
	 */
	float _minDataValue;

	/**
	 * Maximum data value
	 */
	float _maxDataValue;

	/**
	 * Gradient Rendering boolean
	 */
	bool _gradientRendering;

	int _renderMode;

	uint _radius;

	float _xRayConst;

	/******************************** METHODS *********************************/

	/**
	 * Initialize the transfer function
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool initializeTransferFunction();

	/**
	 * Finalize the transfer function
	 *
	 * @return flag to tell wheter or not it succeeded
	 */
	bool finalizeTransferFunction();

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * GigaVoxels pipeline
	 */
	PipelineType* _pipeline;

	size_t _nodeMemoryPool;

	size_t _brickMemoryPool;

	unsigned int _trueX;
	unsigned int _trueY;
	unsigned int _trueZ;

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

	/**
	 * Transfer function
	 */
	GvUtils::GsTransferFunction* _transferFunction;

	/******************************** METHODS *********************************/

};

#endif // !_SAMPLE_CORE_H_
