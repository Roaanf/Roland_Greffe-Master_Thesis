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

#include "SampleCore.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsVector.h>
#include <GvStructure/GsVolumeTree.h>
#include <GvStructure/GsDataProductionManager.h>
#include <GvRendering/GsRendererCUDA.h>
#include <GvUtils/GsSimplePipeline.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsCommonGraphicsPass.h>
#include <GvUtils/GsTransferFunction.h>
#include <GvCore/GsError.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GsGraphics/GsShaderProgram.h>
#include <GvUtils/GsDataLoader.h>
#include <GvUtils/GsEnvironment.h>

// Project
#include "Producer.h"
#include "ShaderKernel.h"

// GvViewer
#include <GvvApplication.h>

// Cuda SDK
#include <helper_math.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GvRendering;
using namespace GvUtils;
using namespace GsGraphics;

// GigaVoxels viewer
using namespace GvViewerCore;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )	// 256 Mo

///**
// * Tag name identifying a space profile element
// */
//const char* SampleCore::cTypeName = "GigaVoxelsPipeline";

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

//template< typename TDataTypeList, int TChannel >
//void listDataTypes()
//{
//	// Typedef to access the channel in the data type list
//	typedef typename Loki::TL::TypeAt< TDataTypeList, TChannel >::Result ChannelType;
//
//	std::cout << GvCore::typeToString< ChannelType >() << std::endl;
////	// Build filename according to GigaVoxels internal syntax
////	std::stringstream filename;
////	filename << mFileName << "_BR" << mBrickSize << "_B" << mBorderSize << "_L" << mLevel
////		<< "_C" << TChannel << "_" << GvCore::typeToString< ChannelType >() << mFileExt;
////
////	// Store generated filename
////	mResult->push_back( filename.str() );
//}


/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	GvViewerScene::GvvPipeline()
,	_pipeline( NULL )
,	_renderer( NULL )
,	_graphicsEnvironment( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
,	_width( 0 )
,	_height( 0 )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_textureSamplerUniformLocation( 0 )
,	_filename()
,	_resolution( 0 )
,	_transferFunction( NULL )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Rotation used to position the GigaVoxels data structure
	_rotation[ 0 ] = 0.0f;
	_rotation[ 1 ] = 0.0f;
	_rotation[ 2 ] = 0.0f;
	_rotation[ 3 ] = 0.0f;

	// Scale used to transform the GigaVoxels data structure
	_scale = 1.0f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	// Delete the GigaVoxels pipeline
	delete _pipeline;
	_pipeline = NULL;

	// Finalize the transfer function (free memory)
	finalizeTransferFunction();
}

///******************************************************************************
// * Returns the type of this browsable. The type is used for retrieving
// * the context menu or when requested or assigning an icon to the
// * corresponding item
// *
// * @return the type name of this browsable
// ******************************************************************************/
//const char* SampleCore::getTypeName() const
//{
//	return cTypeName;
//}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "DicomViewer";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = PipelineType::NodeTileResolution::numElements * sizeof( GvStructure::GsNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< PipelineType::DataTypeList >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3((uint)floorf(powf((float)nodePoolNumElems, 1.0f / 3.0f))) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3((uint)floorf(powf((float)brickPoolNumElems, 1.0f / 3.0f))) * RealBrickRes::get();

	//std::cout << "" << std::endl;
	//std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	//std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Instanciate our objects
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon512" );
	//set3DModelFilename( filename.toLatin1().constData() );
	// TO DO :
	// Test empty and existence of filename
	QString filename( get3DModelFilename().c_str() );
	GvUtils::GsDataLoader< DataType >* dataLoader = new GvUtils::GsDataLoader< DataType >(
														filename.toStdString(), PipelineType::BrickTileResolution::get(), PipelineType::BrickTileBorderSize, true );

	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline = new PipelineType();
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, shader );

	// Producer initialization
	_producer = new ProducerType( 64 * 1024 * 1024, nodePoolRes.x * nodePoolRes.y * nodePoolRes.z );
	assert( _producer != NULL );
	_producer->attachProducer( dataLoader );
	_pipeline->addProducer( _producer );

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	unsigned int dataResolution = dataLoader->getDataResolution().x;	// TODO : change that (uniform data)
	const unsigned int nbLevelOfResolution = static_cast< unsigned int >( log( static_cast< float >( dataResolution / 8/*<== if 8 voxels by bricks*/ ) ) / log( static_cast< float >( 2 ) ) );
	//_maxVolTreeDepth = nbLevelOfResolution;
	_maxVolTreeDepth = 6;
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 100 );
	_pipeline->editCache()->setMaxNbBrickLoads( 100 );

	// Graphics environment creation
	_graphicsEnvironment = new GsCommonGraphicsPass();
	
	// Typedef to access the channel in the data type list
	//for ( int i = 0; i < GvCore::DataNumChannels< DataType >::value; i++ )
	//{
	//	listDataTypes< DataType, Loki::Int2Type< i > ) >();
	//}
		
	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Need to initialize CUDA memory with light position
	float x,y,z;
	getLightPosition( x,y,z );
	setLightPosition( x,y,z );

	// Create and link a GLSL shader program
	//QString dataRepository = QCoreApplication::applicationDirPath();
	//dataRepository += QDir::separator();
	//dataRepository += QString( "Data" );
	//dataRepository += QDir::separator();
	//dataRepository += QString( "Shaders" );
	QString dataRepository = GvUtils::GsEnvironment::getDataDir( GvUtils::GsEnvironment::eShadersDir ).c_str();
	dataRepository += QDir::separator();
	dataRepository += QString( "GvDynamicLoad" );
	dataRepository += QDir::separator();
	// Initialize points shader program
	QString vertexShaderFilename = dataRepository + QString( "fullscreenTriangle_vert.glsl" );
	QString fragmentShaderFilename = dataRepository + QString( "fullscreenTriangle_frag.glsl" );

	// Initialize the transfer function
	initializeTransferFunction();

	// Initialize shader program
	_shaderProgram = new GsShaderProgram();
	bool statusOK = _shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}
	statusOK = _shaderProgram->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}
	statusOK = _shaderProgram->link();
	assert( statusOK );
	if ( ! statusOK )
	{
		// TO DO
		// - handle error
	}

	// After linking has occurred, the command glGetUniformLocation can be used to obtain the location of a uniform variable
	// - beware, after "shader link", this location my changed
	_textureSamplerUniformLocation = glGetUniformLocation( _shaderProgram->_program, "uTextureSampler" );
	assert( _textureSamplerUniformLocation != -1 );
	if ( _textureSamplerUniformLocation == -1 )
	{
		// TO DO
		// - handle error
	}
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glMatrixMode( GL_MODELVIEW );

	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		// Translation used to position the GigaVoxels data structure
		glScalef( _scale, _scale, _scale );
		glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
		GV_CHECK_GL_ERROR();
	}
	else
	{
		glClear( GL_COLOR_BUFFER_BIT );
	}

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Translation used to position the GigaVoxels data structure
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	//  Render the result to the screen
	// - see if it can be cached
	glDisable( GL_DEPTH_TEST );
	// - Draw fullscreen textured quad or triangle
	_shaderProgram->use();
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	glUniform1i( _textureSamplerUniformLocation, 0 );
	/*glBindVertexArray( _fullscreenQuadVAO );
	glDrawArrays( GL_QUADS, 0, 4 );
	glBindVertexArray( 0 );*/
	glDrawArrays( GL_TRIANGLES, 0, 3 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glUseProgram( 0 );

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	//_volumeTreeRenderer->doPostRender();
	
	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
		
		// SORTIE CONSOLE
		// GvPerfMon::CUDAPerfMon::get().displayFrame();

		// HOST
		//GvPerfMon::CUDAPerfMon::get().displayFrameGL( 1 );
		
		// DEVICE
		//GvPerfMon::CUDAPerfMon::get().displayFrameGL( 0 );
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param pWidth the new width
 * @param pHeight the new height
 ******************************************************************************/
void SampleCore::resize( int pWidth, int pHeight )
{
	// LOG
	//
	// @todo : check and avoid 0 values, replace by 1 and warn user
	if ( pWidth == 0 )
	{
		// TO DO
		// ...
	}
	if ( pHeight == 0 )
	{
		// TO DO
		// ...
	}

	_width = pWidth;
	_height = pHeight;

	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setBufferSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
	}
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	setDynamicUpdate( ! hasDynamicUpdate() );
}

/******************************************************************************
 * Get the dynamic update state
 *
 * @return the dynamic update state
 ******************************************************************************/
bool SampleCore::hasDynamicUpdate() const
{
	return _pipeline->hasDynamicUpdate();
}

/******************************************************************************
 * Set the dynamic update state
 *
 * @param pFlag the dynamic update state
 ******************************************************************************/
void SampleCore::setDynamicUpdate( bool pFlag )
{
	_pipeline->setDynamicUpdate( pFlag );
}

/******************************************************************************
 * Toggle the display of the performance monitor utility if
 * GigaVoxels has been compiled with the Performance Monitor option
 *
 * @param mode The performance monitor mode (1 for CPU, 2 for DEVICE)
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if ( _displayPerfmon )
	{
		_displayPerfmon = 0;

		GvPerfMon::CUDAPerfMon::_isActivated = false;
	}
	else
	{
		_displayPerfmon = mode;

		GvPerfMon::CUDAPerfMon::_isActivated = true;
	}
}

/******************************************************************************
 * Increment the max resolution of the data structure
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth < 32 )
	{
		_maxVolTreeDepth++;
	}

	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Decrement the max resolution of the data structure
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( _maxVolTreeDepth > 0 )
	{
		_maxVolTreeDepth--;
	}

	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * Get the max depth.
 *
 * @return the max depth
 ******************************************************************************/
unsigned int SampleCore::getRendererMaxDepth() const
{
	return _pipeline->editDataStructure()->getMaxDepth();
}

/******************************************************************************
 * Set the max depth.
 *
 * @param pValue the max depth
 ******************************************************************************/
void SampleCore::setRendererMaxDepth( unsigned int pValue )
{
	_pipeline->editDataStructure()->setMaxDepth( pValue );
}

/******************************************************************************
 * Set the request strategy indicating if, during data structure traversal,
 * priority of requests is set on brick loads or on node subdivisions first.
 *
 * @param pFlag the flag indicating the request strategy
 ******************************************************************************/
void SampleCore::setRendererPriorityOnBricks( bool pFlag )
{
	/*_pipeline->editRenderer()*/_renderer->setPriorityOnBricks( pFlag );
}

/******************************************************************************
 * Specify color to clear the color buffer
 *
 * @param pRed red component
 * @param pGreen green component
 * @param pBlue blue component
 * @param pAlpha alpha component
 ******************************************************************************/
void SampleCore::setClearColor( unsigned char pRed, unsigned char pGreen, unsigned char pBlue, unsigned char pAlpha )
{
	/*_pipeline->editRenderer()*/_renderer->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
}

/******************************************************************************
 * Tell whether or not the pipeline has a light.
 *
 * @return the flag telling whether or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
	//return true;
	return false;
}

/******************************************************************************
 * Get the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::getLightPosition( float& pX, float& pY, float& pZ ) const
{
	pX = _lightPosition.x;
	pY = _lightPosition.y;
	pZ = _lightPosition.z;
}

/******************************************************************************
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{
	// Update DEVICE memory with "light position"
	//
	// WARNING
	// Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
	// Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
	_lightPosition.x = pX/* - _translation[ 0 ]*/;
	_lightPosition.y = pY/* - _translation[ 1 ]*/;
	_lightPosition.z = pZ/* - _translation[ 2 ]*/;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
}

/******************************************************************************
 * Set the translation
 *
 * @param pX the translation on x axis
 * @param pY the translation on y axis
 * @param pZ the translation on z axis
 ******************************************************************************/
void SampleCore::setTranslation( float pX, float pY, float pZ )
{
	_translation[ 0 ] = pX;
	_translation[ 1 ] = pY;
	_translation[ 2 ] = pZ;
}

/******************************************************************************
 * Get the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const
{
	pAngle = _rotation[ 0 ];
	pX = _rotation[ 1 ];
	pY = _rotation[ 2 ];
	pZ = _rotation[ 3 ];
}

/******************************************************************************
 * Set the rotation
 *
 * @param pAngle the rotation angle (in degree)
 * @param pX the x component of the rotation vector
 * @param pY the y component of the rotation vector
 * @param pZ the z component of the rotation vector
 ******************************************************************************/
void SampleCore::setRotation( float pAngle, float pX, float pY, float pZ )
{
	_rotation[ 0 ] = pAngle;
	_rotation[ 1 ] = pX;;
	_rotation[ 2 ] = pY;;
	_rotation[ 3 ] = pZ;;
}

/******************************************************************************
 * Get the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::getScale( float& pValue ) const
{
	pValue = _scale;
}

/******************************************************************************
 * Set the uniform scale
 *
 * @param pValue the uniform scale
 ******************************************************************************/
void SampleCore::setScale( float pValue )
{
	_scale = pValue;
}

/******************************************************************************
 * Tell whether or not the pipeline has a transfer function.
 *
 * @return the flag telling whether or not the pipeline has a transfer function
 ******************************************************************************/
bool SampleCore::hasTransferFunction() const
{
	return true;
}

/******************************************************************************
 * Get the transfer function filename if it has one.
 *
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
const char* SampleCore::getTransferFunctionFilename( unsigned int pIndex ) const
{
	assert( _transferFunction != NULL );
	if ( _transferFunction != NULL )
	{
		return _transferFunction->getFilename().c_str();
	}

	return NULL;
}

/******************************************************************************
 * Set the transfer function filename  if it has one.
 *
 * @param pFilename the transfer function's filename
 * @param pIndex the index of the transfer function
 *
 * @return the transfer function
 ******************************************************************************/
void SampleCore::setTransferFunctionFilename( const char* pFilename, unsigned int pIndex )
{
	assert( _transferFunction != NULL );
	if ( _transferFunction != NULL )
	{
		_transferFunction->setFilename( pFilename );
	}
}

/******************************************************************************
 * Update the associated transfer function
 *
 * @param pData the new transfer function data
 * @param pSize the size of the transfer function
 ******************************************************************************/
void SampleCore::updateTransferFunction( float* pData, unsigned int pSize )
{
	assert( _transferFunction != NULL );
	if ( _transferFunction != NULL )
	{
		// Apply modifications on transfer function's internal data
		float4* tf = _transferFunction->editData();
		unsigned int size = _transferFunction->getResolution();
		assert( size == pSize );
		for ( unsigned int i = 0; i < size; ++i )
		{
			tf[ i ] = make_float4( pData[ 4 * i ], pData[ 4 * i + 1 ], pData[ 4 * i + 2 ], pData[ 4 * i + 3 ] );
		}

		// Apply modifications on device memory
		_transferFunction->updateDeviceMemory();

		// Update cache because transfer function is applied during Producer stage
		// and not in real-time in during Sheder stage.
		//_pipeline->clear();
	}
}

/******************************************************************************
 * Initialize the transfer function
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::initializeTransferFunction()
{
	// Create the transfer function
	_transferFunction = new GvUtils::GsTransferFunction();
	assert( _transferFunction != NULL );
	if ( _transferFunction == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}

	// Initialize transfer fcuntion with a resolution of 256 elements
	_transferFunction->create( 256 );

	// Bind the transfer function's internal data to the texture reference that will be used on device code
	_transferFunction->bindToTextureReference( &transferFunctionTexture, "transferFunctionTexture", true, cudaFilterModeLinear, cudaAddressModeClamp );
	
	// Default filename
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString filename = dataRepository + QDir::separator() + QString( "TransferFunctions" ) + QDir::separator() + QString( "TF_AmplifiedVolume_01.xml" );
	QString filename = GsEnvironment::getDataDir( GsEnvironment::eTransferFunctionsDir ).c_str();
	filename += QDir::separator();
	filename += QString( "TransferFunction_Qtfe_01.xml" );
	QFileInfo fileInfo( filename );
	if ( ( ! fileInfo.isFile() ) || ( ! fileInfo.isReadable() ) )
	{
		// Idea
		// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

		// TO DO
		// Handle error : free memory and exit
		// ...
		std::cout << "ERROR. Check transfer function filename : " << filename.toLatin1().constData() << std::endl;
	}
	setTransferFunctionFilename( filename.toLatin1().constData() );

	return true;
}

/******************************************************************************
 * Finalize the transfer function
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::finalizeTransferFunction()
{
	// Free memory
	delete _transferFunction;

	return true;
}

/******************************************************************************
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void SampleCore::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& nodeTileResolution = _pipeline->getDataStructure()->getNodeTileResolution().get();

	pX = nodeTileResolution.x;
	pY = nodeTileResolution.y;
	pZ = nodeTileResolution.z;
}

/******************************************************************************
 * Get the brick resolution of the data structure (voxels).
 *
 * @param pX the X brick resolution
 * @param pY the Y brick resolution
 * @param pZ the Z brick resolution
 ******************************************************************************/
void SampleCore::getDataStructureBrickResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& brickResolution = _pipeline->getDataStructure()->getBrickResolution().get();

	pX = brickResolution.x;
	pY = brickResolution.y;
	pZ = brickResolution.z;
}

/******************************************************************************
 * Tell whether or not the pipeline uses image downscaling.
 *
 * @return the flag telling whether or not the pipeline uses image downscaling
 ******************************************************************************/
bool SampleCore::hasImageDownscaling() const
{
	return _graphicsEnvironment->hasImageDownscaling();
}

/******************************************************************************
 * Set the flag telling whether or not the pipeline uses image downscaling
 *
 * @param pFlag the flag telling whether or not the pipeline uses image downscaling
 ******************************************************************************/
void SampleCore::setImageDownscaling( bool pFlag )
{
	// Update graphics environment
	_graphicsEnvironment->setImageDownscaling( pFlag );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::getViewportSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
	if ( _graphicsEnvironment != NULL )
	{
		pWidth = static_cast< unsigned int >( _graphicsEnvironment->getBufferWidth() );
		pHeight = static_cast< unsigned int >( _graphicsEnvironment->getBufferHeight() );
	}
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::setViewportSize( unsigned int pWidth, unsigned int pHeight )
{
	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setBufferSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Get the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::getGraphicsBufferSize( unsigned int& pWidth, unsigned int& pHeight ) const
{
	if ( _graphicsEnvironment != NULL )
	{
		pWidth = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
		pHeight = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );
	}
}

/******************************************************************************
 * Set the internal graphics buffer size
 *
 * @param pWidth the internal graphics buffer width
 * @param pHeight the internal graphics buffer height
 ******************************************************************************/
void SampleCore::setGraphicsBufferSize( unsigned int pWidth, unsigned int pHeight )
{
	// --------------------------
	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

	// Update graphics environment
	_graphicsEnvironment->setImageDownscalingSize( pWidth, pHeight );

	// Reset graphics resources
	resetGraphicsresources();
}

/******************************************************************************
 * Reset graphics resources
 ******************************************************************************/
void SampleCore::resetGraphicsresources()
{
	// [ 1 ] - Reset graphics resources

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();
	
	// Update graphics environment
	_graphicsEnvironment->reset();
	
	// Update internal variables
	_depthBuffer = _graphicsEnvironment->getDepthBuffer();
	_colorTex = _graphicsEnvironment->getColorTexture();
	_colorRenderBuffer = _graphicsEnvironment->getColorRenderBuffer();
	_depthTex = _graphicsEnvironment->getDepthTexture();
	_frameBuffer = _graphicsEnvironment->getFrameBuffer();
	
	// [ 2 ] - Connect graphics resources

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
	}
}

/******************************************************************************
 * Get the node cache memory
 *
 * @return the node cache memory
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheMemory() const
{
	return NODEPOOL_MEMSIZE / ( 1024U * 1024U );
}

/******************************************************************************
 * Set the node cache memory
 *
 * @param pValue the node cache memory
 ******************************************************************************/
void SampleCore::setNodeCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
 * Get the brick cache memory
 *
 * @return the brick cache memory
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheMemory() const
{
	return BRICKPOOL_MEMSIZE / ( 1024U * 1024U );
}

/******************************************************************************
 * Set the brick cache memory
 *
 * @param pValue the brick cache memory
 ******************************************************************************/
void SampleCore::setBrickCacheMemory( unsigned int pValue )
{
}

/******************************************************************************
 * Get the node cache capacity
 *
 * @return the node cache capacity
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheCapacity() const
{
	return _pipeline->getCache()->getNodesCacheManager()->getNumElements();
}

/******************************************************************************
 * Set the node cache capacity
 *
 * @param pValue the node cache capacity
 ******************************************************************************/
void SampleCore::setNodeCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the brick cache capacity
 *
 * @return the brick cache capacity
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheCapacity() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getNumElements();
}

/******************************************************************************
 * Set the brick cache capacity
 *
 * @param pValue the brick cache capacity
 ******************************************************************************/
void SampleCore::setBrickCacheCapacity( unsigned int pValue )
{
}

/******************************************************************************
 * Get the number of unused nodes in cache
 *
 * @return the number of unused nodes in cache
 ******************************************************************************/
unsigned int SampleCore::getCacheNbUnusedNodes() const
{
	return _pipeline->getCache()->getNodesCacheManager()->getNbUnusedElements();
}

/******************************************************************************
 * Get the number of unused bricks in cache
 *
 * @return the number of unused bricks in cache
 ******************************************************************************/
unsigned int SampleCore::getCacheNbUnusedBricks() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getNbUnusedElements();
}

/******************************************************************************
 * Get the max number of requests of node subdivisions.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbNodeSubdivisions() const
{
	return _pipeline->getCache()->getMaxNbNodeSubdivisions();
}

/******************************************************************************
 * Set the max number of requests of node subdivisions.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbNodeSubdivisions( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbNodeSubdivisions( pValue );
}

/******************************************************************************
 * Get the max number of requests of brick of voxel loads.
 *
 * @return the max number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheMaxNbBrickLoads() const
{
	return _pipeline->getCache()->getMaxNbBrickLoads();
}

/******************************************************************************
 * Set the max number of requests of brick of voxel loads.
 *
 * @param pValue the max number of requests
 ******************************************************************************/
void SampleCore::setCacheMaxNbBrickLoads( unsigned int pValue )
{
	_pipeline->editCache()->setMaxNbBrickLoads( pValue );
}

/******************************************************************************
 * Get the number of requests of node subdivisions the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheNbNodeSubdivisionRequests() const
{
	return _pipeline->getCache()->getNbNodeSubdivisionRequests();
}

/******************************************************************************
 * Get the number of requests of brick of voxel loads the cache has handled.
 *
 * @return the number of requests
 ******************************************************************************/
unsigned int SampleCore::getCacheNbBrickLoadRequests() const
{
	return _pipeline->getCache()->getNbBrickLoadRequests();
}

/******************************************************************************
 * Get the cache policy
 *
 * @return the cache policy
 ******************************************************************************/
unsigned int SampleCore::getCachePolicy() const
{
	return _pipeline->getCache()->getBricksCacheManager()->getPolicy();
}

/******************************************************************************
 * Set the cache policy
 *
 * @param pValue the cache policy
 ******************************************************************************/
void SampleCore::setCachePolicy( unsigned int pValue )
{
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( static_cast< PipelineType::CacheType::NodesCacheManager::ECachePolicy>( pValue ) );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( static_cast< PipelineType::CacheType::BricksCacheManager::ECachePolicy>( pValue ) );
}

/******************************************************************************
 * Get the nodes cache usage
 *
 * @return the nodes cache usage
 ******************************************************************************/
unsigned int SampleCore::getNodeCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getNodesCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getNodesCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "NODE cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the bricks cache usage
 *
 * @return the bricks cache usage
 ******************************************************************************/
unsigned int SampleCore::getBrickCacheUsage() const
{
	//const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_totalNumLoads;
	const unsigned int nbProducedElements = _pipeline->getCache()->getBricksCacheManager()->_numElemsNotUsed;
	const unsigned int nbElements = _pipeline->getCache()->getBricksCacheManager()->getNumElements();

	const unsigned int cacheUsage = static_cast< unsigned int >( 100.0f * static_cast< float >( nbElements - nbProducedElements ) / static_cast< float >( nbElements ) );

	//std::cout << "BRICK cache usage [ " << nbProducedElements << " / "<< nbElements << " : " << cacheUsage << std::endl;

	return cacheUsage;
}

/******************************************************************************
 * Get the number of tree leaf nodes
 *
 * @return the number of tree leaf nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeLeafNodes() const
{
	return _pipeline->getCache()->_nbLeafNodes;
}

/******************************************************************************
 * Get the number of tree nodes
 *
 * @return the number of tree nodes
 ******************************************************************************/
unsigned int SampleCore::getNbTreeNodes() const
{
	return _pipeline->getCache()->_nbNodes;
}

/******************************************************************************
* Get the flag indicating whether or not data production monitoring is activated
*
* @return the flag indicating whether or not data production monitoring is activated
 ******************************************************************************/
bool SampleCore::hasDataProductionMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating whether or not data production monitoring is activated
*
* @param pFlag the flag indicating whether or not data production monitoring is activated
 ******************************************************************************/
void SampleCore::setDataProductionMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating whether or not cache monitoring is activated
*
* @return the flag indicating whether or not cache monitoring is activated
 ******************************************************************************/
bool SampleCore::hasCacheMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating whether or not cache monitoring is activated
*
* @param pFlag the flag indicating whether or not cache monitoring is activated
 ******************************************************************************/
void SampleCore::setCacheMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating whether or not time budget monitoring is activated
*
* @return the flag indicating whether or not time budget monitoring is activated
 ******************************************************************************/
bool SampleCore::hasTimeBudgetMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating whether or not time budget monitoring is activated
*
* @param pFlag the flag indicating whether or not time budget monitoring is activated
 ******************************************************************************/
void SampleCore::setTimeBudgetMonitoring( bool pFlag )
{
}

/******************************************************************************
 *Tell whether or not time budget is acivated
 *
 * @return a flag to tell whether or not time budget is activated
 ******************************************************************************/
bool SampleCore::hasRenderingTimeBudget() const
{
	return true;
}

/******************************************************************************
 * Set the flag telling whether or not time budget is acivated
 *
 * @param pFlag a flag to tell whether or not time budget is activated
 ******************************************************************************/
void SampleCore::setRenderingTimeBudgetActivated( bool pFlag )
{
}

/******************************************************************************
 * Get the user requested time budget
 *
 * @return the user requested time budget
 ******************************************************************************/
unsigned int SampleCore::getRenderingTimeBudget() const
{
	return static_cast< unsigned int >( /*_pipeline->getRenderer()*/_renderer->getTimeBudget() );
}

/******************************************************************************
 * Set the user requested time budget
 *
 * @param pValue the user requested time budget
 ******************************************************************************/
void SampleCore::setRenderingTimeBudget( unsigned int pValue )
{
	/*_pipeline->editRenderer()*/_renderer->setTimeBudget( static_cast< float >( pValue ) );
}

/******************************************************************************
 * This method return the duration of the timer event between start and stop event
 *
 * @return the duration of the event in milliseconds
 ******************************************************************************/
float SampleCore::getRendererElapsedTime() const
{
	return /*_pipeline->editRenderer()*/_renderer->getElapsedTime();
}

/******************************************************************************
 * Tell whether or not pipeline uses programmable shaders
 *
 * @return a flag telling whether or not pipeline uses programmable shaders
 ******************************************************************************/
bool SampleCore::hasProgrammableShaders() const
{
	return true;
}

/******************************************************************************
 * Tell whether or not pipeline has a given type of shader
 *
 * @param pShaderType the type of shader to test
 *
 * @return a flag telling whether or not pipeline has a given type of shader
 ******************************************************************************/
bool SampleCore::hasShaderType( unsigned int pShaderType ) const
{
	return _shaderProgram->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}

/******************************************************************************
 * Get the source code associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader source code
 ******************************************************************************/
std::string SampleCore::getShaderSourceCode( unsigned int pShaderType ) const
{
	return _shaderProgram->getShaderSourceCode( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}

/******************************************************************************
 * Get the filename associated to a given type of shader
 *
 * @param pShaderType the type of shader
 *
 * @return the associated shader filename
 ******************************************************************************/
std::string SampleCore::getShaderFilename( unsigned int pShaderType ) const
{
	return _shaderProgram->getShaderFilename( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}

/******************************************************************************
 * ...
 *
 * @param pShaderType the type of shader
 *
 * @return ...
 ******************************************************************************/
bool SampleCore::reloadShader( unsigned int pShaderType )
{
	return _shaderProgram->reloadShader( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}

/******************************************************************************
 * Tell whether or not the pipeline has a light.
 *
 * @return the flag telling whether or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::has3DModel() const
{
	return true;
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
string SampleCore::get3DModelFilename() const
{
	return _filename;
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	_filename = pFilename;
}

/******************************************************************************
 * Get the 3D model resolution
 *
 * @return the 3D model resolution
 ******************************************************************************/
unsigned int SampleCore::get3DModelResolution() const
{
	return _resolution;
}

/******************************************************************************
 * Set the 3D model resolution
 *
 * @param pValue the 3D model resolution
 ******************************************************************************/
void SampleCore::set3DModelResolution( unsigned int pValue )
{
	_resolution = pValue;
}
