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
#include <GvUtils/GsSimplePipeline.h>
#include <GvUtils/GsSimpleHostProducer.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsCommonGraphicsPass.h>
#include <GvCore/GsError.h>
#include <GvPerfMon/GsPerformanceMonitor.h>

// Project
#include "ProducerKernel.h"
#include "ShaderKernel.h"
#include "RendererCUDA.h"
#include "ProxyGeometry.h"
#include "Mesh.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

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

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	_pipeline( NULL )
,	_producer( NULL )
,	_renderer( NULL )
,	_graphicsEnvironment( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 5 )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_width( 0 )
,	_height( 0 )
,	_proxyGeometry( NULL )
{
	// Translation used to position the GigaVoxels data structure
	_translation[ 0 ] = -0.5f;
	_translation[ 1 ] = -0.5f;
	_translation[ 2 ] = -0.5f;

	// Light position
	_lightPosition = make_float3( 1.f, 1.f, 1.f );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
	delete _graphicsEnvironment;

	delete _proxyGeometry;
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "Proxy Geometry Manager";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Pipeline creation
	_pipeline = new PipelineType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, shader );

	// Producer initialization
	_producer = new ProducerType();
	assert( _producer != NULL );
	_pipeline->addProducer( _producer );

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Configure the Cache Management System
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 500 );
	_pipeline->editCache()->setMaxNbBrickLoads( 300 );
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( DataProductionManagerType::NodesCacheManager::eAllPolicies );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( DataProductionManagerType::BricksCacheManager::eAllPolicies );
	
	// Graphics environment creation
	_graphicsEnvironment = new GsCommonGraphicsPass();

	// Initialize proxy geometry
	//
	// - find a way to modify internal buffer size
	_proxyGeometry = new ProxyGeometry();
	const QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	const QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" );
	const QString meshFilename = meshRepository + QDir::separator() + QString( "bunny.obj" );
	_proxyGeometry->set3DModelFilename( meshFilename.toStdString() );
	_proxyGeometry->initialize();
	// Restore previous proxy geometry state
	_proxyGeometry->setScreenBasedCriteria( false );
	_proxyGeometry->setScreenBasedCriteriaCoefficient( 1.f );
	_proxyGeometry->setMaterialAlphaCorrectionCoefficient( 512.f );
	// Register proxy geometry
	/*_pipeline->editRenderer()*/_renderer->setProxyGeometry( _proxyGeometry );

	// Noise parameters
	setNoiseFirstFrequency( 3.61f );
	setNoiseStrength( 44.8f );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glMatrixMode( GL_MODELVIEW );
	
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
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
		//glClear( GL_COLOR_BUFFER_BIT );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
	}
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// extract viewport
	GLint params[ 4 ];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[ 0 ], params[ 1 ], params[ 2 ], params[ 3 ] );

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Generate depth maps from mesh
	// - min depth from closest faces
	// - max depth from farthest faces
	// => we get a "shell" from the mesh
	glEnable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );
	float4x4 proxyGeometryModelViewMatrix;
	glPushMatrix();
	// Add Model transformation to lie between -0.5 and 0.5
	const IMesh* mesh = _proxyGeometry->getMesh();
	const float minX = mesh->_minX;
	const float minY = mesh->_minY;
	const float minZ = mesh->_minZ;
	const float maxX = mesh->_maxX;
	const float maxY = mesh->_maxY;
	const float maxZ = mesh->_maxZ;
	const float uniformScale = 0.99f / std::max( std::max( maxX - minX, maxY - minY ), maxZ - minZ );
	glScalef( uniformScale, uniformScale, uniformScale );
	const float3 translate = make_float3( - ( minX + maxX ) * 0.5f, - ( minY + maxY ) * 0.5f, - ( minZ + maxZ ) * 0.5f );
	glTranslatef( translate.x, translate.y, translate.z );
	glGetFloatv( GL_MODELVIEW_MATRIX, proxyGeometryModelViewMatrix._array );
	// TO DO : add a screen based criteria to stop division => ...
	_proxyGeometry->render( proxyGeometryModelViewMatrix, projectionMatrix, viewport );
	glPopMatrix();

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	// Render the result to the screen
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	glDisable( GL_DEPTH_TEST );
	glEnable( GL_TEXTURE_RECTANGLE_EXT );
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	
	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;
	glBegin( GL_QUADS );
	glColor3f( 1.0f, 1.0f, 1.0f );
	glTexCoord2i( sMin, tMin ); glVertex2i( -1, -1 );
	glTexCoord2i( sMax, tMin ); glVertex2i(  1, -1 );
	glTexCoord2i( sMax, tMax ); glVertex2i(  1,  1 );
	glTexCoord2i( sMin, tMax ); glVertex2i( -1,  1 );
	glEnd();

	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	glDisable( GL_TEXTURE_RECTANGLE_EXT );
	
	glPopMatrix();
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	///*_pipeline->editRenderer()*/_renderer->doPostRender();
	
	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
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

	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );

	// Update graphics environment
	_graphicsEnvironment->setBufferSize( pWidth, pHeight );

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
	//_colorRenderBuffer = _graphicsEnvironment->getColorRenderBuffer();
	_depthTex = _graphicsEnvironment->getDepthTexture();
	_frameBuffer = _graphicsEnvironment->getFrameBuffer();
	
	// [ 2 ] - Connect graphics resources

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}

	// Reset proxy geometry resources
	/*_pipeline->editRenderer()*/_renderer->unregisterProxyGeometryGraphicsResources();
	_proxyGeometry->setBufferSize( _width, _height );
	/*_pipeline->editRenderer()*/_renderer->registerProxyGeometryGraphicsResources();
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
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * Get the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::getDataStructureAppearance( bool& pShowNodeHasBrickTerminal, bool& pShowNodeHasBrickNotTerminal, bool& pShowNodeIsBrickNotInCache, bool& pShowNodeEmptyOrConstant
											, float& pNodeHasBrickTerminalColorR, float& pNodeHasBrickTerminalColorG, float& pNodeHasBrickTerminalColorB, float& pNodeHasBrickTerminalColorA
											, float& pNodeHasBrickNotTerminalColorR, float& pNodeHasBrickNotTerminalColorG, float& pNodeHasBrickNotTerminalColorB, float& pNodeHasBrickNotTerminalColorA
											, float& pNodeIsBrickNotInCacheColorR, float& pNodeIsBrickNotInCacheColorG, float& pNodeIsBrickNotInCacheColorB, float& pNodeIsBrickNotInCacheColorA
											, float& pNodeEmptyOrConstantColorR, float& pNodeEmptyOrConstantColorG, float& pNodeEmptyOrConstantColorB, float& pNodeEmptyOrConstantColorA ) const
{
	float4 nodeHasBrickTerminalColor;
	float4 nodeHasBrickNotTerminalColor;
	float4 nodeIsBrickNotInCacheColor;
	float4 nodeEmptyOrConstantColor;
										
	_pipeline->getDataStructure()->getDataStructureAppearance( pShowNodeHasBrickTerminal, pShowNodeHasBrickNotTerminal, pShowNodeIsBrickNotInCache, pShowNodeEmptyOrConstant
											, nodeHasBrickTerminalColor, nodeHasBrickNotTerminalColor, nodeIsBrickNotInCacheColor, nodeEmptyOrConstantColor );

	pNodeHasBrickTerminalColorR = nodeHasBrickTerminalColor.x;
	pNodeHasBrickTerminalColorG = nodeHasBrickTerminalColor.y;
	pNodeHasBrickTerminalColorB = nodeHasBrickTerminalColor.z;
	pNodeHasBrickTerminalColorA = nodeHasBrickTerminalColor.w;

	pNodeHasBrickNotTerminalColorR = nodeHasBrickNotTerminalColor.x;
	pNodeHasBrickNotTerminalColorG = nodeHasBrickNotTerminalColor.y;
	pNodeHasBrickNotTerminalColorB = nodeHasBrickNotTerminalColor.z;
	pNodeHasBrickNotTerminalColorA = nodeHasBrickNotTerminalColor.w;

	pNodeIsBrickNotInCacheColorR = nodeIsBrickNotInCacheColor.x;
	pNodeIsBrickNotInCacheColorG = nodeIsBrickNotInCacheColor.y;
	pNodeIsBrickNotInCacheColorB = nodeIsBrickNotInCacheColor.z;
	pNodeIsBrickNotInCacheColorA = nodeIsBrickNotInCacheColor.w;

	pNodeEmptyOrConstantColorR = nodeEmptyOrConstantColor.x;
	pNodeEmptyOrConstantColorG = nodeEmptyOrConstantColor.y;
	pNodeEmptyOrConstantColorB = nodeEmptyOrConstantColor.z;
	pNodeEmptyOrConstantColorA = nodeEmptyOrConstantColor.w;
}

/******************************************************************************
 * Set the appearance of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::setDataStructureAppearance( bool pShowNodeHasBrickTerminal, bool pShowNodeHasBrickNotTerminal, bool pShowNodeIsBrickNotInCache, bool pShowNodeEmptyOrConstant
											, float pNodeHasBrickTerminalColorR, float pNodeHasBrickTerminalColorG, float pNodeHasBrickTerminalColorB, float pNodeHasBrickTerminalColorA
											, float pNodeHasBrickNotTerminalColorR, float pNodeHasBrickNotTerminalColorG, float pNodeHasBrickNotTerminalColorB, float pNodeHasBrickNotTerminalColorA
											, float pNodeIsBrickNotInCacheColorR, float pNodeIsBrickNotInCacheColorG, float pNodeIsBrickNotInCacheColorB, float pNodeIsBrickNotInCacheColorA
											, float pNodeEmptyOrConstantColorR, float pNodeEmptyOrConstantColorG, float pNodeEmptyOrConstantColorB, float pNodeEmptyOrConstantColorA )
{
	float4 nodeHasBrickTerminalColor = make_float4( pNodeHasBrickTerminalColorR, pNodeHasBrickTerminalColorG, pNodeHasBrickTerminalColorB, pNodeHasBrickTerminalColorA );
	float4 nodeHasBrickNotTerminalColor = make_float4( pNodeHasBrickNotTerminalColorR, pNodeHasBrickNotTerminalColorG, pNodeHasBrickNotTerminalColorB, pNodeHasBrickNotTerminalColorA );
	float4 nodeIsBrickNotInCacheColor = make_float4( pNodeIsBrickNotInCacheColorR, pNodeIsBrickNotInCacheColorG, pNodeIsBrickNotInCacheColorB, pNodeIsBrickNotInCacheColorA );
	float4 nodeEmptyOrConstantColor = make_float4( pNodeEmptyOrConstantColorR, pNodeEmptyOrConstantColorG, pNodeEmptyOrConstantColorB, pNodeEmptyOrConstantColorA );

	_pipeline->editDataStructure()->setDataStructureAppearance( pShowNodeHasBrickTerminal, pShowNodeHasBrickNotTerminal, pShowNodeIsBrickNotInCache, pShowNodeEmptyOrConstant
											, nodeHasBrickTerminalColor, nodeHasBrickNotTerminalColor, nodeIsBrickNotInCacheColor, nodeEmptyOrConstantColor );
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
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
	}
	else
	{
		_displayPerfmon = mode;
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
 * Get the node tile resolution of the data structure.
 *
 * @param pX the X node tile resolution
 * @param pY the Y node tile resolution
 * @param pZ the Z node tile resolution
 ******************************************************************************/
void SampleCore::getDataStructureNodeTileResolution( unsigned int& pX, unsigned int& pY, unsigned int& pZ ) const
{
	const uint3& nodeTileResolution = _pipeline->editDataStructure()->getNodeTileResolution().get();

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
	const uint3& brickResolution = _pipeline->editDataStructure()->getBrickResolution().get();

	pX = brickResolution.x;
	pY = brickResolution.y;
	pZ = brickResolution.z;
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
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
 ******************************************************************************/
bool SampleCore::hasLight() const
{
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
 * Get the translation used to position the GigaVoxels data structure
 *
 * @param pX the x componenet of the translation
 * @param pX the y componenet of the translation
 * @param pX the z componenet of the translation
 ******************************************************************************/
void SampleCore::getTranslation( float& pX, float& pY, float& pZ ) const
{
	pX = _translation[ 0 ];
	pY = _translation[ 1 ];
	pZ = _translation[ 2 ];
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
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( static_cast< DataProductionManagerType::NodesCacheManager::ECachePolicy >( pValue ) );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( static_cast< DataProductionManagerType::BricksCacheManager::ECachePolicy >( pValue ) );
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
 * Tell wheter or not the pipeline has a light.
 *
 * @return the flag telling wheter or not the pipeline has a light
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
	return _proxyGeometry->get3DModelFilename();
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void SampleCore::set3DModelFilename( const string& pFilename )
{
	// Store current proxy geometry state
	const bool screenBasedCriteria =_proxyGeometry->getScreenBasedCriteria();
	const float screenBasedCriteriaCoefficient =_proxyGeometry->getScreenBasedCriteriaCoefficient();
	const float materialAlphaCorrectionCoefficient =_proxyGeometry->getMaterialAlphaCorrectionCoefficient();
	
	// ---- Delete the 3D scene if needed ----
	
	if ( _proxyGeometry != NULL )
	{
		delete _proxyGeometry;
		_proxyGeometry = NULL;

		// Clear the GigaVoxels cache
		_pipeline->editCache()->clearCache();
	}

	// Initialize proxy geometry (load the 3D scene)
	//
	// - find a way to modify internal buffer size
	_proxyGeometry = new ProxyGeometry();
	_proxyGeometry->set3DModelFilename( pFilename );
	_proxyGeometry->initialize();
	// Restore previous proxy geometry state
	_proxyGeometry->setScreenBasedCriteria( screenBasedCriteria );
	_proxyGeometry->setScreenBasedCriteriaCoefficient( screenBasedCriteriaCoefficient );
	_proxyGeometry->setMaterialAlphaCorrectionCoefficient( materialAlphaCorrectionCoefficient );
	/*_pipeline->editRenderer()*/_renderer->setProxyGeometry( _proxyGeometry );
	// Reset proxy geometry resources
	/*_pipeline->editRenderer()*/_renderer->unregisterProxyGeometryGraphicsResources();
	_proxyGeometry->setBufferSize( _width, _height );
	/*_pipeline->editRenderer()*/_renderer->registerProxyGeometryGraphicsResources();
	// Noise parameters
	setNoiseFirstFrequency( _noiseFirstFrequency );
	setNoiseStrength( _noiseStrength );
}

/******************************************************************************
 * Tell wheter or not the screen-based criteria is activated
 *
 * @return a flag telling wheter or not the screen-based criteria is activated
 ******************************************************************************/
bool SampleCore::getScreenBasedCriteria() const
{
	assert( _proxyGeometry != NULL );

	return _proxyGeometry->getScreenBasedCriteria();
}

/******************************************************************************
 * Set the flag telling wheter or not the screen-based criteria is activated
 *
 * @param pFlag a flag telling wheter or not the screen-based criteria is activated
 ******************************************************************************/
void SampleCore::setScreenBasedCriteria( bool pFlag )
{
	assert( _proxyGeometry != NULL );

	_proxyGeometry->setScreenBasedCriteria( pFlag );

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cScreenBasedCriteria, &pFlag, sizeof( pFlag ), 0, cudaMemcpyHostToDevice ) );

	// Clear the GigaVoxels cache
	clearCache();
}

/******************************************************************************
 * Get the screen-based criteria coefficient
 *
 * @return the screen-based criteria coefficient
 ******************************************************************************/
float SampleCore::getScreenBasedCriteriaCoefficient() const
{
	assert( _proxyGeometry != NULL );

	return _proxyGeometry->getScreenBasedCriteriaCoefficient();
}

/******************************************************************************
 * Set the screen-based criteria coefficient
 *
 * @param pValue the screen-based criteria coefficient
 ******************************************************************************/
void SampleCore::setScreenBasedCriteriaCoefficient( float pValue )
{
	assert( _proxyGeometry != NULL );

	_proxyGeometry->setScreenBasedCriteriaCoefficient( pValue );

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cScreenBasedCriteriaCoefficient, &pValue, sizeof( pValue ), 0, cudaMemcpyHostToDevice ) );

	// Clear the GigaVoxels cache
	clearCache();
}

/******************************************************************************
 * Get the material alpha correction coefficient
 *
 * @return the material alpha correction coefficient
 ******************************************************************************/
float SampleCore::getMaterialAlphaCorrectionCoefficient() const
{
	assert( _proxyGeometry != NULL );

	return _proxyGeometry->getMaterialAlphaCorrectionCoefficient();
}

/******************************************************************************
 * Set the material alpha correction coefficient
 *
 * @param pValue the material alpha correction coefficient
 ******************************************************************************/
void SampleCore::setMaterialAlphaCorrectionCoefficient( float pValue )
{
	assert( _proxyGeometry != NULL );

	_proxyGeometry->setMaterialAlphaCorrectionCoefficient( pValue );

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cMaterialAlphaCorrectionCoefficient, &pValue, sizeof( pValue ), 0, cudaMemcpyHostToDevice ) );

	// No need to clear the cache
	// - this variable is evaluated in the shader, not in the producer.
}

/******************************************************************************
 * Get the noise first frequency
 *
 * @return the noise first frequency
 ******************************************************************************/
float SampleCore::getNoiseFirstFrequency() const
{
	return _noiseFirstFrequency;
}

/******************************************************************************
 * Set the noise first frequency
 *
 * @param pValue the noise first frequency
 ******************************************************************************/
void SampleCore::setNoiseFirstFrequency( float pValue )
{
	_noiseFirstFrequency = pValue;
	
	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseFirstFrequency, &_noiseFirstFrequency, sizeof( _noiseFirstFrequency ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the noise strength
 *
 * @return the noise strength
 ******************************************************************************/
float SampleCore::getNoiseStrength() const
{
	return _noiseStrength;
}

/******************************************************************************
 * Set the noise strength
 *
 * @param pValue the noise strength
 ******************************************************************************/
void SampleCore::setNoiseStrength( float pValue )
{
	_noiseStrength = pValue;
	
	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseStrength, &_noiseStrength, sizeof( _noiseStrength ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the voxel size multiplier
 *
 * @return the voxel size multiplier
 ******************************************************************************/
float SampleCore::getVoxelSizeMultiplier() const
{
	return /*_pipeline->editRenderer()*/_renderer->getVoxelSizeMultiplier();
}

/******************************************************************************
 * Set the voxel size multiplier
 *
 * @param the voxel size multiplier
 ******************************************************************************/
void SampleCore::setVoxelSizeMultiplier( float pValue )
{
	/*_pipeline->editRenderer()*/_renderer->setVoxelSizeMultiplier( pValue );
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
* Get the flag indicating wheter or not data production monitoring is activated
*
* @return the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
bool SampleCore::hasDataProductionMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not data production monitoring is activated
*
* @param pFlag the flag indicating wheter or not data production monitoring is activated
 ******************************************************************************/
void SampleCore::setDataProductionMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not cache monitoring is activated
*
* @return the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
bool SampleCore::hasCacheMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not cache monitoring is activated
*
* @param pFlag the flag indicating wheter or not cache monitoring is activated
 ******************************************************************************/
void SampleCore::setCacheMonitoring( bool pFlag )
{
}

/******************************************************************************
* Get the flag indicating wheter or not time budget monitoring is activated
*
* @return the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
bool SampleCore::hasTimeBudgetMonitoring() const
{
	return true;
}

/******************************************************************************
* Set the the flag indicating wheter or not time budget monitoring is activated
*
* @param pFlag the flag indicating wheter or not time budget monitoring is activated
 ******************************************************************************/
void SampleCore::setTimeBudgetMonitoring( bool pFlag )
{
}

/******************************************************************************
 *Tell wheter or not time budget is acivated
 *
 * @return a flag to tell wheter or not time budget is activated
 ******************************************************************************/
bool SampleCore::hasRenderingTimeBudget() const
{
	return true;
}

/******************************************************************************
 * Set the flag telling wheter or not time budget is acivated
 *
 * @param pFlag a flag to tell wheter or not time budget is activated
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
 * Tell wheter or not pipeline uses programmable shaders
 *
 * @return a flag telling wheter or not pipeline uses programmable shaders
 ******************************************************************************/
bool SampleCore::hasProgrammableShaders() const
{
	return false;//true;
}

/******************************************************************************
 * Tell wheter or not pipeline has a given type of shader
 *
 * @param pShaderType the type of shader to test
 *
 * @return a flag telling wheter or not pipeline has a given type of shader
 ******************************************************************************/
bool SampleCore::hasShaderType( unsigned int pShaderType ) const
{
	return false;//_shaderProgram->hasShaderType( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return string();//_shaderProgram->getShaderSourceCode( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return string();// _shaderProgram->getShaderFilename( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
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
	return false;//_shaderProgram->reloadShader( static_cast< GsShaderProgram::ShaderType >( pShaderType ) );
}
