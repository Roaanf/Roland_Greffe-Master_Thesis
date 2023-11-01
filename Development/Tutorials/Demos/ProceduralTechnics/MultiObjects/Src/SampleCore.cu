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
#include <GvUtils/GsSimpleHostProducer.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsCommonGraphicsPass.h>
#include <GvCore/GsError.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GsGraphics/GsShaderProgram.h>
#include <GvUtils/GsEnvironment.h>

// Project
#include "ProducerKernel.h"
#include /*ProducerKernel2*/"ProducerTorusKernel.h"
#include "ShaderKernel.h"

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
template< typename TShaderType, typename TDataStructureType, typename TCacheType >
MyPipeline< TShaderType, TDataStructureType, TCacheType >
::MyPipeline()
:	GvUtils::GsSimplePipeline< TShaderType, TDataStructureType, TCacheType >()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TShaderType, typename TDataStructureType, typename TCacheType >
MyPipeline< TShaderType, TDataStructureType, TCacheType >
::~MyPipeline()
{
}

/******************************************************************************
 * Launch the main GigaSpace flow sequence
 ******************************************************************************/
template< typename TShaderType, typename TDataStructureType, typename TCacheType >
void MyPipeline< TShaderType, TDataStructureType, TCacheType >
::execute( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Check if a "clear request" has been asked
	if ( this->_clearRequested )
	{
		CUDAPM_START_EVENT( gpucache_clear );

		// Clear the cache
		this->_cache->clearCache();

		// Bug [#16161] "Cache : not cleared as it should be"
		//
		// Without the following clear of the data structure node pool, artefacts should appear.
		// - it is visible in the Slisesix and ProceduralTerrain demos.
		//
		// It seems that the brick addresses of the node pool need to be reset.
		// Maybe it's a problem of "time stamp" and index of current frame (or time).
		// 
		// @todo : study this problem
		this->_dataStructure->clearVolTree();

		CUDAPM_STOP_EVENT( gpucache_clear );

		// Update "clear request" flag
		this->_clearRequested = false;
	}

	// [ 1 ] - Pre-render stage
	this->_cache->preRenderPass();

	// [ 2 ] - Rendering stage
	//for ( size_t i = 0; i < _renderers.size(); i++ )
	//{
		// Object #1
		unsigned int maxVolTreeDepth = 5;
		GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
		unsigned int _producerIndex = 0;
		unsigned int objectID = 1; // BEWARE : it must be différent from 0 (or use signed int instead)
		this->setCurrentProducer( this->_producers[ _producerIndex ] );
		this->_dataStructure->volumeTreeKernel._rootAddress = ( _producerIndex + 1 ) * /*NodeTileRes::getNumElements()*/8;
#ifdef GS_USE_MULTI_OBJECTS
		GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_objectID, &/*_producerIndex*/objectID, sizeof( unsigned int ), 0, cudaMemcpyHostToDevice ) );
#endif
		this->_renderers[ 0 ]->preRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		this->_renderers[ 0 ]->render( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		this->_renderers[ 0 ]->postRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		
		// Object #2
		_producerIndex = 1;
		this->setCurrentProducer( this->_producers[ _producerIndex ] );
		this->_dataStructure->volumeTreeKernel._rootAddress = ( _producerIndex + 1 ) * /*NodeTileRes::getNumElements()*/8;
		objectID = 2; // BEWARE : it must be différent from 0 (or use signed int instead)
#ifdef GS_USE_MULTI_OBJECTS
		GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_objectID, &/*_producerIndex*/objectID, sizeof( unsigned int ), 0, cudaMemcpyHostToDevice ) );
#endif
		// Build the world transformation matrix
		float4x4 modelMatrix;
		{
			glPushMatrix();
			glLoadIdentity();
			glTranslatef( -2.5f, -1.5f, -0.5f );
			glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
			glPopMatrix();
		}
		maxVolTreeDepth = 4;
		GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_maxVolTreeDepth, &maxVolTreeDepth, sizeof( maxVolTreeDepth ), 0, cudaMemcpyHostToDevice) );
		this->_renderers[ 0 ]->preRender( modelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		this->_renderers[ 0 ]->render( modelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		this->_renderers[ 0 ]->postRender( modelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		//_renderers[ 0 ]->preRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		//_renderers[ 0 ]->render( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
		//_renderers[ 0 ]->postRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
	//}

	// [ 3 ] - Post-render stage (i.e. Data Production Management)
	CUDAPM_START_EVENT( dataProduction_handleRequests );
	if ( this->_dynamicUpdate )
	{
		this->_cache->_intraFramePass = false;

		// Post render pass
		// This is where requests are processed : produce or load data
		this->_cache->handleRequests();
	}
	CUDAPM_STOP_EVENT( dataProduction_handleRequests );
}

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
SampleCore::SampleCore()
:	_pipeline( NULL )
,	_renderer( NULL )
,	_graphicsEnvironment( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 6 )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_width( 0 )
,	_height( 0 )
,	_shaderProgram( NULL )
,	_fullscreenQuadVAO( 0 )
,	_fullscreenQuadVertexBuffer( 0 )
,	_textureSamplerUniformLocation( 0 )
{
	// Initialize matrices
	memset( _modelMatrix._array, 0, 16 * sizeof( float ) );
	memset( _modelViewMatrix._array, 0, 16 * sizeof( float ) );
	memset( _projectionMatrix._array, 0, 16 * sizeof( float ) );
	_viewport = make_int4( 0 );
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
	delete _graphicsEnvironment;
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
	//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
	cudaSetDevice( gpuGetMaxGflopsDeviceId() );
	GV_CHECK_CUDA_ERROR( "cudaSetDevice" );

	// Pipeline creation
	_pipeline = new PipelineType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, shader );
	// TEST
	_pipeline->editCache()->setNbObjects( 2 );

	// Producer initialization
	_producer = new ProducerType();
	assert( _producer != NULL );
	_pipeline->addProducer( _producer );
	_producer2 = new ProducerType2();
	assert( _producer2 != NULL );
	_pipeline->addProducer( _producer2 );

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	
	// Graphics environment creation
	_graphicsEnvironment = new GsCommonGraphicsPass();

	// Build default model transformation matrix
	// - translation to center GigaSpace BBox (originally in [0;1]x[0;1]x[0;1])
	_modelMatrix._array[ 0 ] = 1.0f;
	_modelMatrix._array[ 5 ] = 1.0f;
	_modelMatrix._array[ 10 ] = 1.0f;
	_modelMatrix._array[ 15 ] = 1.0f;
	_modelMatrix._array[ 12 ] = -0.5f;
	_modelMatrix._array[ 13 ] = -0.5f;
	_modelMatrix._array[ 14 ] = -0.5f;

	// Create and link a GLSL shader program
	QString shaderRepository = GsEnvironment::getDataDir( GsEnvironment::eShadersDir ).c_str();
	shaderRepository += QDir::separator();
	shaderRepository += QString( "SimpleSphere" );
	shaderRepository += QDir::separator();
	// Initialize points shader program
	QString vertexShaderFilename = shaderRepository + QString( "fullscreenQuad_vert.glsl" );
	QString fragmentShaderFilename = shaderRepository + QString( "fullscreenQuad_frag.glsl" );
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
	// Avoid redundant call by initializing data
	// - by default, after been linked, a program has already its uniforms set to 0
	_shaderProgram->use();
	glUniform1i( _textureSamplerUniformLocation, 0 );
	GsShaderProgram::unuse();

	// Vertex position buffer initialization
	glGenBuffers( 1, &_fullscreenQuadVertexBuffer );
	glBindBuffer( GL_ARRAY_BUFFER, _fullscreenQuadVertexBuffer );
	GLsizeiptr fullscreenQuadVertexBufferSize = sizeof( GLfloat ) * 4/*nbVertices*/ * 2/*nb components per vertex*/;
	float2 fullscreenQuadVertices[ 4/*nbVertices*/ ] =
	{
		{ -1.0, -1.0 },
		{ 1.0, -1.0 },
		{ 1.0, 1.0 },
		{ -1.0, 1.0 }
	};
	glBufferData( GL_ARRAY_BUFFER, fullscreenQuadVertexBufferSize, &fullscreenQuadVertices[ 0 ], GL_STATIC_DRAW );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );

	// Vertex array object initialization
	glGenVertexArrays( 1, &_fullscreenQuadVAO );
	glBindVertexArray( _fullscreenQuadVAO );
	glEnableVertexAttribArray( 0 );	// vertex position
	glBindBuffer( GL_ARRAY_BUFFER, _fullscreenQuadVertexBuffer );
	glVertexAttribPointer( 0/*attribute index*/, 2/*nb components per vertex*/, GL_FLOAT/*type*/, GL_FALSE/*un-normalized*/, 0/*memory stride*/, static_cast< GLubyte* >( NULL )/*byte offset from buffer*/ );
	glBindBuffer( GL_ARRAY_BUFFER, 0 );
	glBindVertexArray( 0 );

	// Avoid redundant call by initializing data
	glActiveTexture( GL_TEXTURE0 );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	//glMatrixMode( GL_MODELVIEW );	// already done by QGLViewer
	
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
		//glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );	// deprecated
		GV_CHECK_GL_ERROR();
	}
	else
	{
		//glClear( GL_COLOR_BUFFER_BIT );
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
		glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
		glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );	// deprecated
		GV_CHECK_GL_ERROR();
	}
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	//----------------------------------------------------------------------------------------------------
	// TO DO
	// - optimization : don't use glGetFloatv() it's slow, store matrices with glm (check NSight)
	//
	// - QGLViewer has already theses matrices in preDraw()
	//----------------------------------------------------------------------------------------------------

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Render
	//_pipeline->execute( _modelMatrix, _modelViewMatrix, _projectionMatrix, _viewport );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	{
		glPushMatrix();
		glLoadIdentity();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
		glPopMatrix();
	}
	// Render
	//_pipeline->setCurrentProducer( _producer );
	//unsigned int _producerIndex = 0;
	//_pipeline->editDataStructure()->volumeTreeKernel._rootAddress = ( _producerIndex + 1 ) * /*NodeTileRes::getNumElements()*/8;
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_objectID, &_producerIndex, sizeof( unsigned int ), 0, cudaMemcpyHostToDevice ) );
	_pipeline->execute( modelMatrix, _modelViewMatrix, _projectionMatrix, _viewport );

	// Build the world transformation matrix
	{
		glPushMatrix();
		glLoadIdentity();
		glTranslatef( -2.5f, -1.5f, -0.5f );
		glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
		glPopMatrix();
	}
	// Render
	//_pipeline->setCurrentProducer( _producer2 );
	//_producerIndex = 1;
	//_pipeline->editDataStructure()->volumeTreeKernel._rootAddress = ( _producerIndex + 1 ) * /*NodeTileRes::getNumElements()*/8;
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbolAsync( k_objectID, &_producerIndex, sizeof( unsigned int ), 0, cudaMemcpyHostToDevice ) );
	//_pipeline->execute( modelMatrix, _modelViewMatrix, _projectionMatrix, _viewport );

	// Draw fullscreen textured quad
	_shaderProgram->use();
	//glUniform1i( _textureSamplerUniformLocation, 0 );	// avoid redundant call
	//glActiveTexture( GL_TEXTURE0 );	// avoid redundant call
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	glBindVertexArray( _fullscreenQuadVAO );
	glDrawArrays( GL_QUADS, 0, 4 );
	glBindVertexArray( 0 );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );	// deprecated
	GsShaderProgram::unuse();

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

	// Update internal viewport
	_viewport = make_int4( 0, 0, pWidth, pHeight );

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
		///*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		///*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadWriteSlot, _depthBuffer );
	}
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
		///*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );

		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		///*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadWriteSlot, _depthBuffer );
	}
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
 * Set the light position
 *
 * @param pX the X light position
 * @param pY the Y light position
 * @param pZ the Z light position
 ******************************************************************************/
void SampleCore::setLightPosition( float pX, float pY, float pZ )
{
	// Update DEVICE memory with "light position"
	float3  lightPosition = make_float3( pX, pY, pZ );
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPosition, sizeof( lightPosition ), 0, cudaMemcpyHostToDevice ) );
}
