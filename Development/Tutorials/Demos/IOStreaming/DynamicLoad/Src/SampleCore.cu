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

// Gigavoxels
#include <GvCore/GsVector.h>
#include <GvStructure/GsVolumeTree.h>
#include <GvStructure/GsDataProductionManager.h>
#include <GvRendering/GsRendererCUDA.h>
#include <GvUtils/GsSimplePipeline.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsDataLoader.h>
#include <GvUtils/GsCommonGraphicsPass.h>
#include <GvCore/GsError.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvStructure/GsNode.h>
#include <GvUtils/GsEnvironment.h>

// Project
#include "Producer.h"
#include "ShaderKernel.h"

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

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 1024U * 1024U * 1024U )		// 1024 Mo
#define BRICKPOOL_MEMSIZE	( 2048U * 1024U * 1024U )		// 2048 Mo

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
,	_renderer( NULL )
,	mColorTex( 0 )
,	mDepthTex( 0 )
,	_frameBuffer( 0 )
,	_inputDepthBuffer( 0 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 16 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	delete _pipeline;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize CUDA with OpenGL Interoperability
	//cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );	// to do : deprecated, use cudaSetDevice()
	//GV_CHECK_CUDA_ERROR( "cudaGLSetGLDevice" );
	cudaSetDevice( gpuGetMaxGflopsDeviceId() );
	GV_CHECK_CUDA_ERROR( "cudaSetDevice" );

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = PipelineType::NodeTileResolution::numElements * sizeof( GvStructure::GsNode );
	//size_t brickElemSize = PipelineType::RealBrickTileResolution::numElements * GvCore::DataTotalChannelSize< PipelineType::DataTypeList >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3((uint)floorf(powf((float)nodePoolNumElems, 1.0f / 3.0f))) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3((uint)floorf(powf((float)brickPoolNumElems, 1.0f / 3.0f))) * RealBrickRes::get();

	//std::cout << "" << std::endl;
	//std::cout << "nodePoolRes: " << nodePoolRes << std::endl;
	//std::cout << "brickPoolRes: " << brickPoolRes << std::endl;

	// Pipeline creation
	_pipeline = new PipelineType();

	// Producer creation
	//QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "xyzrgb_dragon512_BR8_B1" ) + QDir::separator() + QString( "xyzrgb_dragon.xml" );
	QString filename = GsEnvironment::getDataDir( GsEnvironment::eVoxelsDir ).c_str();
	filename += QDir::separator();
	filename += QString( "xyzrgb_dragon512_BR8_B1" );
	filename += QDir::separator();
	filename += QString( "xyzrgb_dragon.xml" );
	GvUtils::GsDataLoader< DataType >* dataLoader = new GvUtils::GsDataLoader< DataType >(
														filename.toStdString(),
														PipelineType::BrickTileResolution::get(), PipelineType::BrickTileBorderSize, true );
													//	make_uint3( 512 ), BrickRes::get(), BrickBorderSize, false );
		
	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
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
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editCache()->setMaxNbNodeSubdivisions( 500 );
	_pipeline->editCache()->setMaxNbBrickLoads( 300 );
	_pipeline->editCache()->editNodesCacheManager()->setPolicy( PipelineType::CacheType::NodesCacheManager::eAllPolicies );
	_pipeline->editCache()->editBricksCacheManager()->setPolicy( PipelineType::CacheType::BricksCacheManager::eAllPolicies );
}

/******************************************************************************
 * ...
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
		glTranslatef( -0.5f, -0.5f, -0.5f );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
		glDisable( GL_DEPTH_TEST );

		// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
		glBindBuffer( GL_PIXEL_PACK_BUFFER, _inputDepthBuffer );
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
	glGetFloatv(GL_MODELVIEW_MATRIX, viewMatrix._array);
	glGetFloatv(GL_PROJECTION_MATRIX, projectionMatrix._array);

	// build and extract tree transformations
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	glTranslatef(-0.5f, -0.5f, -0.5f);
	glGetFloatv(GL_MODELVIEW_MATRIX, modelMatrix._array);
	glPopMatrix();

	// extract viewport
	GLint params[4];
	glGetIntegerv(GL_VIEWPORT, params);
	int4 viewport = make_int4(params[0], params[1], params[2], params[3]);

	CUDAPM_STOP_EVENT( app_init_frame );

	// render the scene into textures
	_pipeline->execute(modelMatrix, viewMatrix, projectionMatrix, viewport);

	// Render the result to the screen
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();

	glEnable(GL_TEXTURE_RECTANGLE_EXT);
	glDisable(GL_DEPTH_TEST);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, mColorTex );

	GLint sMin = 0;
	GLint tMin = 0;
	GLint sMax = _width;
	GLint tMax = _height;

	glBegin(GL_QUADS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glTexCoord2i(sMin, tMin); glVertex2i(-1, -1);
		glTexCoord2i(sMax, tMin); glVertex2i( 1, -1);
		glTexCoord2i(sMax, tMax); glVertex2i( 1,  1);
		glTexCoord2i(sMin, tMax); glVertex2i(-1,  1);
	glEnd();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);

	glDisable(GL_TEXTURE_RECTANGLE_EXT);

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// TEST - optimization due to early unmap() graphics resource from GigaVoxels
	///*_pipeline->editRenderer()*/_renderer->doPostRender();
		
	//mVolumeTreeRenderer->nextFrame();
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	if (_displayPerfmon)
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL(_displayPerfmon - 1);
	}
}

/******************************************************************************
 * Resize the frame
 *
 * @param width the new width
 * @param height the new height
 ******************************************************************************/
void SampleCore::resize( int width, int height )
{
	_width = width;
	_height = height;

	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );
	
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
	cudaMemset(timersMask, 255, _width * _height);*/

	// Create frame-dependent objects

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();
	
	// ...
	if (_inputDepthBuffer)
		glDeleteBuffers(1, &_inputDepthBuffer);

	if (mColorTex)
		glDeleteTextures(1, &mColorTex);
	if (mDepthTex)
		glDeleteTextures(1, &mDepthTex);

	if (_frameBuffer)
		glDeleteFramebuffers(1, &_frameBuffer);

	glGenTextures(1, &mColorTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mColorTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &_inputDepthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, _inputDepthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);	// modifier le param�tre du depth PBO pour de la elcture seule !!! STREAM_READ
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures(1, &mDepthTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, mDepthTex);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL);
	glBindTexture(GL_TEXTURE_RECTANGLE_EXT, 0);
	GV_CHECK_GL_ERROR();

	glGenFramebuffers( 1, &_frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, mColorTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, mDepthTex, 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	if ( _displayOctree )
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _inputDepthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->clear();
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	if ( _displayOctree )
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _inputDepthBuffer );
	}
	else
	{
		/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorWriteSlot, mColorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	const bool status = _pipeline->hasDynamicUpdate();
	_pipeline->setDynamicUpdate( ! status );
}

/******************************************************************************
 * ...
 *
 * @param mode ...
 ******************************************************************************/
void SampleCore::togglePerfmonDisplay( uint mode )
{
	if (_displayPerfmon)
		_displayPerfmon = 0;
	else
		_displayPerfmon = mode;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if (_maxVolTreeDepth < 32)
		_maxVolTreeDepth++;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if (_maxVolTreeDepth > 0)
		_maxVolTreeDepth--;

	//mVolumeTree->setMaxDepth( _maxVolTreeDepth );
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
	//
	// WARNING
	// Apply inverse modelisation matrix applied on the GigaVoxels object to set light position correctly.
	// Here a glTranslatef( -0.5f, -0.5f, -0.5f ) has been used.
	float3 translation = make_float3( -0.5f, -0.5f, -0.5f );
	float3 lightPosition = make_float3( pX, pY, pZ ) - translation;
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPosition, sizeof( lightPosition ), 0, cudaMemcpyHostToDevice ) );
}
