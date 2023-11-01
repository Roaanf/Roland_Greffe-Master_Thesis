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
#include <GvUtils/GsDataLoader.h>
#include <GvUtils/GsSimpleHostShader.h>
#include <GvUtils/GsSimplePriorityPoliciesManagerKernel.h>
#include <GvUtils/GsCommonGraphicsPass.h>
#include <GvCore/GsError.h>
#include <GvPerfMon/GsPerformanceMonitor.h>
#include <GvStructure/GsNode.h>
#include <GvUtils/GsEnvironment.h>

// Project
#include "Producer.h"
#include "VolumeTreeRendererGLSL.h"

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaSpace
using namespace GvUtils;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/**
 * Defines the size allowed for each type of pool
 */
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		// 8 Mo
#define BRICKPOOL_MEMSIZE	( 128U * 1024U * 1024U )	// 128 Mo

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
,	mDisplayOctree( false )
,	mDisplayPerfmon( 0 )
,	mMaxVolTreeDepth( 16 )
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

	// Compute the size of one element in the cache for nodes and bricks
	size_t nodeElemSize = NodeRes::numElements * sizeof( GvStructure::GsNode );
	//size_t brickElemSize = RealBrickRes::numElements * GvCore::DataTotalChannelSize< DataType >::value;

	// Compute how many we can fit into the given memory size
	size_t nodePoolNumElems = NODEPOOL_MEMSIZE / nodeElemSize;
	//size_t brickPoolNumElems = BRICKPOOL_MEMSIZE / brickElemSize;

	// Compute the resolution of the pools
	uint3 nodePoolRes = make_uint3( (uint)floorf( powf( (float)nodePoolNumElems, 1.0f / 3.0f ) ) ) * NodeRes::get();
	//uint3 brickPoolRes = make_uint3( (uint)floorf( powf( (float)brickPoolNumElems, 1.0f / 3.0f ) ) ) * RealBrickRes::get();

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
	GvUtils::GsDataLoader< DataType >* dataLoader = new GvUtils::GsDataLoader< DataType >( filename.toStdString(), BrickRes::get(), BrickBorderSize, true );

	// Shader creation
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	const bool useGraphicsLibraryInteroperability = true;
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, shader, useGraphicsLibraryInteroperability );

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
	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	//glClearColor( 0.0f, 0.1f, 0.3f, 0.0f );
	//glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

	glEnable( GL_DEPTH_TEST );

	glMatrixMode( GL_MODELVIEW);

	// Display the data structure (space partitioning)
	if ( mDisplayOctree )
	{
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
	}

	// extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	// FIXME
	glPushMatrix();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	// FIXME
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );
	// FIXME
	glPopMatrix();

	// build and extract tree transformations
	float4x4 modelMatrix;

	glPushMatrix();
	glLoadIdentity();
	//glTranslatef(-0.5f, -0.5f, -0.5f);
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[0], params[1], params[2], params[3] );

	CUDAPM_STOP_EVENT( app_init_frame );

	// Render the scene into textures
	_pipeline->execute( modelMatrix, viewMatrix, projectionMatrix, viewport );

	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the performance monitor
	if ( mDisplayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( mDisplayPerfmon - 1 );
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
	mWidth = width;
	mHeight = height;

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( mWidth, mHeight ) );

	/*uchar *timersMask = GvPerfMon::CUDAPerfMon::get().getKernelTimerMask();
	cudaMemset(timersMask, 255, mWidth * mHeight);*/
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
	mDisplayOctree = !mDisplayOctree;
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
	if ( mDisplayPerfmon )
	{
		mDisplayPerfmon = 0;
	}
	else
	{
		mDisplayPerfmon = mode;
	}
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::incMaxVolTreeDepth()
{
	if ( mMaxVolTreeDepth < 32 )
	{
		mMaxVolTreeDepth++;
	}

	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
}

/******************************************************************************
 * ...
 ******************************************************************************/
void SampleCore::decMaxVolTreeDepth()
{
	if ( mMaxVolTreeDepth > 0 )
	{
		mMaxVolTreeDepth--;
	}

	_pipeline->editDataStructure()->setMaxDepth( mMaxVolTreeDepth );
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
	float3 lightPos = make_float3( pX, pY, pZ );
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPos, sizeof( lightPos ), 0, cudaMemcpyHostToDevice ) );
}
