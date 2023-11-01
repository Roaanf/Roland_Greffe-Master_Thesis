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

// Project
#include "HostProducer.h"
#include "ShaderKernel.h"
#include "Particle.h"

// GvViewer
#include <GvvApplication.h>
#include <GvvMainWindow.h>

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
#define NODEPOOL_MEMSIZE	( 16U * 1024U * 1024U )		// 16 Mo
#define BRICKPOOL_MEMSIZE	( 256U * 1024U * 1024U )		// 256 Mo

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
:	GvvPipelineInterface()
,	_pipeline( NULL )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 0 )
,	_depthBuffer( 0 )
,	_width( 512 )
,	_height( 512 )
,	_colorTex( 0 )
,	_colorRenderBuffer( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_transferFunction( NULL )
, 	_frameNumber(0)
, 	_lastsTimesBricks()
, 	_lastsTimesNodesPool()
, 	_nParticles(50000)
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
	setLightPosition(.5f, .5f, .5f);
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Finalize the GigaVoxels pipeline (free memory)
	finalizePipeline();

	// Finalize the transfer function (free memory)
	finalizeTransferFunction();

	// Finalize graphics resources
	finalizeGraphicsResources();

	// Finalize particles
	finalizeParticles();
}

/******************************************************************************
 * Gets the name of this browsable
 *
 * @return the name of this browsable
 ******************************************************************************/
const char* SampleCore::getName() const
{
	return "Collision";
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 ******************************************************************************/
void SampleCore::init()
{
	CUDAPM_INIT();

	// Initialize the GigaVoxels pipeline
	initializePipeline();

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

	// Initialize the transfer function
	initializeTransferFunction();

	// Custom initialization
	// Note : this could be done via an XML settings file loaded at initialization
	// Note : the SET accessors call clearCache() which is useless...
	setNoiseFirstFrequency( 2.f );
	setNoiseMaxFrequency( 101.f );
	setNoiseStrength( 0.4f );
	setNoiseType( ProducerKernel< DataStructureType >::SIMPLEX );
	setBrightness( 10.f );
	setLightingType( PHONG );
	setAmbient( 0.3f );
	setDiffuse( 0.6f );
	setSpecular( 1.2f );
	setTorusRadius( 0.6f );
	setTubeRadius( 0.22f );

	// Initialize particles
	initializeParticles();
}

/******************************************************************************
 * Initialize the particles
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::initializeParticles()
{
	cudaMemcpyToSymbol( Particles::nParticles, &_nParticles, sizeof( unsigned int ), size_t(0), cudaMemcpyHostToDevice );

	// Generate VBO to store particles positions
	// create buffer object
	glGenBuffers( 1, &Particles::positionBuffer );

	// Initialize buffer object
	glBindBuffer( GL_ARRAY_BUFFER, Particles::positionBuffer );
	unsigned int bufferSize = _nParticles * sizeof( float3 );
	glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	// Register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer( &Particles::cuda_vbo_resource,
			Particles::positionBuffer,
			cudaGraphicsMapFlagsWriteDiscard );

	// Allocate an array of particle on the device
	const size_t particleArraySize = _nParticles * sizeof( Particle );
	Particle *particleArray;
	cudaMalloc( (void **)&particleArray, particleArraySize );
	GV_CHECK_CUDA_ERROR( "initializeParticles: out of memory." );
	cudaMemcpyToSymbol( Particles::particles, &particleArray, sizeof(Particle *), size_t(0), cudaMemcpyHostToDevice );
	// Create random particles
	// unsigned long long seed = static_cast <unsigned> (time(NULL));
	unsigned long long seed = 5;
	Particles::createParticles( seed, _nParticles );

	return true;
}

/******************************************************************************
 * Finalize the particles
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::finalizeParticles()
{
	bool ret = true;

	// Free buffer
	glDeleteBuffers( 1, &Particles::positionBuffer );

	// Free memory
	Particle *particleArray;
	cudaMemcpyFromSymbol( &particleArray, Particles::particles, sizeof(Particle *), size_t(0), cudaMemcpyDeviceToHost );
	cudaFree( (void *)particleArray );
	GV_CHECK_CUDA_ERROR( "finalizeParticles" );

	return ret;
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::initializePipeline()
{
	// Pipeline creation
	_pipeline = new PipelineType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, shader );

	// Producer initialization
	_producer = new ProducerType();
	assert( _producer != NULL );
	_pipeline->addProducer( _producer );
	// Settings
	_producer->_sampleCore = this;

	// Renderer initialization
	_renderer = new RendererType( _pipeline->editDataStructure(), _pipeline->editCache() );
	assert( _renderer != NULL );
	_pipeline->addRenderer( _renderer );

	// Pipeline configuration
	_maxVolTreeDepth = 6;
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Fill the data type list used to store voxels in the data structure
	GvViewerCore::GvvDataType& dataType = editDataTypes();
	vector< string > types;
	types.push_back( "uchar4" );
	types.push_back( "half4" );
	vector< string > names;
	names.push_back( "color" );
	names.push_back( "normal" );
	vector< string > info;
	info.push_back( "RGBA color" );
	info.push_back( "Normal" );
	dataType.setTypes( types );
	dataType.setNames( names );
	dataType.setInfo( info );

	return true;
}

/******************************************************************************
 * Finalize the GigaVoxels pipeline
 *
 * @return flag to tell whether or not it succeeded
 ******************************************************************************/
bool SampleCore::finalizePipeline()
{
	/*_pipeline->editRenderer()*/_renderer->disconnect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot );
	/*_pipeline->editRenderer()*/_renderer->disconnect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot );

	// Free memory
	delete _pipeline;
	_pipeline = NULL;

	return true;
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

	// Initialize transfer function with a resolution of 256 elements
	_transferFunction->create( 256 );

	// Bind the transfer function's internal data to the texture reference that will be used on device code
	_transferFunction->bindToTextureReference( &transferFunctionTexture, "transferFunctionTexture", true, cudaFilterModeLinear, cudaAddressModeClamp );

	return true;
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
		clearCache();
	}
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
 * Finalize graphics resources
 *
 * @return flag to tell whether or not it succeeded
******************************************************************************/
bool SampleCore::finalizeGraphicsResources()
{
	if ( _depthBuffer )
	{
		glDeleteBuffers( 1, &_depthBuffer );
	}

	if ( _colorTex )
	{
		glDeleteTextures( 1, &_colorTex );
	}
	if ( _depthTex )
	{
		glDeleteTextures( 1, &_depthTex );
	}

	if ( _frameBuffer )
	{
		glDeleteFramebuffers( 1, &_frameBuffer );
	}

	return true;
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

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );
	glEnable( GL_DEPTH_TEST );

	if( _displayOctree )
	{
		// Display the GigaVoxels N3-tree space partitioning structure
		glPushMatrix();
		// Transformations used to position the GigaVoxels data structure
		glRotatef( _rotation[ 0 ], _rotation[ 1 ], _rotation[ 2 ], _rotation[ 3 ] );
		glScalef( _scale, _scale, _scale );
		glTranslatef( _translation[ 0 ], _translation[ 1 ], _translation[ 2 ] );
		_pipeline->editDataStructure()->render();
		glPopMatrix();
	}

	//
	//glDisable( GL_LIGHTING );
	//glDisable( GL_BLEND );
	//glPolygonMode( GL_FRONT_AND_BACK , GL_FILL);

	// Manage particles
		// Clean cache
	_pipeline->editCache()->preRenderPass();

		// Attach vbo to cuda.
	float3 *devicePointer;
	cudaGraphicsMapResources(1, &Particles::cuda_vbo_resource, 0);
	size_t num_bytes; 
	cudaGraphicsResourceGetMappedPointer((void **)&devicePointer, &num_bytes, Particles::cuda_vbo_resource);

	Particles::animation< DataStructureTypeKernel, PipelineType::CacheType::DataProductionManagerKernelType > ( 
			_pipeline->getDataStructure()->volumeTreeKernel,
			_pipeline->getCache()->getKernelObject(),
			devicePointer,
		   	_nParticles );

		// Detach vbo and Cuda.
	cudaGraphicsUnmapResources( 1, &Particles::cuda_vbo_resource, 0 );

		// Draw particles
	glBindBuffer( GL_ARRAY_BUFFER, Particles::positionBuffer ); 
	glEnableVertexAttribArray( 0 );
	glVertexAttribPointer( 0, 3, GL_FLOAT, GL_FALSE, 0, 0 ); 
	glPointSize( 2.0f );
	glColor3f( 1.f, 1.f, 1.f );
	glDrawArrays( GL_POINTS, 0, _nParticles );

	glDisableVertexAttribArray( 0 ); 
	glBindBuffer( GL_ARRAY_BUFFER, 0 ); 


	glDisable( GL_DEPTH_TEST );

	// Clear the depth PBO (pixel buffer object) by reading from the previously cleared FBO (frame buffer object)
	glBindBuffer( GL_PIXEL_PACK_BUFFER, _depthBuffer );
	glReadPixels( 0, 0, _width, _height, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, 0 );
	glBindBuffer( GL_PIXEL_PACK_BUFFER, 0 );
	GV_CHECK_GL_ERROR();

	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// Extract view transformations
	float4x4 viewMatrix;
	float4x4 projectionMatrix;
	glGetFloatv( GL_MODELVIEW_MATRIX, viewMatrix._array );
	glGetFloatv( GL_PROJECTION_MATRIX, projectionMatrix._array );

	// Extract viewport
	GLint params[4];
	glGetIntegerv( GL_VIEWPORT, params );
	int4 viewport = make_int4( params[0], params[1], params[2], params[3] );

	// Render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	// Transformations used to position the GigaVoxels data structure
	glRotatef( _rotation[0], _rotation[1], _rotation[2], _rotation[3] );
	glScalef( _scale, _scale, _scale );
	glTranslatef( _translation[0], _translation[1], _translation[2] );
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
	//_volumeTreeRenderer->doPostRender();

	// Update GigaVoxels info
	/*_pipeline->editRenderer()*/_renderer->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::get().displayFrameGL( _displayPerfmon - 1 );
	}

	// Modify the frame number
	++_frameNumber;
}

/******************************************************************************
 * Resize the frame
 *
 * @param pWidth the new width
 * @param pHeight the new height
 ******************************************************************************/
void SampleCore::resize( int pWidth, int pHeight )
{
	_width = pWidth;
	_height = pHeight;

	// Reset default active frame region for rendering
	/*_pipeline->editRenderer()*/_renderer->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	// Re-init Perfmon subsystem
	CUDAPM_RESIZE(make_uint2(_width, _height));

	// Disconnect all registered graphics resources
	/*_pipeline->editRenderer()*/_renderer->resetGraphicsResources();

	// Finalize graphics resources
	finalizeGraphicsResources();

	// -- [ Create frame-dependent objects ] --

	glGenTextures( 1, &_colorTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, _width, _height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	GV_CHECK_GL_ERROR();

	glGenBuffers(1, &_depthBuffer);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	glBufferData(GL_PIXEL_PACK_BUFFER, _width * _height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	GV_CHECK_GL_ERROR();

	glGenTextures( 1, &_depthTex );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, _width, _height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	GV_CHECK_GL_ERROR();

	glGenFramebuffers( 1, &_frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	GV_CHECK_GL_ERROR();

	// Create CUDA resources from OpenGL objects
	/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
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

	/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	/*_pipeline->editRenderer()*/_renderer->connect( GsGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
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
	pX = _lightPosition.x + _translation[0];
	pY = _lightPosition.y + _translation[1];
	pZ = _lightPosition.z + _translation[2];
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
	_lightPosition.x = pX - _translation[ 0 ];
	_lightPosition.y = pY - _translation[ 1 ];
	_lightPosition.z = pZ - _translation[ 2 ];

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &_lightPosition, sizeof( _lightPosition ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Toggle the animation.
 ******************************************************************************/
void SampleCore::toggleAnimation( bool run ) 
{
	_runAnimation = run;
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( Particles::cRunAnimation, &_runAnimation, sizeof( _runAnimation ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Set the number of particles.
 ******************************************************************************/
void SampleCore::setParticlesNumber( int particles )
{
	_nParticles = particles;
	resetAnimation();
}

/******************************************************************************
 * Reset the animation.
 ******************************************************************************/
void SampleCore::resetAnimation() 
{
	finalizeParticles();
	initializeParticles();
}

/******************************************************************************
 * Set the gravity
 ******************************************************************************/
void SampleCore::setGravity( float gravity )
{
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( Particles::cGravity, &gravity, sizeof( gravity ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Set the rebound
 ******************************************************************************/
void SampleCore::setRebound( float rebound )
{
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( Particles::cRebound, &rebound, sizeof( rebound ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * @return the frame number.
 ******************************************************************************/
unsigned int SampleCore::getFrameNumber() const {
	return _frameNumber;
}

/******************************************************************************
 * Get the ambient reflection
 *
 * @return ambient reflection
 ******************************************************************************/
float SampleCore::getAmbient() const
{
	return _ambient;
}

/******************************************************************************
 * Set the ambient reflection
 *
 * @param pValue the ambient reflection
 ******************************************************************************/
void SampleCore::setAmbient( float pValue )
{
	_ambient = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cAmbient, &_ambient, sizeof( _ambient ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the diffuse reflection
 *
 * @return diffuse reflection
 ******************************************************************************/
float SampleCore::getDiffuse() const
{
	return _diffuse;
}

/******************************************************************************
 * Set the diffuse reflection
 *
 * @param pValue the diffuse reflection
 ******************************************************************************/
void SampleCore::setDiffuse( float pValue )
{
	_diffuse = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cDiffuse, &_diffuse, sizeof( _diffuse ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the specular reflection
 *
 * @return specular reflection
 ******************************************************************************/
float SampleCore::getSpecular() const
{
	return _specular;
}

/******************************************************************************
 * Set the specular reflection
 *
 * @param pValue the specular reflection
 ******************************************************************************/
void SampleCore::setSpecular( float pValue )
{
	_specular = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cSpecular, &_specular, sizeof( _specular ), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the torus radius
 *
 * @return the torus radius
 ******************************************************************************/
float SampleCore::getTorusRadius() const
{
	return _torusRadius;
}

/******************************************************************************
 * Set the torus radius
 *
 * @param pValue the torus radius
 ******************************************************************************/
void SampleCore::setTorusRadius( float pValue )
{
	_torusRadius = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cTorusRadius, &_torusRadius, sizeof( _torusRadius ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the tube radius
 *
 * @return the tube radius
 ******************************************************************************/
float SampleCore::getTubeRadius() const
{
	return _tubeRadius;
}

/******************************************************************************
 * Set the tube radius
 *
 * @param pValue the tube radius
 ******************************************************************************/
void SampleCore::setTubeRadius( float pValue )
{
	_tubeRadius = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cTubeRadius, &_tubeRadius, sizeof( _tubeRadius ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the noise first frequency
 *
 * @return the noise first frequency
 ******************************************************************************/
unsigned int SampleCore::getNoiseFirstFrequency() const
{
	return _noiseFirstFrequency;
}

/******************************************************************************
 * Set the noise first frequency
 *
 * @param pValue the noise first frequency
 ******************************************************************************/
void SampleCore::setNoiseFirstFrequency( unsigned int pValue )
{
	_noiseFirstFrequency = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseFirstFrequency, &_noiseFirstFrequency, sizeof( _noiseFirstFrequency ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the noise max frequency
 *
 * @return the noise max frequency
 ******************************************************************************/
unsigned int SampleCore::getNoiseMaxFrequency() const
{
	return _noiseMaxFrequency;
}

/******************************************************************************
 * Set the noise max frequency
 *
 * @param pValue the noise max frequency
 ******************************************************************************/
void SampleCore::setNoiseMaxFrequency( unsigned int pValue )
{
	_noiseMaxFrequency = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseMaxFrequency, &_noiseMaxFrequency, sizeof( _noiseMaxFrequency ), 0, cudaMemcpyHostToDevice ) );

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
 * Get the noise type
 *
 * @return the noise type
 ******************************************************************************/
int SampleCore::getNoiseType() const
{
	return _noiseType;
}

/******************************************************************************
 * Set the noise type
 *
 * @param pValue the noise type
 ******************************************************************************/
void SampleCore::setNoiseType( int pValue )
{
	_noiseType = pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cNoiseType, &_noiseType, sizeof( _noiseType ), 0, cudaMemcpyHostToDevice ) );

	// Clear cache
	clearCache();
}

/******************************************************************************
 * Get the lighting type
 *
 * @return the lighting type
 ******************************************************************************/
int SampleCore::getLightingType() const
{
	return _lightingType;
}

/******************************************************************************
 * Set the lighting type
 *
 * @param pValue the lighting type
 ******************************************************************************/
void SampleCore::setLightingType( int pValue )
{
	_lightingType= pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightingType, &_lightingType, sizeof( _lightingType), 0, cudaMemcpyHostToDevice ) );
}

/******************************************************************************
 * Get the brightness
 *
 * @return the brightness
 ******************************************************************************/
float SampleCore::getBrightness() const
{
	return _brightness;
}

/******************************************************************************
 * Set the brightness
 *
 * @param pValue the brightness
 ******************************************************************************/
void SampleCore::setBrightness( float pValue )
{
	_brightness= pValue;

	// Update device memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cBrightness, &_brightness, sizeof( _brightness), 0, cudaMemcpyHostToDevice ) );
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
 * Add the last time it took to compute the bricks to the list of times.
 *
 * @param time the last time observed (in ms).
 ******************************************************************************/
void SampleCore::addTimeBrick( float time )
{
	_lastsTimesBricks.push_back( pair< float, unsigned int >( time, _frameNumber ));
}

/******************************************************************************
 * Add the last time it took to compute the node pool to the list of times.
 *
 * @param time the last time observed (in ms).
 ******************************************************************************/
void SampleCore::addTimeNodePool( float time )
{
	_lastsTimesNodesPool.push_back( pair< float, unsigned int >( time, _frameNumber ));
}

/******************************************************************************
 * Return the last times it took to compute the bricks since the last time this
 * function was called.
 *
 * @param time the last time observed (in ms).
 ******************************************************************************/
std::vector< std::pair< float, unsigned int > > SampleCore::getTimeBrick()
{
	std::vector< std::pair< float, unsigned int > > ret = _lastsTimesBricks;
	_lastsTimesBricks.clear();
	return ret;
}

/******************************************************************************
 * Return the last times it took to compute the nodes pool since the last time this
 * function was called.
 *
 * @param time the last time observed (in ms).
 ******************************************************************************/
std::vector< std::pair< float, unsigned int > > SampleCore::getTimeNodePool()
{
	std::vector< std::pair< float, unsigned int > > ret = _lastsTimesNodesPool;
	_lastsTimesNodesPool.clear();
	return ret;
}
