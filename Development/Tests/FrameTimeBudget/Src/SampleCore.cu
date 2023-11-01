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
#include <GvCore/StaticRes3D.h>
#include <GvStructure/GvVolumeTree.h>
#include <GvStructure/GvDataProductionManager.h>
#include <GvRendering/GvRendererCUDA.h>
#include <GvUtils/GvSimplePipeline.h>
#include <GvUtils/GvSimpleHostProducer.h>
#include <GvUtils/GvSimpleHostShader.h>
#include <GvUtils/GvCommonGraphicsPass.h>
#include <GvUtils/GvTransferFunction.h>
#include <GvCore/GvError.h>
#include <GvPerfMon/GvPerformanceMonitor.h>

// Project
#include "ProducerKernel.h"
#include "ShaderKernel.h"
#include "vdCube3D4.h"
#include "TimeBudgetView.h"

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

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// Defines the size allowed for each type of pool
#define NODEPOOL_MEMSIZE	( 8U * 1024U * 1024U )		//   8 Mo
#define BRICKPOOL_MEMSIZE	( 384U * 1024U * 1024U )	// 384 Mo

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
,	_graphicsEnvironment( NULL )
,	_depthBuffer( 0 )
,	_colorTex( 0 )
,	_depthTex( 0 )
,	_frameBuffer( 0 )
,	_width( 512 )
,	_height( 512 )
,	_displayOctree( false )
,	_displayPerfmon( 0 )
,	_maxVolTreeDepth( 6 )
,	_signedDistanceField( NULL )
,	_transferFunction( NULL )
,	_timeBudgetView( NULL )
,	_hasTimeBudget( false )
,	_timeBudget( 60 )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
SampleCore::~SampleCore()
{
	// Finalize the GigaVoxels pipeline (free memory)
	finalizePipeline();

	// Finalize the 3D model (free memory)
	finalize3DModel();

	// Finalize the transfer function (free memory)
	finalizeTransferFunction();

	// Finalize graphics resources
	finalizeGraphicsResources();

	delete _timeBudgetView;
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

	// Initialize the GigaVoxels pipeline
	initializePipeline();

	// Initialize the 3D model
	initialize3DModel();

	// Initialize the transfer function
	initializeTransferFunction();

	// Plot view
	setTimeBudgetActivated( false );
	setTimeBudget( 60 );
	_timeBudgetView = new TimeBudgetView();
	_timeBudgetView->populate( this );
	_timeBudgetView->show();
}

/******************************************************************************
 * Draw function called of frame
 ******************************************************************************/
void SampleCore::draw()
{
	_pipeline->editRenderer()->startTimer();

	CUDAPM_START_FRAME;
	CUDAPM_START_EVENT( frame );
	CUDAPM_START_EVENT( app_init_frame );

	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );

	glMatrixMode( GL_MODELVIEW );

	// Handle image downscaling if activated
	int bufferWidth = _graphicsEnvironment->getBufferWidth();
	int bufferHeight = _graphicsEnvironment->getBufferHeight();
	if ( _graphicsEnvironment->hasImageDownscaling() )
	{
		bufferWidth = _graphicsEnvironment->getImageDownscalingWidth();
		bufferHeight = _graphicsEnvironment->getImageDownscalingHeight();
		glViewport( 0, 0, bufferWidth, bufferHeight );
	}

	if ( _displayOctree )
	{
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT );

		// Display the GigaVoxels N3-tree space partitioning structure
		glEnable( GL_DEPTH_TEST );
		glPushMatrix();
		glTranslatef( -0.5f, -0.5f, -0.5f );
		_pipeline->editDataStructure()->displayDebugOctree();
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
	// Handle image downscaling if activated
	if ( _graphicsEnvironment->hasImageDownscaling() )
	{
		// TO DO : clean this... it would better to send real viewport info and retrieve realBufferSize in the renderer ?
		viewport.z = bufferWidth;
		viewport.w = bufferHeight;
	}

	// render the scene into textures
	CUDAPM_STOP_EVENT( app_init_frame );

	// Build the world transformation matrix
	float4x4 modelMatrix;
	glPushMatrix();
	glLoadIdentity();
	glTranslatef( -0.5f, -0.5f, -0.5f );
	glGetFloatv( GL_MODELVIEW_MATRIX, modelMatrix._array );
	glPopMatrix();

	// Render
	_pipeline->editRenderer()->render( modelMatrix, viewMatrix, projectionMatrix, viewport );

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

	// Handle image downscaling if activated
	if ( _graphicsEnvironment->hasImageDownscaling() )
	{
		glViewport( 0, 0, _graphicsEnvironment->getBufferWidth(), _graphicsEnvironment->getBufferHeight() );
	}

	// Draw a full screen quad
	GLint sMin = 0;
	GLint tMin = 0;
	//GLint sMax = _width;
	//GLint tMax = _height;
	GLint sMax = bufferWidth;
	GLint tMax = bufferHeight;
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
	_pipeline->editRenderer()->nextFrame();

	CUDAPM_STOP_EVENT( frame );
	CUDAPM_STOP_FRAME;

	// Display the GigaVoxels performance monitor (if it has been activated during GigaVoxels compilation)
	if ( _displayPerfmon )
	{
		GvPerfMon::CUDAPerfMon::getApplicationPerfMon().displayFrameGL( _displayPerfmon - 1 );
	}

	// Stop the timer
	_pipeline->editRenderer()->stopTimer();

	// Get elapsed time
	//float frameDuration = _pipeline->editRenderer()->getElapsedTime() - lastTime;
	//frameDuration -= lastTime;
	//lastTime += frameDuration;
	const float frameDuration = _pipeline->editRenderer()->getElapsedTime();
	const float timeBudget = 1.f / static_cast< float >( getTimeBudget() ) * 1000.f; // (1/60Hz) x 1000 for time in milliseconds
	//std::cout << "Time : " << frameDuration << " ms - budget : " << timeBudget << std::endl;
	//if ( frameDuration > timeBudget )
	//{
	//	//std::cout << "\tWARNING : use Image Downscaling" << std::endl;

	//	// Update graphics environment
	//	int width = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
	//	int height = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );
	//	const float relaxationFactor = 0.02f;	// 2%
	//	int newWidth = width - relaxationFactor * width;
	//	int newHeight = height - relaxationFactor * height;

	//	_graphicsEnvironment->setImageDownscaling( true );
	//	_graphicsEnvironment->setImageDownscalingSize( newWidth > 0 ? newWidth : 1, newHeight > 0 ? newHeight : 1 );
	//	resetGraphicsresources();

	//	// Update graphics environment
	//	int bufferWidth = _graphicsEnvironment->getBufferWidth();
	//	int bufferHeight = _graphicsEnvironment->getBufferHeight();
	//	//std::cout << "\tWINDOW : " << bufferWidth << " - " << bufferHeight << std::endl;
	//	//std::cout << "\tBUFFER : " << newWidth << " - " << newHeight;// << std::endl;
	//	//std::cout << std::endl;

	//	std::cout << "DOWN" << std::endl;
	//}
	//else
	//{
	//	// Handle image downscaling if activated
	//	if ( _graphicsEnvironment->hasImageDownscaling() )
	//	{
	//		// Here, try to reach first user request ?
	//		// ...

	//		int bufferWidth = _graphicsEnvironment->getBufferWidth();
	//		int bufferHeight = _graphicsEnvironment->getBufferHeight();

	//		// Update graphics environment
	//		int width = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
	//		int height = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );
	//		const float relaxationFactor = 0.02f;	// 2%
	//		int newWidth = width + relaxationFactor * width;
	//		int newHeight = height + relaxationFactor * height;

	//		if ( ( newWidth > bufferWidth ) || ( newHeight > bufferHeight ) )
	//		{
	//			_graphicsEnvironment->setImageDownscaling( false );
	//			_graphicsEnvironment->setImageDownscalingSize( 512, 512 );	// reset values ?
	//			resetGraphicsresources();
	//		}
	//		else
	//		{
	//			_graphicsEnvironment->setImageDownscaling( true );
	//			_graphicsEnvironment->setImageDownscalingSize( newWidth < bufferWidth ? newWidth : bufferWidth, newHeight < bufferHeight ? newHeight : bufferHeight );
	//			resetGraphicsresources();
	//		}

	//	//	std::cout << "\tWINDOW : " << bufferWidth << " - " << bufferHeight << std::endl;
	//	//	std::cout << "\tBUFFER : " << newWidth << " - " << newHeight;// << std::endl;
	//	//	std::cout << std::endl;

	//		std::cout << "UP" << std::endl;
	//	}
	//	else
	//	{
	//		// Nothing to do
	//		//std::cout << "\tClassic Rendering" << std::endl;
	//	}
	//}

	static unsigned int frame = 0;
	if ( _timeBudgetView != NULL )
	{
		_timeBudgetView->onCurveChanged( frame, frameDuration );
		frame++;
	}

	const float relaxationFactor = 0.02f;	// 2%

	if ( ! hasTimeBudget() )
	{
		if ( _graphicsEnvironment->hasImageDownscaling() )
		{
			_graphicsEnvironment->setImageDownscaling( false );

			int bufferWidth = _graphicsEnvironment->getBufferWidth();
			int bufferHeight = _graphicsEnvironment->getBufferHeight();
			//_graphicsEnvironment->setImageDownscalingSize( 512, 512 );	// reset values ?
			_graphicsEnvironment->setImageDownscalingSize( bufferWidth, bufferHeight );	// reset values ?

			resetGraphicsresources();
		}
		else
		{
			// Nothing to do
		}
	}
	else
	{
		if ( frameDuration > timeBudget )
		{
			// [ Decrease offscreen buffer size ] - relaxation scheme

			//std::cout << "\tWARNING : use Image Downscaling" << std::endl;

			// Update graphics environment
			int width = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
			int height = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );
			
			int newWidth = width - relaxationFactor * width;
			int newHeight = height - relaxationFactor * height;

			_graphicsEnvironment->setImageDownscaling( true );
			_graphicsEnvironment->setImageDownscalingSize( newWidth > 0 ? newWidth : 1, newHeight > 0 ? newHeight : 1 );
			resetGraphicsresources();

			// Update graphics environment
			int bufferWidth = _graphicsEnvironment->getBufferWidth();
			int bufferHeight = _graphicsEnvironment->getBufferHeight();
			std::cout << "\tWINDOW : " << bufferWidth << " - " << bufferHeight << std::endl;
			std::cout << "\tBUFFER : " << newWidth << " - " << newHeight;// << std::endl;
			std::cout << std::endl;

			std::cout << "DOWN" << std::endl;
		}
		else
		{
			// [ Increase offscreen buffer size ] - relaxation scheme

			//std::cout << "OK" << std::endl;

			// Handle image downscaling if activated
			if ( _graphicsEnvironment->hasImageDownscaling() )
			{
				// Here, try to reach first user request ?
				// ...

				int bufferWidth = _graphicsEnvironment->getBufferWidth();
				int bufferHeight = _graphicsEnvironment->getBufferHeight();

				// Update graphics environment
				int width = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingWidth() );
				int height = static_cast< unsigned int >( _graphicsEnvironment->getImageDownscalingHeight() );

				//  Epsillon : 10% => on essaie de ne remonte que jusqu'� 10 % de la limite (user request) afin d'�viter un "yoyo"
				const float epsillon = 0.01f;
				const float ratio = frameDuration / timeBudget;
				if ( ratio < 0.75f )	// not good => at 60 Hz OK but at 120 Hz, the ratio dosn't not work in special case => we have "yoyo" effect...
				{
					//const float relaxationFactor = 0.02f;	// 2%
					int newWidth = width + relaxationFactor * width;
					int newHeight = height + relaxationFactor * height;

					if ( ( newWidth > bufferWidth ) || ( newHeight > bufferHeight ) )
					{
						_graphicsEnvironment->setImageDownscaling( false );
						//_graphicsEnvironment->setImageDownscalingSize( 512, 512 );	// reset values ?
						_graphicsEnvironment->setImageDownscalingSize( bufferWidth, bufferHeight );	// reset values ?
						resetGraphicsresources();
					}
					else
					{
						_graphicsEnvironment->setImageDownscaling( true );
						_graphicsEnvironment->setImageDownscalingSize( newWidth < bufferWidth ? newWidth : bufferWidth, newHeight < bufferHeight ? newHeight : bufferHeight );
						resetGraphicsresources();
					}

					//	std::cout << "\tWINDOW : " << bufferWidth << " - " << bufferHeight << std::endl;
					//	std::cout << "\tBUFFER : " << newWidth << " - " << newHeight;// << std::endl;
					//	std::cout << std::endl;

					std::cout << "UP" << std::endl;
				}
			}
			else
			{
				// Nothing to do
				//std::cout << "\tClassic Rendering" << std::endl;
			}
		}
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
	//_width = width;
	//_height = height;

	//// Reset default active frame region for rendering
	//_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, _width, _height ) );

	//// Re-init Perfmon subsystem
	//CUDAPM_RESIZE( make_uint2( _width, _height ) );

	//// Disconnect all registered graphics resources
	//_pipeline->editRenderer()->resetGraphicsResources();

	//// Finalize graphics resources
	//finalizeGraphicsResources();

	//// -- [ Create frame-dependent objects ] --

	//glGenTextures( 1, &_colorTex );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _colorTex );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	//glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	//GV_CHECK_GL_ERROR();

	//glGenBuffers(1, &_depthBuffer);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, _depthBuffer);
	//glBufferData(GL_PIXEL_PACK_BUFFER, width * height * sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	//glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	//GV_CHECK_GL_ERROR();

	//glGenTextures( 1, &_depthTex );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, _depthTex );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	//glTexParameteri( GL_TEXTURE_RECTANGLE_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	//glTexImage2D( GL_TEXTURE_RECTANGLE_EXT, 0, GL_DEPTH24_STENCIL8_EXT, width, height, 0, GL_DEPTH_STENCIL_EXT, GL_UNSIGNED_INT_24_8_EXT, NULL );
	//glBindTexture( GL_TEXTURE_RECTANGLE_EXT, 0 );
	//GV_CHECK_GL_ERROR();

	//glGenFramebuffers( 1, &_frameBuffer );
	//glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	//glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE_EXT, _colorTex, 0 );
	//glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	//glFramebufferTexture2D( GL_FRAMEBUFFER, GL_STENCIL_ATTACHMENT, GL_TEXTURE_RECTANGLE_EXT, _depthTex, 0 );
	//glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	//GV_CHECK_GL_ERROR();

	//// Create CUDA resources from OpenGL objects
	//if ( _displayOctree )
	//{
	//	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	//	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	//}
	//else
	//{
	//	_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	//}

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

	// --------------------------
	// Reset default active frame region for rendering
	_pipeline->editRenderer()->setProjectedBBox( make_uint4( 0, 0, pWidth, pHeight ) );
	// Re-init Perfmon subsystem
	CUDAPM_RESIZE( make_uint2( pWidth, pHeight ) );
	// --------------------------

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
	_pipeline->editRenderer()->resetGraphicsResources();
	
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
		if ( _graphicsEnvironment->getType() == 0 )
		{
			_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			//_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		if ( _graphicsEnvironment->getType() == 0 )
		{
			_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		}
		else
		{
			//_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorRenderBuffer, GL_RENDERBUFFER );
		}
	}
}

/******************************************************************************
 * Clear the GigaVoxels cache
 ******************************************************************************/
void SampleCore::clearCache()
{
	_pipeline->editRenderer()->clearCache();
}

/******************************************************************************
 * Toggle the display of the N-tree (octree) of the data structure
 ******************************************************************************/
void SampleCore::toggleDisplayOctree()
{
	_displayOctree = !_displayOctree;

	// Disconnect all registered graphics resources
	_pipeline->editRenderer()->resetGraphicsResources();

	if ( _displayOctree )
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorReadWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eDepthReadSlot, _depthBuffer );
	}
	else
	{
		_pipeline->editRenderer()->connect( GvGraphicsInteroperabiltyHandler::eColorWriteSlot, _colorTex, GL_TEXTURE_RECTANGLE_EXT );
	}
}

/******************************************************************************
 * Toggle the GigaVoxels dynamic update mode
 ******************************************************************************/
void SampleCore::toggleDynamicUpdate()
{
	_pipeline->editRenderer()->dynamicUpdateState() = !_pipeline->editRenderer()->dynamicUpdateState();
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
	_pipeline->editRenderer()->setClearColor( make_uchar4( pRed, pGreen, pBlue, pAlpha ) );
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
	GV_CUDA_SAFE_CALL( cudaMemcpyToSymbol( cLightPosition, &lightPos, sizeof( lightPos ), 0, cudaMemcpyHostToDevice ) );
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
		_pipeline->editRenderer()->clearCache();
	}
}

/******************************************************************************
 * Initialize the GigaVoxels pipeline
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::initializePipeline()
{
	// Pipeline creation
	_pipeline = new PipelineType();
	ProducerType* producer = new ProducerType();
	ShaderType* shader = new ShaderType();

	// Pipeline initialization
	_pipeline->initialize( NODEPOOL_MEMSIZE, BRICKPOOL_MEMSIZE, producer, shader );

	// Pipeline configuration
	_pipeline->editDataStructure()->setMaxDepth( _maxVolTreeDepth );

	// Graphics environment creation
	_graphicsEnvironment = new GvCommonGraphicsPass();

	return true;
}

/******************************************************************************
 * Finalize the GigaVoxels pipeline
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::finalizePipeline()
{
	// Free memory
	delete _pipeline;
	_pipeline = NULL;

	delete _graphicsEnvironment;
	_graphicsEnvironment = NULL;
	
	return true;
}

/******************************************************************************
 * Initialize the 3D model
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::initialize3DModel()
{
	// Upload the texture.
	// It contains a 4 components 3D float data :
	// - normal [ 3 components ]
	// - distance (signed distance field) [ 1 component ]
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	QString filename = dataRepository + QDir::separator() + QString( "Voxels" ) + QDir::separator() + QString( "vd4" ) + QDir::separator() + QString( "bunny.vdCube3D4" );
	QFileInfo fileInfo( filename );
	if ( ( ! fileInfo.isFile() ) || ( ! fileInfo.isReadable() ) )
	{
		// Idea
		// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

		// TO DO
		// Handle error : free memory and exit
		// ...
		std::cout << "ERROR. Check filename : " << filename.toLatin1().constData() << std::endl;
	}

	// Create 3D model
	_signedDistanceField = new VolumeData::vdCube3D4( filename.toStdString() );
	assert( _signedDistanceField != NULL );
	if ( _signedDistanceField == NULL )
	{
		// TO DO
		// Handle error
		// ...

		return false;
	}

	// Initialize 3D model
	_signedDistanceField->initialize();
	
	// Bind the 3D model's internal data to the texture reference that will be used on device code,
	// and set texture parameters :
	// ---- access with normalized texture coordinates
	// ---- linear interpolation
	// ---- wrap texture coordinates
	_signedDistanceField->bindToTextureReference( &volumeTex, "volumeTex", true, cudaFilterModeLinear, cudaAddressModeWrap );

    return true;
}

/******************************************************************************
 * Finalize the 3D model
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::finalize3DModel()
{
	delete _signedDistanceField;

	return true;
}

/******************************************************************************
 * Initialize the transfer function
 *
 * @return flag to tell wheter or not it succeeded
 ******************************************************************************/
bool SampleCore::initializeTransferFunction()
{
	// Create the transfer function
	_transferFunction = new GvUtils::GvTransferFunction();
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
	
	return true;
}

/******************************************************************************
 * Finalize the transfer function
 *
 * @return flag to tell wheter or not it succeeded
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
 * @return flag to tell wheter or not it succeeded
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
 *Tell wheter or not time budget is acivated
 *
 * @return a flag to tell wheter or not time budget is activated
 ******************************************************************************/
bool SampleCore::hasTimeBudget() const
{
	return _hasTimeBudget;
}

/******************************************************************************
 * Set the flag telling wheter or not time budget is acivated
 *
 * @param pFlag a flag to tell wheter or not time budget is activated
 ******************************************************************************/
void SampleCore::setTimeBudgetActivated( bool pFlag )
{
	_hasTimeBudget = pFlag;
	//_timeBudgetView->populate( this );
}

/******************************************************************************
 * Get the user requested time budget
 *
 * @return the user requested time budget
 ******************************************************************************/
unsigned int SampleCore::getTimeBudget() const
{
	return _timeBudget;
}

/******************************************************************************
 * Set the user requested time budget
 *
 * @param pValue the user requested time budget
 ******************************************************************************/
void SampleCore::setTimeBudget( unsigned int pValue )
{
	_timeBudget = pValue;
	//_timeBudgetView->populate( this );
}
