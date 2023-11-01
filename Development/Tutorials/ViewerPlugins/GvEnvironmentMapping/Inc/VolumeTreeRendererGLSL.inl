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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsError.h>
//#include <GvUtils/GsShaderManager.h>
//#include <GsGraphics/GsShaderProgram.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>

// STL
#include <iostream>

//--------------------------------------------------------------
// CUDA - NSight
//#define GV_NSIGHT_PROLIFING

#ifdef GV_NSIGHT_PROLIFING
	#include "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\nvToolsExt\\include\\nvToolsExtCuda.h"
#endif
//--------------------------------------------------------------

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 *
 * It initializes all OpenGL-related stuff
 *
 * @param pVolumeTree data structure to render
 * @param pVolumeTreeCache cache
 * @param pProducer producer of data
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::VolumeTreeRendererGLSL( TVolumeTreeType* pVolumeTree, TVolumeTreeCacheType* pVolumeTreeCache )
:	GvRendering::GsRenderer< TVolumeTreeType, TVolumeTreeCacheType >( pVolumeTree, pVolumeTreeCache )
,	_coneApertureScale( 0.f )
,	_maxNbLoops( 0 )
,	_reflectionCoeff( 0.0f )
,	_refractionCoeff( 0.0f )
{
	// Retrieve useful GigaVoxels arrays
	GvCore::GsLinearMemory< uint >* volTreeChildArray = this->_volumeTree->_childArray;
	GvCore::GsLinearMemory< uint >* volTreeDataArray = this->_volumeTree->_dataArray;
	GvCore::GsLinearMemory< uint >* updateBufferArray = this->_volumeTreeCache->getUpdateBuffer();
	GvCore::GsLinearMemory< uint >* nodeTimeStampArray = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	GvCore::GsLinearMemory< uint >* brickTimeStampArray = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	volTreeChildArray->unmapResource();
	volTreeDataArray->unmapResource();
	updateBufferArray->unmapResource();
	nodeTimeStampArray->unmapResource();
	brickTimeStampArray->unmapResource();

	// Create a buffer texture associated to the GigaVoxels update buffer array
	glGenTextures( 1, &_updateBufferTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _updateBufferTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, updateBufferArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Create a buffer texture associated to the GigaVoxels node time stamp array
	glGenTextures( 1, &_nodeTimeStampTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _nodeTimeStampTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, nodeTimeStampArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Create a buffer texture associated to the GigaVoxels brick time stamp array
	glGenTextures( 1, &_brickTimeStampTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _brickTimeStampTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, brickTimeStampArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Create a buffer texture associated to the GigaVoxels data structure's child array
	glGenTextures( 1, &_childArrayTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _childArrayTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, volTreeChildArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Create a buffer texture associated to the GigaVoxels data structure's data array
	glGenTextures( 1, &_dataArrayTBO );
	glBindTexture( GL_TEXTURE_BUFFER, _dataArrayTBO );
	// Attach the storage of buffer object to buffer texture
	glTexBuffer( GL_TEXTURE_BUFFER, GL_R32UI, volTreeDataArray->getBufferName() );
	glBindTexture( GL_TEXTURE_BUFFER, 0 );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	volTreeChildArray->mapResource();
	volTreeDataArray->mapResource();
	updateBufferArray->mapResource();
	nodeTimeStampArray->mapResource();
	brickTimeStampArray->mapResource();

	// Create and link a GLSL shader program
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	//QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GsRendererGLSL" ) + QDir::separator() + QString( "rayCastVert.glsl" );
	QString vertexShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvEnvironmentMapping" ) + QDir::separator() + QString( "gigaspace_vert.glsl" );
	//QString vertexShaderFilename = QString( "rayCastVert.glsl" );
	//QFileInfo vertexShaderFileInfo( vertexShaderFilename );
	//if ( ( ! vertexShaderFileInfo.isFile() ) || ( ! vertexShaderFileInfo.isReadable() ) )
	//{
	//	// Idea
	//	// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

	//	// TO DO
	//	// Handle error : free memory and exit
	//	// ...
	//	std::cout << "ERROR. Check filename : " << vertexShaderFilename.toLatin1().constData() << std::endl;
	//}
	QString fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvEnvironmentMapping" ) + QDir::separator() + QString( "gigaspace_frag.glsl" );
	//QString fragmentShaderFilename = QString( "rayCastFrag.glsl" );
	//QFileInfo fragmentShaderFileInfo( fragmentShaderFilename );
	//if ( ( ! fragmentShaderFileInfo.isFile() ) || ( ! fragmentShaderFileInfo.isReadable() ) )
	//{
	//	// Idea
	//	// Maybe use Qt function : bool QFileInfo::permission ( QFile::Permissions permissions ) const

	//	// TO DO
	//	// Handle error : free memory and exit
	//	// ...
	//	std::cout << "ERROR. Check filename : " << fragmentShaderFilename.toLatin1().constData() << std::endl;
	//}
	//_rayCastProg = GvUtils::GsShaderManager::createShaderProgram( vertexShaderFilename.toLatin1().constData(), NULL, fragmentShaderFilename.toLatin1().constData() );
	//GvUtils::GsShaderManager::linkShaderProgram( _rayCastProg );

	// Initialize shader program
	_shaderProgram = new GsGraphics::GsShaderProgram();
	_shaderProgram->addShader( GsGraphics::GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	_shaderProgram->addShader( GsGraphics::GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	_shaderProgram->link();

	/////////////////////////////////////////////////////////////////////
	//_graphicsResources = new cudaGraphicsResource[ 6 ];
	/*_graphicsResources[ 0 ] = updateBufferArray->_bufferResource;
	_graphicsResources[ 1 ] = nodeTimeStampArray->_bufferResource;
	_graphicsResources[ 2 ] = brickTimeStampArray->_bufferResource;
	_graphicsResources[ 3 ] = volTreeChildArray->_bufferResource;
	_graphicsResources[ 4 ] = volTreeDataArray->_bufferResource;
	_graphicsResources[ 5 ] = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->_bufferResource;*/
	/////////////////////////////////////////////////////////////////////

	// Settings
	_coneApertureScale = 1.333f;
	_maxNbLoops = 200;
	_reflectionCoeff = 0.5f;
	_refractionCoeff = 0.15f;
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::~VolumeTreeRendererGLSL()
{
	// Destroy TBO
	if ( _updateBufferTBO )
	{
		glDeleteTextures( 1, &_updateBufferTBO );
	}

	// Destroy TBO
	if (_nodeTimeStampTBO)
	{
		glDeleteTextures( 1, &_nodeTimeStampTBO );
	}

	// Destroy TBO
	if ( _brickTimeStampTBO )
	{
		glDeleteTextures( 1, &_brickTimeStampTBO );
	}

	// Destroy TBO
	if ( _childArrayTBO )
	{
		glDeleteTextures( 1, &_childArrayTBO );
	}

	// Destroy TBO
	if ( _dataArrayTBO )
	{
		glDeleteTextures( 1, &_dataArrayTBO );
	}
}

/******************************************************************************
 * pre-render stage
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
inline void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::preRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
}

/******************************************************************************
* post-render stage
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
inline void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::postRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
}

/******************************************************************************
 * This function is the specific implementation method called
 * by the parent GsIRenderer::render() method during rendering.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::render( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	// Call internal render method
	doRender( pModelMatrix, pViewMatrix, pProjectionMatrix, pViewport );
}

/******************************************************************************
 * Start the rendering process.
 *
 * @param pModelMatrix the current model matrix
 * @param pViewMatrix the current view matrix
 * @param pProjectionMatrix the current projection matrix
 * @param pViewport the viewport configuration
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::doRender( const float4x4& pModelMatrix, const float4x4& pViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	CUDAPM_START_EVENT( vsrender_pre_frame );
	CUDAPM_START_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame_mapbuffers );
	CUDAPM_STOP_EVENT( vsrender_pre_frame );

	// Extract zNear, zFar as well as the distance in view space
	// from the center of the screen to each side of the screen.
	float fleft   = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 8 ] - 1.0f ) / ( pProjectionMatrix._array[ 0 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fright  = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 8 ] + 1.0f ) / ( pProjectionMatrix._array[ 0 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float ftop    = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 9 ] + 1.0f ) / ( pProjectionMatrix._array[ 5 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fbottom = pProjectionMatrix._array[ 14 ] * ( pProjectionMatrix._array[ 9 ] - 1.0f ) / ( pProjectionMatrix._array[ 5 ] * ( pProjectionMatrix._array[ 10 ] - 1.0f ) );
	float fnear   = pProjectionMatrix._array[ 14 ] / ( pProjectionMatrix._array[ 10 ] - 1.0f );
	float ffar    = pProjectionMatrix._array[ 14 ] / ( pProjectionMatrix._array[ 10 ] + 1.0f );

	float2 viewSurfaceVS[ 2 ];
	viewSurfaceVS[ 0 ] = make_float2( fleft, fbottom );
	viewSurfaceVS[ 1 ] = make_float2( fright, ftop );
	float2 viewSurfaceVS_Size = viewSurfaceVS[ 1 ] - viewSurfaceVS[ 0 ];
	
	// transfor matrices
	float4x4 invprojectionMatrixT = transpose( inverse( pProjectionMatrix ) );
	float4x4 invViewMatrixT = transpose( inverse( pViewMatrix ) );

	float4x4 projectionMatrixT=transpose(pProjectionMatrix);
	float4x4 viewMatrixT=transpose(pViewMatrix);

	CUDAPM_START_EVENT(vsrender_copyconsts_frame);

	viewContext.invViewMatrix = invViewMatrixT;
	viewContext.viewMatrix = viewMatrixT;
	//viewContext.invProjMatrix = invprojectionMatrixT;
	//viewContext.projMatrix = projectionMatrixT;

	// Store frustum parameters
	viewContext.frustumNear = fnear;
	viewContext.frustumNearINV = 1.0f / fnear;
	viewContext.frustumFar = ffar;
	viewContext.frustumRight = fright;
	viewContext.frustumTop = ftop;
	viewContext.frustumC = pProjectionMatrix._array[ 10 ]; // - ( ffar + fnear ) / ( ffar - fnear );
	viewContext.frustumD = pProjectionMatrix._array[ 14 ]; // ( -2.0f * ffar * fnear ) / ( ffar - fnear );

	float3 viewPlanePosWP = mul( viewContext.invViewMatrix, make_float3( fleft, fbottom, -fnear ) );
	viewContext.viewCenterWP = mul( viewContext.invViewMatrix, make_float3( 0.0f, 0.0f, 0.0f ) );
	viewContext.viewPlaneDirWP = viewPlanePosWP - viewContext.viewCenterWP;

	// Resolution dependant stuff
	viewContext.frameSize = make_uint2( pViewport.z, pViewport.w );
	float2 pixelSize = viewSurfaceVS_Size / make_float2( (float)viewContext.frameSize.x, (float)viewContext.frameSize.y );
	//viewContext.pixelSize=pixelSize;
	viewContext.viewPlaneXAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fright, fbottom, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.x );
	viewContext.viewPlaneYAxisWP = ( mul( viewContext.invViewMatrix, make_float3( fleft, ftop, -fnear ) ) - viewPlanePosWP ) / static_cast< float >( viewContext.frameSize.y );

	// Copy data to CUDA memory
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_renderViewContext, &viewContext, sizeof( viewContext ) ) );
	GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( k_currentTime, &(this->_currentTime), sizeof( this->_currentTime ), 0, cudaMemcpyHostToDevice ) );
	
	// TEST -------------------------------------
	//GS_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------

	CUDAPM_STOP_EVENT( vsrender_copyconsts_frame );

	///////////// TEST
	_shaderProgram->use();
	///////////// TEST

	// Viewing System uniform parameters
	glProgramUniform3fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uViewPos" ), viewContext.viewCenterWP.x, viewContext.viewCenterWP.y, viewContext.viewCenterWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uViewPlane" ), viewContext.viewPlaneDirWP.x, viewContext.viewPlaneDirWP.y, viewContext.viewPlaneDirWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uViewAxisX" ), viewContext.viewPlaneXAxisWP.x, viewContext.viewPlaneXAxisWP.y, viewContext.viewPlaneXAxisWP.z );
	glProgramUniform3fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uViewAxisY" ), viewContext.viewPlaneYAxisWP.x, viewContext.viewPlaneYAxisWP.y, viewContext.viewPlaneYAxisWP.z );
	glProgramUniform2fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uPixelSize" ), pixelSize.x, pixelSize.y );
	glProgramUniform1fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uFrustumNearInv" ), viewContext.frustumNearINV );
	// Cone aperture management
	glProgramUniform1fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uConeApertureScale" ), _coneApertureScale );
	// GigaSpace pipeline uniform parameters
	glProgramUniform1uiEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uMaxNbLoops" ), _maxNbLoops );

	//glProgramUniform2fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "frameSize" ), static_cast< float >( viewContext.frameSize.x ), static_cast< float >( viewContext.frameSize.y ) );
	//glProgramUniformMatrix4fvEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "modelViewMat" ), 1, GL_FALSE, pViewMatrix._array );

	//float voxelSizeMultiplier = 1.0f;
	//GS_CUDA_SAFE_CALL( cudaMemcpyToSymbol( GvRendering::k_voxelSizeMultiplier, (&voxelSizeMultiplier), sizeof( voxelSizeMultiplier ), 0, cudaMemcpyHostToDevice ) );

	//uint numLoop = 0;

	//--------------------------------------------------------------------
	// GigaSpace pipeline uniform parameters
	GLint location;
	location = glGetUniformLocation( _shaderProgram->_program, "uReflectionCoeff" );
	if ( location < 0 )
	{
	//	std::cout << "\nError : location uReflectionCoeff" << std::endl;
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uRefractionCoeff" );
	if ( location < 0 )
	{
	//	std::cout << "\nError : location uRefractionCoeff" << std::endl;
	}
	glProgramUniform1fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uReflectionCoeff" ), _reflectionCoeff );
	glProgramUniform1fEXT( _shaderProgram->_program, glGetUniformLocation( _shaderProgram->_program, "uRefractionCoeff" ), _refractionCoeff );
	glActiveTexture( GL_TEXTURE1 );
	glBindTexture( GL_TEXTURE_CUBE_MAP, _cubeMap );
	location = glGetUniformLocation( _shaderProgram->_program, "uSkyBoxTex" );
	if ( location >= 0 )
	{
		glUniform1i( location, 1 );
	}
	else
	{
		std::cout << "\nError : location uSkyBoxTex" << std::endl;
	}
	//--------------------------------------------------------------------

	CUDAPM_START_EVENT_GPU( gv_rendering );

	/*if ( this->_dynamicUpdate )
	{
	}
	else
	{
	}*/

	// Disable writing into the depth buffer
	//glEnable(GL_DEPTH_TEST);
	glDepthMask( GL_FALSE );

	// Activate blending
	glEnable( GL_BLEND );
	glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

	// Installs program object as part of current rendering state
	//_shaderProgram->use();
	//glUseProgram( _shaderProgram->_program );

	// Retrieve useful GigaVoxels arrays
	GvCore::GsLinearMemory< uint >* volTreeChildArray = this->_volumeTree->_childArray;
	GvCore::GsLinearMemory< uint >* volTreeDataArray = this->_volumeTree->_dataArray;
	GvCore::GsLinearMemory< uint >* updateBufferArray = this->_volumeTreeCache->getUpdateBuffer();
	GvCore::GsLinearMemory< uint >* nodeTimeStampArray = this->_volumeTreeCache->getNodesCacheManager()->getTimeStampList();
	GvCore::GsLinearMemory< uint >* brickTimeStampArray = this->_volumeTreeCache->getBricksCacheManager()->getTimeStampList();

	// Unmap GigaVoxels graphics resources from CUDA environment in order to be used by OpenGL
	// non-overlapping range
	//--------------------------------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idUnmapResources = nvtxRangeStartA( "UNMAP resources" );
#endif
	//--------------------------------------------------------------
	updateBufferArray->unmapResource();
	nodeTimeStampArray->unmapResource();
	brickTimeStampArray->unmapResource();
	volTreeChildArray->unmapResource();
	volTreeDataArray->unmapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->unmapResource();
#ifdef GV_NSIGHT_PROLIFING
	// TEST -------------------------------------
	GS_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------
	//--------------------------------------------------------------
	nvtxRangeEnd( idUnmapResources );
#endif
	//--------------------------------------------------------------

	/////////////////////////////////////////////////////////////////////
	//
	//C:\Program Files\NVIDIA GPU Computing Toolkit\nvToolsExt\lib\x64nvToolsExt64_1.lib
	//
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idTestMapResources = nvtxRangeStartA( "MAP test resources" );
	updateBufferArray->mapResource();
	nodeTimeStampArray->mapResource();
	brickTimeStampArray->mapResource();
	volTreeChildArray->mapResource();
	volTreeDataArray->mapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
	GS_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	nvtxRangeEnd( idTestMapResources );
	//--------------------------------------------------------------
	nvtxRangeId_t idEnhancedUnmapResources = nvtxRangeStartA( "enhanced UNMAP resources" );
#endif
	//GS_CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 6, &_graphicsResources[ 0 ], 0 ) );
#ifdef GV_NSIGHT_PROLIFING
	GS_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	nvtxRangeEnd( idEnhancedUnmapResources );
#endif
	/////////////////////////////////////////////////////////////////////

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idSetGLUniforms = nvtxRangeStartA( "Set GL uniforms" );
#endif

	// Returns location of uniform variables
	GLint volTreeChildArrayLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolChildArray" );
	GLint volTreeDataArrayLoc = glGetUniformLocation( _shaderProgram->_program, "uNodePoolDataArray" );
	GLint updateBufferArrayLoc = glGetUniformLocation( _shaderProgram->_program, "uUpdateBufferArray" );
	GLint nodeTimeStampArrayLoc = glGetUniformLocation( _shaderProgram->_program, "uNodeTimeStampArray" );
	GLint brickTimeStampArrayLoc = glGetUniformLocation( _shaderProgram->_program, "uBrickTimeStampArray" );
	GLint currentTimeLoc = glGetUniformLocation( _shaderProgram->_program, "uCurrentTime" );

	// Note :
	// glBindImageTextureEXT() command binds a single level of a texture to an image unit
	// for the purpose of reading and writing it from shaders.
	//
	// Specification :
	// void glBindImageTexture( GLuint  unit,  GLuint  texture,  GLint  level,  GLboolean  layered,  GLint  layer,  GLenum  access,  GLenum  format );

	// Data Production Mnagement
	//
	// - buffer of requests
	glBindImageTextureEXT( 0, _updateBufferTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( updateBufferArrayLoc, 0 );
	// - nodes time stamps buffer
	glBindImageTextureEXT( 1, _nodeTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( nodeTimeStampArrayLoc, 1 );
	// - bricks time stamps buffer
	glBindImageTextureEXT(2, _brickTimeStampTBO, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32UI );
	glUniform1i( brickTimeStampArrayLoc, 2 );

	// Node Pool
	//
	// - child array
	glBindImageTextureEXT(3, _childArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glUniform1i( volTreeChildArrayLoc, 3 );
	// - data array
	glBindImageTextureEXT( 4, _dataArrayTBO, 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32UI );
	glUniform1i( volTreeDataArrayLoc, 4 );

	// Time
	glUniform1ui( currentTimeLoc, this->_currentTime );

	// Retrieve node pool and brick pool resolution
	uint3 nodePoolRes = this->_volumeTree->_nodePool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	uint3 brickPoolRes = this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getResolution();
	// Retrieve node cache and brick cache size
	uint3 nodeCacheSize = nodeTimeStampArray->getResolution();
	uint3 brickCacheSize = brickTimeStampArray->getResolution();
	//glUniform3ui( glGetUniformLocation( _shaderProgram->_program, "nodeCacheSize" ),
	//	nodeCacheSize.x, nodeCacheSize.y, nodeCacheSize.z );
	glUniform3ui( glGetUniformLocation( _shaderProgram->_program, "uBrickCacheSize" ),
		brickCacheSize.x, brickCacheSize.y, brickCacheSize.z );

	// Data Pool
	GLint dataPoolLoc = glGetUniformLocation( _shaderProgram->_program, "uDataPool" );
	glUniform1i( dataPoolLoc, 0 );

	//glUniform3f( glGetUniformLocation( _shaderProgram->_program, "nodePoolResInv" ),
	//	1.0f / (GLfloat)nodePoolRes.x, 1.0f / (GLfloat)nodePoolRes.y, 1.0f / (GLfloat)nodePoolRes.z );
	glUniform3f( glGetUniformLocation( _shaderProgram->_program, "uBrickPoolResInv" ),
		1.0f / (GLfloat)brickPoolRes.x, 1.0f / (GLfloat)brickPoolRes.y, 1.0f / (GLfloat)brickPoolRes.z );

	glUniform1ui( glGetUniformLocation( _shaderProgram->_program, "uMaxDepth" ), this->_volumeTree->getMaxDepth() );

	//glUniform1f( glGetUniformLocation( _shaderProgram->_program, "frustumC" ), viewContext.frustumC );
	//glUniform1f( glGetUniformLocation( _shaderProgram->_program, "frustumD" ), viewContext.frustumD );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// Bind user data as 3D texture for rendering
	//
	// Content of one voxel has been defined as :
	// typedef Loki::TL::MakeTypelist< uchar4 >::Result DataType;
	glActiveTexture( GL_TEXTURE0 );
	glBindTexture( GL_TEXTURE_3D, this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->getBufferName() );

	GLuint vploc = glGetAttribLocation( _shaderProgram->_program, "iPosition" );

	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	glFlush();
	glFinish();
#endif
	// TEST -------------------------------------

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeEnd( idSetGLUniforms );
#endif

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idGLRender = nvtxRangeStartA( "GL render" );
#endif

	// Set projection matrix to identity
	glMatrixMode( GL_PROJECTION );
	glPushMatrix();
	glLoadIdentity();

	// Set model/view matrix to identity
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();
	glLoadIdentity();

	// Draw a quad on full screen
	glBegin( GL_QUADS );
	glVertexAttrib2f( vploc, -1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f, -1.0f );
	glVertexAttrib2f( vploc,  1.0f,  1.0f );
	glVertexAttrib2f( vploc, -1.0f,  1.0f );
	glEnd();

	// Restore previous projection matrix
	glMatrixMode( GL_PROJECTION );
	glPopMatrix();

	// Restore previous model/view matrix
	glMatrixMode( GL_MODELVIEW );
	glPopMatrix();

	// Unbind user data as 3D texture for rendering
	glBindTexture( GL_TEXTURE_3D, 0 );

	// Stop using shader program
	//
	// Programmable processors will be disabled and fixed functionality will be used
	// for both vertex and fragment processing.
	glUseProgram( 0 );

	//glMemoryBarrierEXT( GL_BUFFER_UPDATE_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_SHADER_IMAGE_ACCESS_BARRIER_BIT_EXT );
	//glMemoryBarrierEXT( GL_ALL_BARRIER_BITS_EXT );

	// Check for OpenGL error(s)
	GV_CHECK_GL_ERROR();

	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	glFlush();
	glFinish();
#endif
	// TEST -------------------------------------

#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeEnd( idGLRender );
#endif

	// Map GigaVoxels graphics resources in order to be used by CUDA environment
	//--------------------------------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	nvtxRangeId_t idMapResources = nvtxRangeStartA( "MAP resources" );
#endif
	//--------------------------------------------------------------
	updateBufferArray->mapResource();
	nodeTimeStampArray->mapResource();
	brickTimeStampArray->mapResource();
	volTreeChildArray->mapResource();
	volTreeDataArray->mapResource();
	this->_volumeTree->_dataPool->getChannel( Loki::Int2Type< 0 >() )->mapResource();
	// TEST -------------------------------------
#ifdef GV_NSIGHT_PROLIFING
	GS_CUDA_SAFE_CALL( cudaDeviceSynchronize() );
	// TEST -------------------------------------
	//--------------------------------------------------------------
	nvtxRangeEnd( idMapResources );
#endif
	//--------------------------------------------------------------

	// Disable blending and enable writing into depth buffer
	glDisable( GL_BLEND );
	//glEnable( GL_DEPTH_TEST );
	glDepthMask( GL_TRUE );

	CUDAPM_STOP_EVENT_GPU( gv_rendering );

	CUDAPM_START_EVENT( vsrender_post_frame );
	CUDAPM_START_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame_unmapbuffers );
	CUDAPM_STOP_EVENT( vsrender_post_frame );

	GV_CHECK_CUDA_ERROR( "RendererVolTreeGLSL::render" );
}

/******************************************************************************
 * Get the cone aperture scale
 *
 * @return the cone aperture scale
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
float VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getConeApertureScale() const
{
	return _coneApertureScale;
}

/******************************************************************************
 * Set the cone aperture scale
 *
 * @param pValue the cone aperture scale
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setConeApertureScale( float pValue )
{
	_coneApertureScale = pValue;
}

/******************************************************************************
 * Get the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
 *
 * @return the max number of loops
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
unsigned int VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getMaxNbLoops() const
{
	return _maxNbLoops;
}

/******************************************************************************
 * Set the max number of loops during the main GigaSpace pipeline pass (GLSL shader)
 *
 * @param pValue the max number of loops
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setMaxNbLoops( unsigned int pValue )
{
	_maxNbLoops = pValue;
}

/******************************************************************************
 * Get the environment mapping's reflection coefficient
 *
 * @return the environment mapping's reflection coefficient
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
float VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getReflectionCoeff() const
{
	return _reflectionCoeff;
}

/******************************************************************************
 * Get the environment mapping's reflection coefficient
 *
 * @param pValue the environment mapping's reflection coefficient
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setReflectionCoeff( float pValue )
{
	_reflectionCoeff = pValue;
}

/******************************************************************************
 * Get the environment mapping's refraction coefficient
 *
 * @return the environment mapping's refraction coefficient
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
float VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::getRefractionCoeff() const
{
	return _refractionCoeff;
}

/******************************************************************************
 * Get the environment mapping's refraction coefficient
 *
 * @param pValue the environment mapping's refraction coefficient
 ******************************************************************************/
template< typename TVolumeTreeType, typename TVolumeTreeCacheType, typename TSampleShader >
void VolumeTreeRendererGLSL< TVolumeTreeType, TVolumeTreeCacheType, TSampleShader >
::setRefractionCoeff( float pValue )
{
	_refractionCoeff = pValue;
}
