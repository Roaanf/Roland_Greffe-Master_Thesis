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

#include "ProxyGeometry.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GsGraphics/GsShaderProgram.h>
#include <GvCore/GsError.h>

// Project
#include "Mesh.h"

// glm
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtx/transform2.hpp>
//#include <glm/gtx/projection.hpp>

// Cuda
#include <vector_types.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QImage>

// STL
#include <iostream>

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GigaVoxels
using namespace GsGraphics;

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
ProxyGeometry::ProxyGeometry()
:	_mesh( NULL )
,	_shaderProgram( NULL )
,	_meshShaderProgram( NULL )
,	_frameBuffer( 0 )
,	_depthMinTex( 0 )
,	_depthMaxTex( 0 )
,	_depthTex( 0 )
,	_bufferWidth( 0 )
,	_bufferHeight( 0 )
,	_filename()
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ProxyGeometry::~ProxyGeometry()
{
	finalize();
}

/******************************************************************************
 * Initialize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool ProxyGeometry::initialize()
{
	assert( ! _filename.empty() );

	bool statusOK = false;

	const QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );
	const QString shaderRepository = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvNoiseInAShellGLSL" );

	const QString vertexShaderFilename = shaderRepository + QDir::separator() + QString( "proxyGeometry_vert.glsl" );
	const QString fragmentShaderFilename = shaderRepository + QDir::separator() + QString( "proxyGeometry_frag.glsl" );
	
	// Initialize shader program
	_shaderProgram = new GsShaderProgram();
	statusOK = _shaderProgram->addShader( GsShaderProgram::eVertexShader, vertexShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->addShader( GsShaderProgram::eFragmentShader, fragmentShaderFilename.toStdString() );
	assert( statusOK );
	statusOK = _shaderProgram->link();
	assert( statusOK );
		
	// Initialize mesh
	_mesh = new Mesh();
	//IMesh::ShaderProgramConfiguration shaderProgramConfiguration;
	//shaderProgramConfiguration._shaders[ GsGraphics::GsShaderProgram::eVertexShader ] = "xxx_vert.glsl";
	//shaderProgramConfiguration._shaders[ GsGraphics::GsShaderProgram::eFragmentShader ] = "xxx_frag.glsl";
	//_mesh->setShaderProgramConfiguration( shaderProgramConfiguration );
	//const QString meshRepository = dataRepository + QDir::separator() + QString( "3DModels" );
	//const QString meshFilename = meshRepository + QDir::separator() + QString( "bunny.obj" );
	//statusOK = _mesh->load( meshFilename.toLatin1().constData() );
	statusOK = _mesh->load( _filename.c_str() );
	
	return true;
}

/******************************************************************************
 * Finalize
 *
 * @return a flag telling wheter or not it succeeds
 ******************************************************************************/
bool ProxyGeometry::finalize()
{
	delete _shaderProgram;
	_shaderProgram = NULL;

	delete _meshShaderProgram;
	_meshShaderProgram = NULL;

	glDeleteFramebuffers( 1, &_frameBuffer );
	glDeleteTextures( 1, &_depthTex );
	glDeleteTextures( 1, &_depthMaxTex );
	glDeleteTextures( 1, &_depthMinTex );

	delete _mesh;
	_mesh = NULL;

	return true;
}

/******************************************************************************
 * Render
 ******************************************************************************/
void ProxyGeometry::render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport )
{
	GLfloat clearColor[ 4 ];
	glGetFloatv( GL_COLOR_CLEAR_VALUE, clearColor );
	
	GLint location;

	GLenum drawBuffers[ 2 ] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };

	// Configure OpenGL pipeline
	glColorMask( GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE );
	glClearColor( -10000.0f, -10000.0f, -10000.0f, -10000.0f );	// check if not clamp to [ 0.0; 1.0 ]
	glEnable( GL_DEPTH_TEST );
	glDisable( GL_CULL_FACE );

	// Bind frame buffer
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	
	// ---- [ 1 ] - Ray MIN pass ----

	// Reset color and depth buffers of FBO
	glClearDepth( 1.0f );
	glDrawBuffers( 2, drawBuffers ); // to be sure to clear the 2 buffers
	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );	// glclear() peut �tre factoris� pour les 2 colorAttachment : 0_1
	// Specify the value used for depth buffer comparisons
	// - GL_LESS : passes if the incoming depth value is less than the stored depth value
	glDepthFunc( GL_LESS );
	// Set draw buffer - GL_COLOR_ATTACHMENT0
    glDrawBuffers( 1, drawBuffers );

	// Render proxy geometry
	_shaderProgram->use();

	// Set uniforms
	location = glGetUniformLocation( _shaderProgram->_program, "uModelViewMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pModelViewMatrix._array );
	}
	location = glGetUniformLocation( _shaderProgram->_program, "uProjectionMatrix" );
	if ( location >= 0 )
	{
		glUniformMatrix4fv( location, 1, GL_FALSE, pProjectionMatrix._array );
	}
	_mesh->render( float4x4(), float4x4(), int4() );	// TO DO : think about API => what parameters do we need in render() ?
	glUseProgram( 0 );

	// ----  [ 2 ] - Ray MAX pass ----

	// Reset color and depth buffers of FBO
	glClearDepth( 0.0f );
	//glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
	glClear( GL_DEPTH_BUFFER_BIT );	// COLOR has already been cleared
	// Specify the value used for depth buffer comparisons
	// - GL_GREATER : passes if the incoming depth value is greater than the stored depth value
	glDepthFunc( GL_GREATER );
	// Set draw buffer - GL_COLOR_ATTACHMENT1
	drawBuffers[ 0 ] = GL_COLOR_ATTACHMENT1;
    glDrawBuffers( 1, drawBuffers );
	
	// Render proxy geometry
	_shaderProgram->use();

	// No need to set uniforms
	// - there are the same...
	_mesh->render( float4x4(), float4x4(), int4() );	// TO DO : think about API => what parameters do we need in render() ?
	glUseProgram( 0 );

	// Unbind frame buffer
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );

	// Configure OpenGL pipeline
	glDisable( GL_DEPTH_TEST );
	glColorMask( GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE );

	glClearColor( clearColor[ 0 ], clearColor[ 1 ], clearColor[ 2 ], clearColor[ 3 ] );	// check if not clamp to [ 0.0; 1.0 ]
}

/******************************************************************************
 * Set buffer size
 *
 * @param pWidth buffer width
 * @param pHeight buffer height
 ******************************************************************************/
void ProxyGeometry::setBufferSize( int pWidth, int pHeight )
{
	_bufferWidth = pWidth;
	_bufferHeight = pHeight;

	if ( _frameBuffer )
	{
		glDeleteFramebuffers( 1, &_frameBuffer );
	}
	if ( _depthTex )
	{
		glDeleteTextures( 1, &_depthTex );
	}
	if ( _depthMaxTex )
	{
		glDeleteTextures( 1, &_depthMaxTex );
	}
	if ( _depthMinTex )
	{
		glDeleteTextures( 1, &_depthMinTex );
	}

	// Initialize graphics resource
	glGenTextures( 1, &_depthMinTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, _depthMinTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_R32F/*internal format*/, _bufferWidth, _bufferHeight, 0/*border*/, GL_RED/*format*/, GL_FLOAT/*type*/, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	// Initialize graphics resource
	glGenTextures( 1, &_depthMaxTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, _depthMaxTex );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_R32F/*internal format*/, _bufferWidth, _bufferHeight, 0/*border*/, GL_RED/*format*/, GL_FLOAT/*type*/, NULL );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	// Initialize graphics resource
	glGenTextures( 1, &_depthTex );
	glBindTexture( GL_TEXTURE_RECTANGLE, _depthTex );
	glTexImage2D( GL_TEXTURE_RECTANGLE, 0/*level*/, GL_DEPTH_COMPONENT32F/*internal format*/, _bufferWidth, _bufferHeight, 0/*border*/, GL_DEPTH_COMPONENT/*format*/, GL_FLOAT/*type*/, NULL );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	glBindTexture( GL_TEXTURE_RECTANGLE, 0 );

	// Initialize graphics resource
	glGenFramebuffers( 1, &_frameBuffer );
	glBindFramebuffer( GL_FRAMEBUFFER, _frameBuffer );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_RECTANGLE, _depthMinTex, 0/*level*/ );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_RECTANGLE, _depthMaxTex, 0/*level*/ );
	glFramebufferTexture2D( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_RECTANGLE, _depthTex, 0/*level*/ );
	glBindFramebuffer( GL_FRAMEBUFFER, 0 );
	// TO DO
	// - test completeness
	// ...
}

/******************************************************************************
 * Get the 3D model filename to load
 *
 * @return the 3D model filename to load
 ******************************************************************************/
const string& ProxyGeometry::get3DModelFilename() const
{
	return _filename;
}

/******************************************************************************
 * Set the 3D model filename to load
 *
 * @param pFilename the 3D model filename to load
 ******************************************************************************/
void ProxyGeometry::set3DModelFilename( const string& pFilename )
{
	_filename = pFilename;
}

/******************************************************************************
 * Get the associated mesh
 *
 * @return the associated mesh
 ******************************************************************************/
const IMesh* ProxyGeometry::getMesh() const
{
	return _mesh;
}

/******************************************************************************
 * Get the associated mesh
 *
 * @return the associated mesh
 ******************************************************************************/
IMesh* ProxyGeometry::editMesh()
{
	return _mesh;
}
