/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/** 
 * @version 1.0
 */

#ifndef _PARTICLE_SYSTEM_H_
#define _PARTICLE_SYSTEM_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>

// GL
#include <GL/glew.h>

// Cuda
#include <vector_types.h>
#include <driver_types.h>

// STL
#include <string>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// GigaVoxels
namespace GvRendering
{
	class GsGraphicsResource;
}
namespace GsGraphics
{
	class GsShaderProgram;
}

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class ParticleSystem
 *
 * @brief The ParticleSystem class provides the mecanisms to generate star positions.
 *
 * ...
 */
class ParticleSystem
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Rendering type
	 */
	enum RenderingType
	{
		ePoints = 0,		// points
		ePointSprite,		// textured quads
		eParticleSystem		// instanced 3D models
	};

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Current number of points in the buffer to be drawned
	 */
	unsigned int _nbRenderablePoints;

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 *
	 * Recupere deux points de la diagonale du cube pour connaitre l'intervalle de valeurs
	 */
	ParticleSystem( const float3& pPoint1, const float3& pPoint2 );

	/**
	 * Destructor
	 */
	~ParticleSystem();

	/**
	 * Initialise le buffer GPU contenant les positions
	 */
	void initGPUBuf();

	/**
	  * Get the buffer of data (sphere positions and radius)
	 *
	 * @return the buffer of data (sphere positions and radius)
	 */
	// GsLinearMemory< float4 >* getGPUBuf();
	float4* getGPUBuf();

	/**
	 * Get the number of particles
	 *
	 * @return the number of particles
	 */
	unsigned int getNbParticles() const;

	/**
	 * Set the number of particles
	 *
	 * @param pValue the number of particles
	 */
	void setNbParticles( unsigned int pValue );

	/**
	 * Spheres ray-tracing methods
	 */
	float getPointSizeFader() const;
	void setPointSizeFader( float pValue );
	float getFixedSizePointSize() const;
	void setFixedSizePointSize( float pValue );
	//bool hasMeanSizeOfSpheres() const;
	//void setMeanSizeOfSpheres( bool pFlag );

	/**
	 * Render the particle system
	 */
	void render( const float4x4& pModelViewMatrix, const float4x4& pProjectionMatrix, const int4& pViewport );

	/**
	 * Get the associated graphics resource
	 * 
	 * return the associated graphics resource
	 */
	GvRendering::GsGraphicsResource* getGraphicsResource();

	/**
	 * ...
	 *
	 * return ...
	 */
	bool initGraphicsResources();

	/**
	 * ...
	 *
	 * return ...
	 */
	bool releaseGraphicsResources();

	bool hasShaderUniformColor() const;
	void setShaderUniformColorMode( bool pFlag );
	const float4& getShaderUniformColor() const;
	void setShaderUniformColor( float pR, float pG, float pB, float pA );
	bool hasShaderAnimation() const;
	void setShaderAnimation( bool pFlag );
	bool hasTexture() const;
	void setTexture( bool pFlag );
	const std::string& getTextureFilename() const;
	void setTextureFilename( const std::string& pFilename );
	
	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Rendering type
	 */
	RenderingType _renderingType;

	/**
	 * Point rendering's GLSL shader program
	 */
	GLuint _pointsShaderProgram;

	/**
	 * Point sprites rendering's GLSL shader program
	 */
	//GLuint _pointSpritesShaderProgram;

	/**
	 * Vertex buffer
	 */
	GLuint _positionBuffer;

	/**
	 * Vertex array object
	 */
	GLuint _vao;

	/**
	 * Sprite texture
	 */
	GLuint _spriteTexture;

	///**
	// * Current number of points in the buffer to be drawned
	// */
	//unsigned int _nbRenderablePoints;

	/**
	 * Graphics resource
	 */
	GvRendering::GsGraphicsResource* _graphicsResource;

	/**
	 * Uniform color
	 */
	float4 _uniformColor;

	/**
	 * Texture filename
	 */
	std::string _textureFilename;

	bool _shaderUseUniformColor;
	float4 _shaderUniformColor;
	bool _shaderAnimation;
	bool _hasTexture;

	/**
	 * Shader program
	 */
	GsGraphics::GsShaderProgram* _shaderProgramPoints;
	GsGraphics::GsShaderProgram* _shaderProgramPointSprite;
	GsGraphics::GsShaderProgram* _shaderProgramParticleSystem;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private :

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Point devant en bas a gauche du cube
	 */
	float3 _p1;

	/**
	 * Point derriere en haut a droite du cube
	 */
	float3 _p2;

	/**
	 * Max number of particles
	 */
	unsigned int _nbParticles;

	/**
	 * The buffer of data (sphere positions and radius)
	 */
	float4* _d_particleBuffer;

	/**
	 * Spheres ray-tracing parameters
	 */
	float _pointSizeFader;
	float _fixedSizePointSize;

	/******************************** METHODS *********************************/

	/**
	 * Genere une position aleatoire
	 *
	 * @param pSeed ...
	 */
	float3 genPos( int pSeed );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ParticleSystem.inl"

#endif
