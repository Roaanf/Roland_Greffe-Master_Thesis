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

#include "ParticleSystem.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// STL
#include <iostream>

// System
#include <cstdlib>

// Cuda
#include <cuda_runtime.h>

// GigaVoxels
#include <GvCore/GsVectorTypesExt.h>
#include <GvCore/GsError.h>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

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
ParticleSystem::ParticleSystem( const float3& pPoint1, const float3& pPoint2 )
:	_p1( pPoint1 )
,	_p2( pPoint2 )
,	_nbParticles( 998 )
,	_d_particleBuffer( NULL )
,	_sphereRadiusFader( 1.f )
,	_fixedSizeSphereRadius( 0.f )
{
    _particleBuffer = new float4[_nbParticles];
    initBuf();
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
ParticleSystem::~ParticleSystem()
{
	// TO DO
	// Handle destruction
	// ...
}

/******************************************************************************
 * Initialise le buffer GPU contenant les positions
 ******************************************************************************/
/*
void ParticleSystem::initGPUBuf()
{
    if ( _d_particleBuffer != NULL )
    {
        GS_CUDA_SAFE_CALL( cudaFree( _d_particleBuffer ) );
        _d_particleBuffer = NULL;
    }

    float4* part_buf = new float4[ _nbParticles ];

    for ( unsigned int i = 0; i < _nbParticles; ++i )
    {
        // Position (generee aleatoirement)
        float4 sphere = genPos( rand() );
        part_buf[ i ] = sphere;
    }


    size_t size = _nbParticles * sizeof( float4 );

    //_d_particleBuffer = new GvCore::GsLinearMemory( make_int3( _nbParticles, 1, 1 ), 0 );
    if ( cudaSuccess != cudaMalloc( &_d_particleBuffer, size ) )
    {
        return;
    }

    GS_CUDA_SAFE_CALL( cudaMemcpy( _d_particleBuffer, part_buf, size, cudaMemcpyHostToDevice ) );

    // TO DO
    // Delete the temporary buffer : part_buf
    // ...
}
*/

void ParticleSystem::initBuf()
{

    for ( unsigned int i = 0; i < _nbParticles; ++i )
    {
        // Position (generee aleatoirement)
        float4 sphere = genPos( rand() );
        _particleBuffer[i] = sphere;
    }

    // TO DO
    // Delete the temporary buffer : part_buf
    // ...
}

void ParticleSystem::loadGPUBuf(){

    if ( _d_particleBuffer != NULL )
    {
        GS_CUDA_SAFE_CALL( cudaFree( _d_particleBuffer ) );
        _d_particleBuffer = NULL;
    }
    size_t size = _offset * sizeof( float4 );

    //_d_particleBuffer = new GvCore::GsLinearMemory( make_int3( _nbParticles, 1, 1 ), 0 );
    if ( cudaSuccess != cudaMalloc( &_d_particleBuffer, size ) )
    {
        return;
    }

    GS_CUDA_SAFE_CALL( cudaMemcpy( _d_particleBuffer, _particleBuffer, size, cudaMemcpyHostToDevice ) );


}

/******************************************************************************
 * Get the buffer of data (sphere positions and radius)
 *
 * @return the buffer of data (sphere positions and radius)
 ******************************************************************************/
float4* ParticleSystem::getGPUBuf()
{
	return _d_particleBuffer;
}

/******************************************************************************
 * Genere une position aleatoire
 *
 * @param pSeed ...
 ******************************************************************************/
float4 ParticleSystem::genPos( int pSeed )
{
    float4 p;

	srand( pSeed );

	float min;	// min de l'interval des valeurs sur l'axe
	float max;	// max de l'interval des valeurs sur l'axe

    // Radius
    p.w = .005f + static_cast< float >( rand() ) / ( static_cast< float >( RAND_MAX ) / ( 0.02f - 0.005f ) );	// rayon de l'etoile dans [0.005 : 0.02]
    // Global size gain
    p.w *= _sphereRadiusFader;

    // genere la coordonnee en X
	if ( _p1.x < _p2.x )
	{
		min = _p1.x;
		max = _p2.x;
	}
	else
	{
		min = _p2.x;
		max = _p1.x;
	}
    p.x = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

    // genere la coordonnee en Y
	if ( _p1.y < _p2.y )
	{
		min = _p1.y;
		max = _p2.y;
	}
	else
	{
		min = _p2.y;
		max = _p1.y;
	}
    p.y = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

    // genere la coordonnee en Z
	if ( _p1.z < _p2.z )
	{
		min = _p1.z;
		max = _p2.z;
	}
	else
	{
		min = _p2.z;
		max = _p1.z;
	}
    p.z = (min+p.w+p.w) + (float)rand() / ((float)RAND_MAX / ((max-p.w-p.w)-(min+p.w+p.w)));

	return p;
}

/******************************************************************************
 * Get the number of particles
 *
 * @return the number of particles
 ******************************************************************************/
unsigned int ParticleSystem::getNbParticles() const
{
    //return _nbParticles;
    return _offset;
}


/******************************************************************************
 * Set the number of particles
 *
 * @param pValue the number of particles
 ******************************************************************************/
void ParticleSystem::setNbParticles( unsigned int pValue )
{
    //_nbParticles = pValue;
    _offset = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getSphereRadiusFader() const
{
	return _sphereRadiusFader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setSphereRadiusFader( float pValue )
{
	_sphereRadiusFader = pValue;
}

/******************************************************************************
 * ...
 ******************************************************************************/
float ParticleSystem::getFixedSizeSphereRadius() const
{
	return _fixedSizeSphereRadius;
}

/******************************************************************************
 * ...
 ******************************************************************************/
void ParticleSystem::setFixedSizeSphereRadius( float pValue )
{
	_fixedSizeSphereRadius = pValue;
}
