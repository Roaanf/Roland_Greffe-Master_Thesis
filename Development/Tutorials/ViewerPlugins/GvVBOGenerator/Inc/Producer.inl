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

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvCore/GsIProviderKernel.h>

// Project
#include "ParticleSystem.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::Producer()
:	GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >()
,	_particleSystem( NULL )
{
}

/******************************************************************************
 * Destructor
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
Producer< TDataStructureType, TDataProductionManager >
::~Producer()
{
	// Finalize the producer and its particle system
	finalize();
}

/******************************************************************************
 * Initialize producer and generate particles
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::initialize( GvStructure::GsIDataStructure* pDataStructure, GvStructure::GsIDataProductionManager* pDataProductionManager )
{
	// Call parent class
	GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >::initialize( pDataStructure, pDataProductionManager );

	// Points definissant l'interval du cube
	float3 p1 = make_float3( 0.f, 0.f, 0.f );
	float3 p2 = make_float3( 1.f, 1.f, 1.f );

	// Nombre de particules a generer aleatoirement dans le cube
	_particleSystem = new ParticleSystem( p1, p2 );
}

/******************************************************************************
 * Finalize the producer and its particle system
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::finalize()
{
	// TO DO
	// Check if there are special things to do here... ?
	// ...
	delete _particleSystem;
	_particleSystem = NULL;
}

/******************************************************************************
 * Spheres ray-tracing
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline unsigned int Producer< TDataStructureType, TDataProductionManager >
::getNbPoints() const
{
	unsigned int nbPoints = 0;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		nbPoints = _particleSystem->getNbParticles();
	}

	return nbPoints;
}

/******************************************************************************
 * Spheres ray-tracing
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setNbPoints( unsigned int pValue )
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		_particleSystem->setNbParticles( pValue );
	}

	// Update the particle system
	updateParticleSystem();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline float Producer< TDataStructureType, TDataProductionManager >
::getPointSizeFader() const
{
	float pointSizeFader = 0.f;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		pointSizeFader = _particleSystem->getPointSizeFader();
	}

	return pointSizeFader;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setPointSizeFader( float pValue )
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		_particleSystem->setPointSizeFader( pValue );
	}

	// Update the particle system
	updateParticleSystem();
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline float Producer< TDataStructureType, TDataProductionManager >
::getFixedSizePointSize() const
{
	float fixedSizePointSize = 0.f;

	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		fixedSizePointSize = _particleSystem->getFixedSizePointSize();
	}

	return fixedSizePointSize;
}

/******************************************************************************
 * ...
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::setFixedSizePointSize( float pValue )
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		_particleSystem->setFixedSizePointSize( pValue );
	}

	// Update the particle system
	updateParticleSystem();
}

/******************************************************************************
 * Update the associated particle system
 ******************************************************************************/
template< typename TDataStructureType, typename TDataProductionManager >
inline void Producer< TDataStructureType, TDataProductionManager >
::updateParticleSystem()
{
	assert( _particleSystem != NULL );
	if ( _particleSystem != NULL )
	{
		// Re-init spheres buffer
		_particleSystem->initGPUBuf();
	
		// Update Kernel Producer info
		this->_kernelProducer.setPositionBuffer( _particleSystem->getGPUBuf() );
	}
}
