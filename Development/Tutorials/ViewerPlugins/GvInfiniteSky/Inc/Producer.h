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

#ifndef _PRODUCER_H_
#define _PRODUCER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Gigavoxels
#include <GvUtils/GsSimpleHostProducer.h>

// Project
#include "ProducerKernel.h"

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// Project
class ParticleSystem;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class Producer
 *
 * @brief The Producer class provides the mecanisms to produce data
 * following data requests emitted during N-tree traversal (rendering).
 *
 * It is the main user entry point to produce data from CPU, for instance,
 * loading data from disk or procedurally generating data.
 *
 * This class is implements the mandatory functions of the GsIProvider base class.
 */
template< typename TDataStructureType, typename TDataProductionManager >
class Producer : public GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager >
{
	
	
	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/**
	 * Typedef the associated device-side producer
	 */
	typedef ProducerKernel
	<
		TDataStructureType
	>
	KernelProducerType;

	/**
	 * Type definition of the node page table
	 */
	typedef typename TDataProductionManager::NodePageTableType NodePageTableType;

	/**
	 * Type definition of the node page table
	 */
	typedef typename TDataProductionManager::BrickPageTableType DataPageTableType;

	/**
	 * Linear representation of a node tile
	 */
	typedef typename TDataProductionManager::NodeTileResLinear NodeTileResLinear;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataProductionManager::BrickFullRes BrickFullRes;

	/**
	 * Type definition of the node pool type
	 */
	typedef typename TDataStructureType::NodePoolType NodePoolType;

	/**
	 * Type definition of the data pool type
	 */
	typedef typename TDataStructureType::DataPoolType DataPoolType;

	/**
	 * Defines the size of a brick tile
	 */
	typedef typename TDataStructureType::BrickResolution BrickTileResolution;

	/**
	 * Defines the size of the border around a brick tile
	 */
	enum
	{
		BrickTileBorderSize = TDataStructureType::BrickBorderSize
	};

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Constructor
	 */
	Producer();

	/**
	 * Destructor
	 */
	virtual ~Producer();

	/**
	 * Initialize
	 *
	 * @param pDataStructure data structure
	 * @param pDataProductionManager data production manager
	 */
	virtual void initialize( GvStructure::GsIDataStructure* pDataStructure, GvStructure::GsIDataProductionManager* pDataProductionManager );

	/**
	 * Finalize
	 */
	virtual void finalize();
	

	/**
	 * Spheres ray-tracing methods
	 */
	unsigned int getNbSpheres() const;
	void setNbSpheres( unsigned int pValue );
    void generateNewParticleBuffer();
	float getSphereRadiusFader() const;
	void setSphereRadiusFader( float pValue );
    float getFixedSizeSphereRadius() const;
    void setFixedSizeSphereRadius( float pValue );
    //bool hasMeanSizeOfSpheres() const;
	//void setMeanSizeOfSpheres( bool pFlag );

	 /**
	  * Update the associated particle system
	  */
	 void updateParticleSystem();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Particle system
	 */
	ParticleSystem* _particleSystem;

	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Producer.inl"

#endif // !_PRODUCER_H_
