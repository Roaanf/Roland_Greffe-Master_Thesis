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

// GigaVoxels
#include <GvUtils/GsSimpleHostProducer.h>
#include <GvUtils/GsShaderManager.h>

// Project
#include "ProducerKernel.h"
#include "Scene.h"
#include "helper_matrix.h"

// OpenGL
#include <GL/glew.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

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
	 * Type definition of the inherited parent class
	 */
	typedef GvUtils::GsSimpleHostProducer< ProducerKernel< TDataStructureType >, TDataStructureType, TDataProductionManager > ParentClassType;

	/**
	 * Type definition of the node tile resolution
	 */
	typedef typename TDataStructureType::NodeTileResolution NodeRes;

	/**
	 * Type definition of the brick resolution
	 */
	typedef typename TDataStructureType::BrickResolution BrickRes;

	/**
	 * Enumeration to define the brick border size
	 */
	enum
	{
		BorderSize = TDataStructureType::BrickBorderSize
	};

	/**
	 * Linear representation of a node tile
	 */
	//typedef typename TDataProductionManager::NodeTileResLinear NodeTileResLinear;
	typedef typename ParentClassType::NodeTileResLinear NodeTileResLinear;

	/**
	 * Type definition of the full brick resolution (i.e. with border)
	 */
	typedef typename TDataProductionManager::BrickFullRes BrickFullRes;
	//typedef typename ParentClassType::BrickFullRes BrickFullRes;
	
	/**
	 * Defines the data type list
	 */
	typedef typename TDataStructureType::DataTypeList DataTList;
	
	/**
	 * Typedef the kernel part of the producer
	 */
	typedef ProducerKernel< TDataStructureType > KernelProducerType;
	//typedef typename ParentClassType::KernelProducer KernelProducerType;

	/**
	 * This pool will contains an array for each voxel's field
	 */
	typedef GvCore::GPUPoolHost< GvCore::Array3D, DataTList > BricksPool;
		
	/******************************* ATTRIBUTES *******************************/
	
	/**
	 * Defines the maximum number of requests we can handle in one pass
	 */
	static const uint nbMaxRequests = 128;

	/******************************** METHODS *********************************/

	/**
	 * Constructor.
	 * Initialize all buffers.
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
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 0 (nodes)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 0 > corresponds to the index of the node pool
	 */
	inline virtual void produceData( uint pNumElems,
									GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
									GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
									Loki::Int2Type< 0 > );
	
	/**
	 * This method is called by the cache manager when you have to produce data for a given pool.
	 * Implement the produceData method for the channel 1 (bricks)
	 *
	 * @param pNumElems the number of elements you have to produce.
	 * @param pNodeAddressCompactList a list containing the addresses of the numElems nodes concerned.
	 * @param pElemAddressCompactList a list containing numElems addresses where you need to store the result.
	 * @param pGpuPool the pool for which we need to produce elements.
	 * @param pPageTable the page table associated to the pool
	 * @param Loki::Int2Type< 1 > corresponds to the index of the brick pool
	 */
	inline virtual void produceData( uint pNumElems,
									GvCore::GsLinearMemory< uint >* pNodesAddressCompactList,
									GvCore::GsLinearMemory< uint >* pElemAddressCompactList,
									Loki::Int2Type< 1 > );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * Produce nodes
	 *
	 * Node production is associated to node subdivision to refine data.
	 * With the help of an oracle, user has to tell what is inside each subregion
	 * of its children.
	 *
	 * @param pNbElements number of elements to process (i.e. nodes)
	 * @param pRequestListCodePtr localization code list on device
	 * @param pRequestListDepthPtr localization depth list on device
	 */
	inline void produceNodes( const uint pNbElements, const GvCore::GsLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GsLocalizationInfo::DepthType* pRequestListDepthPtr );

	/**
	 * Produce bricks
	 *
	 * Brick production is associated to fill brick with voxels.
	 *
	 * @param pNbElements number of elements to process (i.e. bricks)
	 * @param pRequestListCodePtr localization code list on device
	 * @param pRequestListDepthPtr localization depth list on device
	 * @param pElemAddressListPtr ...
	 */
	inline void produceBricks( const uint pNbElements, const GvCore::GsLocalizationInfo::CodeType* pRequestListCodePtr, const GvCore::GsLocalizationInfo::DepthType* pRequestListDepthPtr, const uint* pElemAddressListPtr );

	/**
	 * Helper function used to retrieve the number of voxels at a given level of resolution
	 *
	 * @param pLevel level of resolution
	 *
	 * @return the number of voxels at given level of resolution
	 */
	inline uint3 getLevelResolution( const uint pLevel ) const;

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Node pool
	 */
	GvCore::Array3D< uint >* nodesBuffer;

	/**
	 * Data pool
	 */
	BricksPool* bricksPool;

	/**
	 * List of localization code
	 */
	GvCore::GsLocalizationInfo::CodeType* requestListCode;

	/**
	 * List of localization depth
	 */
	GvCore::GsLocalizationInfo::DepthType* requestListDepth;

	/**
	 * List of localization code
	 */
	thrust::device_vector< GvCore::GsLocalizationInfo::CodeType >* requestListCodeDevice;

	/**
	 * List of localization depth
	 */
	thrust::device_vector< GvCore::GsLocalizationInfo::DepthType >* requestListDepthDevice;

	/**
	 * List of brick addresses in cache
	 */
	uint* requestListAddress;

	/**
	 * Shader program handle
	 *
	 * ...
	 */
	GLuint mDistProg;

	/**
	 * Shader program handle
	 *
	 * ...
	 */
	GLuint mPotentialProg;

	/**
	 * Shader program uniform variables ids
	 */
	GLuint mProjectionMatrixId;
	GLuint mModelViewMatrixId;
	GLuint mAxeId;
	GLuint mSliceId;
	GLuint mDistanceId;
	GLuint mDistanceXId;
	GLuint mDistanceYId;
	GLuint mDistanceZId;
	GLuint mPotentialId;
	GLuint mBrickAddressId;

	/**
	 * ...
	 */
	Scene mScene;

	/**
	 * ...
	 */
	int mWidth, mHeight, mDepth;

	/**
	 * Auxiliary texture used in the signed distance field algorithm
	 */
	GLuint _distanceTexture[ 3 ];
	
	/**
	 * ...
	 */
	GLuint mEmptyData;

	/******************************** METHODS *********************************/

	/**
	 * ...
	 *
	 * ...
	 *
	 * @param pBrickPos position of the brick (same as the position of the node minus the border)
	 * @param pXSize x size of the brick ( same as the size of the node plus the border )
	 * @param pYSize y size of the brick ( same as the size of the node plus the border )
	 * @param pZSize z size of the brick ( same as the size of the node plus the border )
	 * @param pDepth depth localization info of the brick
	 * @param pLocCode code localization info of the brick
	 * @param pAddressBrick address in cache where to write result of production
	 */
	inline void produceSignedDistanceField( float3 brickPos, float xSize, float ySize, float zSize, unsigned int depth, uint3 locCode, uint3 adressBrick );
	
	/**
	 * Copy constructor forbidden.
	 */
	Producer( const Producer& );

	/**
	 * Copy operator forbidden.
	 */
	Producer& operator=( const Producer& );

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "Producer.inl"

#endif // !_PRODUCER_H_
