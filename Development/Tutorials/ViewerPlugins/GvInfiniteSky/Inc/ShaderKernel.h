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

#ifndef _SHADER_KERNEL_H_
#define _SHADER_KERNEL_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GigaVoxels
#include <GvRendering/GsIRenderShader.h>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

// we are not drawing objects but light sources (stars)
#define LIGHT_SOURCE 1

// Enabled if we want to represent fog only in the cube
#define LOCAL_FOG 0

/**
 * Light position
 */
__constant__ float3 cLightPosition;

/**
 * Stop if we reached our maximum opacity
 */
__device__ static const float cOpacityThreshold = 0.99f;

/**
 * Spheres ray-tracing parameters
 */
__constant__ bool cShaderUseUniformColor;
__constant__ float4 cShaderUniformColor;
__constant__ bool cShaderAnimation;
__constant__ bool cShaderBlurSphere;
__constant__ bool cShaderFog;
__constant__ float cShaderFogDensity;
__constant__ float4 cShaderFogColor;
__constant__ bool cShaderLightSourceType;
__constant__ bool cShading;
__constant__ bool cShaderBugCorrection;
__constant__ float cSphereIlluminationCoeff;
__constant__ unsigned int cScreenSpaceCoeff;
__constant__ bool cScreenBasedCriteria;
__constant__ unsigned int cNbMirrorReflections;
__constant__ float3 cNbCameraReflections;

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
 * @struct ShaderKernel
 *
 * @brief The ShaderKernel struct provides the way to shade the data structure.
 *
 * It is used in conjonction with the base class GsIRenderShader to implement the shader functions.
 */
struct ShaderKernel : public GvRendering::GsIRenderShader< ShaderKernel >
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * This method is called just before the cast of a ray. Use it to initialize any data
	 *  you may need. You may also want to modify the initial distance along the ray (tTree).
	 *
	 * @param pRayStartTree the starting position of the ray in octree's space.
	 * @param pRayDirTree the direction of the ray in octree's space.
	 * @param pTTree the distance along the ray's direction we start from.
	 */
	__device__
	inline void preShadeImpl( const float3& pRayStartTree, const float3& pRayDirTree, float& pTTree );

	/**
	 * This method is called after the ray stopped or left the bounding
	 * volume. You may want to do some post-treatment of the color.
	 */
	__device__
	inline void postShadeImpl( /*int pCounter*/ );

	/**
	 * This method returns the cone aperture for a given distance.
	 *
	 * @param pTTree the current distance along the ray's direction.
	 *
	 * @return the cone aperture
	 */
	__device__
	inline float getConeApertureImpl( const float pTTree ) const;

	/**
	 * This method returns the final rgba color that will be written to the color buffer.
	 *
	 * @return the final rgba color.
	 */
	__device__
	inline float4 getColorImpl() const;

	/**
	 * This method is called before each sampling to check whether or not the ray should stop.
	 *
	 * @param pRayPosInWorld the current ray's position in world space.
	 *
	 * @return true if you want to continue the ray. false otherwise.
	 */
	__device__
	inline bool stopCriterionImpl( const float3& pRayPosInWorld ) const;

	/**
	 * This method is called to know if we should stop at the current octree's level.
	 *
	 * @param pElementSize the desired element size in the current octree level.
	 *
	 * @param pConeAperture the ConeAperture at the considered point
	 *
	 * @return false if you want to stop at the current octree's level. true otherwise.
	 */
	__device__
    inline bool descentCriterionImpl( const float pElementSize, const float pConeAperture ) const;

	/**
	 * This method is called for each sample. For example, shading or secondary rays
	 * should be done here.
	 *
	 * @param pBrickSampler brick sampler
	 * @param pSamplePosScene position of the sample in the scene
	 * @param pRayDir ray direction
	 * @param pRayStep ray step
	 * @param pConeAperture cone aperture
	 */
	template< typename TSamplerType >
	__device__
	inline void runImpl( const TSamplerType& pBrickSampler, const float3 pSamplePosScene,
						const float3 pRayDir, float& pRayStep, const float pConeAperture );

    float4 getFogColor();

	float3 _renderViewContext;
	float _distanceBeforeReflection;

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * Accumulated color during ray casting
	 */
    float3 _accColor;

    /**
     * Accumulated opacity during ray casting
     */
    float _accTransparency;

    /**
     * Accumulated fog depth
     */
    float _accFogDepth;



	/******************************** METHODS *********************************/

};

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

#include "ShaderKernel.inl"

#endif // !_SHADER_KERNEL_H_
