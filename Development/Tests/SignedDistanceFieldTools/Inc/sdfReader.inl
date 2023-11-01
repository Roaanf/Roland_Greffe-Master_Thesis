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

/******************************************************************************
 ****************************** INLINE DEFINITION *****************************
 ******************************************************************************/

namespace VolumeData
{

/******************************************************************************
 * Get the size of the data.
 * Resolution of data (4 components 3D float data) [size is identic on each dimension]
 *
 * @return the data size
 ******************************************************************************/
inline int sdfReader::getSize() const
{
	return _size;
};

/******************************************************************************
 * Get the buffer of data
 *
 * @return the pointer on the data buffer
 ******************************************************************************/
inline float* sdfReader::getData()
{
	return _data;
};

/******************************************************************************
 * Get the data at given 3D indexed position and component in the buffer of data
 *
 * @param pX x component of 3D indexed position of data
 * @param pY y component of 3D indexed position of data
 * @param pZ z component of 3D indexed position of data
 * @param pComponent component of the data (data has 4 components)
 *
 * @return a reference on the data
 ******************************************************************************/
inline float& sdfReader::get( int pX, int pY, int pZ, int pComponent )
{
	return _data[ pComponent + ( pX + ( pY + pZ * _size ) * _size ) * 4 ];
};

}
