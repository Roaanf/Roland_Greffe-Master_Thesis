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

/******************************************************************************
 * Define a viewing transformation
 *
 * @param eyeX specifies the position of the eye point
 * @param eyeY specifies the position of the eye point
 * @param eyeZ specifies the position of the eye point
 * @param centerX specifies the position of the reference point
 * @param centerY specifies the position of the reference point
 * @param centerZ specifies the position of the reference point
 * @param upX specifies the direction of the up vector
 * @param upY specifies the direction of the up vector
 * @param upZ specifies the direction of the up vector
 *
 * @return the resulting viewing transformation
 ******************************************************************************/
inline float4x4 MatrixHelper::lookAt( float eyeX, float eyeY, float eyeZ,
								   float centerX, float centerY, float centerZ,
								   float upX, float upY, float upZ )
{
	// TO DO : Can be optimized ?
	float3 f = make_float3( centerX - eyeX, centerY - eyeY, centerZ - eyeZ );
	float3 up = make_float3( upX, upY, upZ );
	
	f = normalize( f );
	up = normalize( up );
	float3 s = cross( f, up );
	//s = normalize( s );	// need to normalize ?
	float3 u = cross( s, f );

	float4x4 m;
	// row 1 
	m.element( 0, 0 ) = s.x;
	m.element( 0, 1 ) = s.y;
	m.element( 0, 2 ) = s.z;
	m.element( 0, 3 ) = 0.0f;
	// row 2
	m.element( 1, 0 ) = u.x;
	m.element( 1, 1 ) = u.y;
	m.element( 1, 2 ) = u.z;
	m.element( 1, 3 ) = 0.0f;
	// row 3
	m.element( 2, 0 ) = - f.x;
	m.element( 2, 1 ) = - f.y;
	m.element( 2, 2 ) = - f.z;
	m.element( 2, 3 ) = 0.0f;
	// row 4
	m.element( 3, 0 ) = 0.0f;
	m.element( 3, 1 ) = 0.0f;
	m.element( 3, 2 ) = 0.0f;
	m.element( 3, 3 ) = 1.0f;

	return mul( m, translate( -eyeX, -eyeY, -eyeZ ) );
}

/******************************************************************************
 * Define a translation matrix
 *
 * @param x specify the x, y, and z coordinates of a translation vector
 * @param y specify the x, y, and z coordinates of a translation vector
 * @param z specify the x, y, and z coordinates of a translation vector
 *
 * @return the resulting translation matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::translate( float x, float y, float z)
{
	// TO DO : Can be optimized ?

	float4x4 res;

	// row 1 
	res.element( 0, 0 ) = 1.0f;
	res.element( 0, 1 ) = 0.0f;
	res.element( 0, 2 ) = 0.0f;
	res.element( 0, 3 ) = x;
	
	// row 2
	res.element( 1, 0 ) = 0.0f;
	res.element( 1, 1 ) = 1.0f;
	res.element( 1, 2 ) = 0.0f;
	res.element( 1, 3 ) = y;
	
	// row 3
	res.element( 2, 0 ) = 0.0f;
	res.element( 2, 1 ) = 0.0f;
	res.element( 2, 2 ) = 1.0f;
	res.element( 2, 3 ) = z;
	
	// row 4
	res.element( 3, 0 ) = 0.0f;
	res.element( 3, 1 ) = 0.0f;
	res.element( 3, 2 ) = 0.0f;
	res.element( 3, 3 ) = 1.0f;

	return res;
}

/******************************************************************************
 * Multiply two matrices
 *
 * @param a first matrix
 * @param b second matrix
 *
 * @return the resulting matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::mul( const float4x4& a, const float4x4& b )
{
	float4x4 res;

	// row 1
	res.element( 0, 0 ) = a.element( 0, 0 ) * b.element( 0, 0 ) + 
					      a.element( 0, 1 ) * b.element( 1, 0 ) + 
						  a.element( 0, 2 ) * b.element( 2, 0 ) +
						  a.element( 0, 3 ) * b.element( 3, 0 ) ;

	res.element( 0, 1 ) = a.element( 0, 0 ) * b.element( 0, 1 ) + 
					      a.element( 0, 1 ) * b.element( 1, 1 ) + 
						  a.element( 0, 2 ) * b.element( 2, 1 ) +
						  a.element( 0, 3 ) * b.element( 3, 1 ) ;

	res.element( 0, 2 ) = a.element( 0, 0 ) * b.element( 0, 2 ) + 
					      a.element( 0, 1 ) * b.element( 1, 2 ) + 
						  a.element( 0, 2 ) * b.element( 2, 2 ) +
						  a.element( 0, 3 ) * b.element( 3, 2 ) ;

	res.element( 0, 3 ) = a.element( 0, 0 ) * b.element( 0, 3 ) + 
					      a.element( 0, 1 ) * b.element( 1, 3 ) + 
						  a.element( 0, 2 ) * b.element( 2, 3 ) +
						  a.element( 0, 3 ) * b.element( 3, 3 ) ;

	// row 1
	res.element( 1, 0 ) = a.element( 1, 0 ) * b.element( 0, 0 ) + 
					      a.element( 1, 1 ) * b.element( 1, 0 ) + 
						  a.element( 1, 2 ) * b.element( 2, 0 ) +
						  a.element( 1, 3 ) * b.element( 3, 0 ) ;

	res.element( 1, 1 ) = a.element( 1, 0 ) * b.element( 0, 1 ) + 
					      a.element( 1, 1 ) * b.element( 1, 1 ) + 
						  a.element( 1, 2 ) * b.element( 2, 1 ) +
						  a.element( 1, 3 ) * b.element( 3, 1 ) ;

	res.element( 1, 2 ) = a.element( 1, 0 ) * b.element( 0, 2 ) + 
					      a.element( 1, 1 ) * b.element( 1, 2 ) + 
						  a.element( 1, 2 ) * b.element( 2, 2 ) +
						  a.element( 1, 3 ) * b.element( 3, 2 ) ;

	res.element( 1, 3 ) = a.element( 1, 0 ) * b.element( 0, 3 ) + 
					      a.element( 1, 1 ) * b.element( 1, 3 ) + 
						  a.element( 1, 2 ) * b.element( 2, 3 ) +
						  a.element( 1, 3 ) * b.element( 3, 3 ) ;

	// row 2
	res.element( 2, 0 ) = a.element( 2, 0 ) * b.element( 0, 0 ) + 
					      a.element( 2, 1 ) * b.element( 1, 0 ) + 
						  a.element( 2, 2 ) * b.element( 2, 0 ) +
						  a.element( 2, 3 ) * b.element( 3, 0 ) ;

	res.element( 2, 1 ) = a.element( 2, 0 ) * b.element( 0, 1 ) + 
					      a.element( 2, 1 ) * b.element( 1, 1 ) + 
						  a.element( 2, 2 ) * b.element( 2, 1 ) +
						  a.element( 2, 3 ) * b.element( 3, 1 ) ;

	res.element( 2, 2 ) = a.element( 2, 0 ) * b.element( 0, 2 ) + 
					      a.element( 2, 1 ) * b.element( 1, 2 ) + 
						  a.element( 2, 2 ) * b.element( 2, 2 ) +
						  a.element( 2, 3 ) * b.element( 3, 2 ) ;

	res.element( 2, 3 ) = a.element( 2, 0 ) * b.element( 0, 3 ) + 
					      a.element( 2, 1 ) * b.element( 1, 3 ) + 
						  a.element( 2, 2 ) * b.element( 2, 3 ) +
						  a.element( 2, 3 ) * b.element( 3, 3 ) ;
	// row 3
	res.element( 3, 0 ) = a.element( 3, 0 ) * b.element( 0, 0 ) + 
					      a.element( 3, 1 ) * b.element( 1, 0 ) + 
						  a.element( 3, 2 ) * b.element( 2, 0 ) +
						  a.element( 3, 3 ) * b.element( 3, 0 ) ;

	res.element( 3, 1 ) = a.element( 3, 0 ) * b.element( 0, 1 ) + 
					      a.element( 3, 1 ) * b.element( 1, 1 ) + 
						  a.element( 3, 2 ) * b.element( 2, 1 ) +
						  a.element( 3, 3 ) * b.element( 3, 1 ) ;

	res.element( 3, 2 ) = a.element( 3, 0 ) * b.element( 0, 2 ) + 
					      a.element( 3, 1 ) * b.element( 1, 2 ) + 
						  a.element( 3, 2 ) * b.element( 2, 2 ) +
						  a.element( 3, 3 ) * b.element( 3, 2 ) ;

	res.element( 3, 3 ) = a.element( 3, 0 ) * b.element( 0, 3 ) + 
					      a.element( 3, 1 ) * b.element( 1, 3 ) + 
						  a.element( 3, 2 ) * b.element( 2, 3 ) +
						  a.element( 3, 3 ) * b.element( 3, 3 ) ;
	return res;

}

/******************************************************************************
 * Define a transformation that produces a parallel projection (i.e. orthographic)
 *
 * @param left specify the coordinates for the left and right vertical clipping planes
 * @param right specify the coordinates for the left and right vertical clipping planes
 * @param bottom specify the coordinates for the bottom and top horizontal clipping planes
 * @param top specify the coordinates for the bottom and top horizontal clipping planes
 * @param nearVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
 * @param farVal specify the distances to the nearer and farther depth clipping planes. These values are negative if the plane is to be behind the viewer.
 *
 * @return the resulting orthographic matrix
 ******************************************************************************/
inline float4x4 MatrixHelper::ortho( float left, float right, float bottom, float top,	float nearVal, float farVal )
{
	// TO DO : Can be optimized ?
	float4x4 res;

	// row 1 
	res.element( 0, 0 ) = 2.0 / ( right - left );
	res.element( 0, 1 ) = 0.0;
	res.element( 0, 2 ) = 0.0;
	res.element( 0, 3 ) = - ( right + left ) / ( right - left );

	// row 2
	res.element( 1, 0 ) = 0.0;
	res.element( 1, 1 ) = 2.0 / ( top - bottom );
	res.element( 1, 2 ) = 0.0;
	res.element( 1, 3 ) = - ( top + bottom ) / ( top - bottom );

	// row 3
	res.element( 2, 0 ) = 0.0;
	res.element( 2, 1 ) = 0.0;
	res.element( 2, 2 ) = - 2.0 / ( farVal - nearVal );
	res.element( 2, 3 ) = - ( farVal + nearVal ) / ( farVal - nearVal );

	// row 4
	res.element( 3, 0 ) = 0.0;
	res.element( 3, 1 ) = 0.0;
	res.element( 3, 2 ) = 0.0;
	res.element( 3, 3 ) = 1.0;

	return res;
}
