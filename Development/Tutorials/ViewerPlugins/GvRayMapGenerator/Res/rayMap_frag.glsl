////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// Points
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Image parameters
uniform vec2 uImageResolution;

// Camera parameters
uniform vec3 uCameraXAxis;
uniform vec3 uCameraYAxis;
uniform vec3 uCameraZAxis;
uniform float uCameraViewPlaneDistance;

// Fish Eye parameters
uniform float uFishEyePsiMaxAngle;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Output used to store ray direction
layout (location = 0) out vec4 oData;

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////

// Use subroutine to select behavior on-the-fly
subroutine vec3 getRayDirectionSubRoutineType( vec2 uv );
subroutine uniform getRayDirectionSubRoutineType getRayDirection;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Retrieve sample point in NDC space [-1;1]
	vec2 uv = ( gl_FragCoord.xy - uImageResolution * 0.5 + vec2( 0.5 ) );
	uv = uv / ( uImageResolution * 0.5 );

	// Generate ray direction
	vec3 rayDirection = getRayDirection( uv );

	// Write output data
	oData = vec4( rayDirection, 0.0 );
}

////////////////////////////////////////////////////////////////////////////////
// Generate ray direction
////////////////////////////////////////////////////////////////////////////////
subroutine( getRayDirectionSubRoutineType )
vec3 getClassicalRayDirection( vec2 uv )
{
	return ( uv.x * uCameraXAxis + uv.y * uCameraYAxis - /*uCameraViewPlaneDistance*/2.0 * uCameraZAxis );
}

////////////////////////////////////////////////////////////////////////////////
// Generate ray direction with Fish Eye nonlinear projection
////////////////////////////////////////////////////////////////////////////////
subroutine( getRayDirectionSubRoutineType )
vec3 getFishEyeRayDirection( vec2 uv )
{
	vec3 rayDirection = vec3( 0.0 );

	float r_squared = uv * uv;
	if ( r_squared <= 1.0 )
	{
		// psi( r ) = r * psiMax
		float r = sqrt( r_squared );
		float psi = r * radians( uFishEyePsiMaxAngle );

		float sin_psi = sin( psi );
		float cos_psi = cos( psi );

		float sin_alpha = uv.y / r;
		float cos_alpha = uv.x / r;

		rayDirection = sin_psi * cos_alpha * uCameraXAxis + sin_psi * sin_alpha * uCameraYAxis - cos_psi * uCameraZAxis;
	}

	return rayDirection;
}

////////////////////////////////////////////////////////////////////////////////
// Generate ray direction with Reflection Map nonlinear projection
////////////////////////////////////////////////////////////////////////////////
subroutine( getRayDirectionSubRoutineType )
vec3 getReflectionMapRayDirection( vec2 uv )
{
	return vec3( 0.0 );
}

////////////////////////////////////////////////////////////////////////////////
// Generate ray direction with Refraction Map nonlinear projection
////////////////////////////////////////////////////////////////////////////////
subroutine( getRayDirectionSubRoutineType )
vec3 getRefractionMapRayDirection( vec2 uv )
{
	return vec3( 0.0 );
}
