////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// Front to back peeling
// - This shader program is used to clip parts of geometry
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

// Depth texture of a previous front geometry
uniform sampler2DRect uDepthTexture;

// Mesh color (including alpha)
uniform vec4 uMeshColor;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Fragment color
layout (location = 0) out vec4 oFragmentColor;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Retrieve depth from depth texture
	float depth = texelFetch( uDepthTexture, ivec2( gl_FragCoord.x, glFragCoord.y ) ).r;

	// If incoming fragment's depth if in front of depth texture value, discard it.
	// This will clip parts of geometry.
	if ( gl_FragCoord.z <= depth )
	{
		discard;
	}

	// Write mesh color
	oFragmentColor = uMeshColor;
}
