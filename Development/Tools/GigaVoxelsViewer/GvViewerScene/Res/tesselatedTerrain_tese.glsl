////////////////////////////////////////////////////////////////////////////////
//
// TESSELATION EVALUATION SHADER
//
// Terrain rendering
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// ...
layout( quads, fractional_even_spacing, cw ) in;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Transform matrices
uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
//uniform mat4 uModelViewProjectionMatrix;

// Heightmap
uniform sampler2D uHeightMap;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Texture coordinate
out vec2 TexCoord;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	vec2 uv = gl_in[ 0 ].gl_Position.xy + gl_TessCoord.st / 64.0;
	
	vec3 point;
	point.xz = uv;

	// Retrieve terrain's height
	point.y = texture( uHeightMap, uv ).r;
	float scale = 0.2;
	point.y *= scale;
	
	// Texture coordinates
	TexCoord = uv;

	// Send vertex to Clip space
	gl_Position = uProjectionMatrix * uModelViewMatrix * vec4( point, 1.0 );
}
