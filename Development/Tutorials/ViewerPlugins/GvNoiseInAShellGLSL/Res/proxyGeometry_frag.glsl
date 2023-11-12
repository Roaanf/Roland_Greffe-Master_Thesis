////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// Proxy Geometry
//
// - Generate a depth map of a mesh faces
// - Depending of the fixed pipeline configuration, it will be based on closests or farthest faces
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Vertex position in Eye space
in vec4 EyeVertexPosition;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

layout (location = 0) out float FragmentDepth;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	FragmentDepth = length( EyeVertexPosition.xyz );
}