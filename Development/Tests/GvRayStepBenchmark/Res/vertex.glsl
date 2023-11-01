////////////////////////////////////////////////////////////////////////////////
//
// VERTEX SHADER
//
// Shadow Mapping
//
// - 2nd pass
//
// The scene is rendered from the point of view of the camera.
//
// Mandatory :
// - a FBO (frame buffer object) with a unique depth texture bound to its depth attachement
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

// Vertex position
layout (location = 0) in vec3 iVertexPosition;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

// Model-View matrix
uniform mat4 uModelViewMatrix;

// Model View Projection matrix
uniform mat4 uProjectionMatrix;

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Transform position from Model space to Clip space
	gl_Position = uProjectionMatrix * uModelViewMatrix * vec4( iVertexPosition, 1.0 );
}
