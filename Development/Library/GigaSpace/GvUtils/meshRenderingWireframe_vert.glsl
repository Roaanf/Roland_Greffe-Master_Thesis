////////////////////////////////////////////////////////////////////////////////
//
// VERTEX SHADER
//
// Spiral arms with wireframe
// - Gardner : cloud opacity simulation
//
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// VERSION
////////////////////////////////////////////////////////////////////////////////

#version 400

////////////////////////////////////////////////////////////////////////////////
// INPUT
////////////////////////////////////////////////////////////////////////////////

layout (location = 0) in vec3 iPosition;
//layout (location = 1) in vec3 iNormal;

////////////////////////////////////////////////////////////////////////////////
// UNIFORM
////////////////////////////////////////////////////////////////////////////////

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;
//uniform mat3 uNormalMatrix; // Simply use the model-view matrix if do only orthogonal transformations (i.e. translation and rotation)

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

out vec3 oPosition;
//out vec3 oNormal;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	// Pass vertex attributes
	oPosition = vec3( uModelViewMatrix * vec4( iPosition, 1.0 ) );
	//oNormal = normalize( uNormalMatrix * iNormal );

	// Output position in clip space
	gl_Position = uProjectionMatrix * uModelViewMatrix * vec4( iPosition, 1.0 );
}
