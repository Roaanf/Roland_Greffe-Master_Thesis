﻿////////////////////////////////////////////////////////////////////////////////
//
// FRAGMENT SHADER
//
// Sky box rendering
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

////////////////////////////////////////////////////////////////////////////////
// OUTPUT
////////////////////////////////////////////////////////////////////////////////

// Ouput color
layout (location = 0) out vec4 oColor;

////////////////////////////////////////////////////////////////////////////////
// PROGRAM
////////////////////////////////////////////////////////////////////////////////
void main()
{
	oColor = vec4( 1.0, 1.0, 1.0, 1.0 );
}