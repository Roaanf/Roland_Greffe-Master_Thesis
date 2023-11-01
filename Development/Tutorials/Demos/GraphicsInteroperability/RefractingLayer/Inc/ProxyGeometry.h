/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

#ifndef _PROXYGEOMETRY_H_
#define _PROXYGEOMETRY_H_

#include <GL/glew.h>
#include <iostream>
#include "ShaderManager.h"
#include "Mesh.h"
#include <GvCore/vector_types_ext.h>


class ProxyGeometry {

private:
	Mesh* mesh;
	string filename;//obj file
	GLuint program;
	float innerDistance;
	bool water;
public:	
	//buffer dimensions
	int width; 
	int height;
	//IDs
	GLuint depthMinTex; 
	GLuint depthMaxTex;
	GLuint depthTex;
	GLuint normalTex;
	GLuint frameBuffer;
	void init();
	void initBuffers();
	void renderMin();
	void renderMax();
	void renderMinAndNorm();
	void setBufferWidth(int w);
	void setBufferHeight(int h);
	void setFilename(string s);
	Mesh* getMesh();
	void setInnerDistance(float d);
	ProxyGeometry(bool w = false);
	~ProxyGeometry();

};

#endif 