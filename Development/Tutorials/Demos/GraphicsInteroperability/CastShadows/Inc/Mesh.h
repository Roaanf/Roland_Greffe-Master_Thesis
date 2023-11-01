/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 */

/** 
 * @version 1.0
 */

#ifndef _MESH_H_
#define _MESH_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Project
#include "ShaderManager.h"

// Assimp
//for assimp 2
//#include <assimp/Importer.hpp> // C++ importer interface
//#include <assimp/assimp.hpp>
//#include <assimp/aiConfig.h>
#include <assimp/Importer.hpp> // C++ importer interface
#include <assimp/scene.h> // Output data structure
#include <assimp/postprocess.h> // Post processing flags

// GL
#include <GL/glew.h>

// Qt
#include <QDir>
#include <QDirIterator>
#include <QGLWidget>

// STL
#include <vector>
#include <iostream>
#include <fstream>
#include <limits>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

/** 
 * @class oneMesh
 *
 * @brief The oneMesh class provides the interface to manage meshes
 * 
 * This class is the base class for all mesh objects.
 */
struct oneMesh
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/**
	 * ...
	 */
	GLuint VB;//vertex buffer id

	/**
	 * ...
	 */
	GLuint IB;//index buffer id

	/**
	 * ...
	 */
	std::vector< GLfloat > Vertices;

	/**
	 * ...
	 */
	std::vector< GLfloat > Normals;

	/**
	 * ...
	 */
	std::vector< GLfloat > Textures;

	/**
	 * ...
	 */
	std::vector<GLuint> Indices;

	/**
	 * ...
	 */
	GLenum mode;//GL_QUADS OR GL_TRIANGLES		

	/**
	 * ...
	 */
	float ambient[ 4 ];

	/**
	 * ...
	 */
	float diffuse[ 4 ];

	/**
	 * ...
	 */
	float specular[ 4 ];

	/**
	 * ...
	 */
	std::vector< std::string > texFiles[ 3 ];//one for ambient, diffuse, specular 

	/**
	 * ...
	 */
	std::vector< GLuint > texIDs[ 3 ];

	/**
	 * ...
	 */
	bool hasATextures;

	/**
	 * ...
	 */
	bool hasDTextures;

	/**
	 * ...
	 */
	bool hasSTextures;

	/**
	 * ...
	 */
	float shininess;

	/******************************** METHODS *********************************/

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

};

/** 
 * @class Mesh
 *
 * @brief The Mesh class provides the interface to manage a scene as a collection of meshes
 * 
 * This class is the base class for all scene objects.
 */
class Mesh
{

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/

public:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	Mesh( GLuint p = 0 );

	/**
	 * ...
	 */
	void loadTexture( const char* filename, GLuint id );

	/**
	 * ...
	 */
	void InitFromScene( const aiScene* scene );

	/**
	 * ...
	 */
	bool chargerMesh( const std::string& filename ); //loads file

	/**
	 * ...
	 */
	void creerVBO();

	/**
	 * ...
	 */
	void renderMesh( int i );

	/**
	 * ...
	 */
	void render(); //renders scene

	/**
	 * ...
	 */
	std::vector< oneMesh > getMeshes();

	/**
	 * ...
	 */
	int getNumberOfMeshes();

	/**
	 * ...
	 */
	void getAmbient( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getDiffuse( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getSpecular( float tab[ 4 ], int i );

	/**
	 * ...
	 */
	void getShininess( float &s, int i );

	/**
	 * ...
	 */
	void setLightPosition( float x, float y, float z );

	/**
	 * ...
	 */
	bool hasTexture( int i );

	/**
	 * ...
	 */
	float getScaleFactor();

	/**
	 * ...
	 */
	void getTranslationFactors( float translation[ 3 ] );

	/**
	 * ...
	 */
	~Mesh();

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

		/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:

	/****************************** INNER TYPES *******************************/

	/******************************* ATTRIBUTES *******************************/

	/******************************** METHODS *********************************/

	/**
	 * ...
	 */
	float boundingBoxSide;

	/**
	 * ...
	 */
	float center[3];

	/**
	 * ...
	 */
	std::vector< oneMesh > meshes;//all the meshes in the scene

	/**
	 * ...
	 */
	std::string Dir;

	/**
	 * ...
	 */
	GLuint program;

	/**
	 * ...
	 */
	float lightPos[ 3 ];

	/**
	 * ...
	 */
	float boxMin[ 3 ];

	/**
	 * ...
	 */
	float boxMax[ 3 ];

};

/**
 * ...
 */
std::string Directory( const std::string& filename );

/**
 * ...
 */
std::string Filename( const std::string& path );

/**************************************************************************
 ***************************** INLINE SECTION *****************************
 **************************************************************************/

//#include "Mesh.inl"

#endif // !_MESH_H_
