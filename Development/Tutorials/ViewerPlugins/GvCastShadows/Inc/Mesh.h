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
	 * Vertex buffer
	 */
	GLuint _vertexBuffer;

	/**
	 * Index buffer
	 */
	GLuint _indexBuffer;

	/**
	 * Vertices
	 */
	std::vector< GLfloat > _vertices;

	/**
	 * Normals
	 */
	std::vector< GLfloat > _normals;

	/**
	 * Texture coordinates
	 */
	std::vector< GLfloat > _texCoords;

	/**
	 * Indices
	 */
	std::vector< GLuint > _indices;

	/**
	 * Primitive type (GL_QUADS or GL_TRIANGLES)
	 */
	GLenum mode;	

	/**
	 * Texture filenames
	 * - one for ambient, diffuse, specular 
	 */
	std::vector< std::string > texFiles[ 3 ];

	/**
	 * Texture IDs
	 * - one for ambient, diffuse, specular 
	 */
	std::vector< GLuint > texIDs[ 3 ];

	/**
	 * Material's ambient term
	 */
	float _ambient[ 4 ];

	/**
	* Material's diffuse term
	 */
	float _diffuse[ 4 ];

	/**
	 * Material's specular term
	 */
	float _specular[ 4 ];

	/**
	 * Material's specular shininess term
	 */
	float _shininess;

	/**
	 * ...
	 */
	bool _hasAmbientTextures;

	/**
	 * ...
	 */
	bool _hasDiffuseTextures;

	/**
	 * ...
	 */
	bool _hasSpecularTextures;

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

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

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
	 * Constructor
	 */
	Mesh( GLuint p = 0 );

	/**
	 * Destructor
	 */
	virtual ~Mesh();

	/**
	 * ...
	 */
	void loadTexture( const char* filename, GLuint id );

	/**
	 * Collects all the information about the meshes 
	 * (vertices, normals, textures, materials, ...)
	 *
	 * @param scene: assimp loaded meshes.
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
	 * Render part of mesh
	 *
	 * @pram pIndex part of mesh
	 */
	void renderMesh( int pIndex );

	/**
	 * Render scene (i.e. all meshes)
	 */
	void render();

	/**
	 * Get the list of meshes
	 *
	 * @return the list of meshes
	 */
	const std::vector< oneMesh >& getMeshes() const;

	/**
	 * Get the number of meshes
	 *
	 * @return the number of meshes
	 */
	int getNbMeshes();

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
	 * Get the 3D model filename to load
	 *
	 * @return the 3D model filename to load
	 */
	const char* get3DModelFilename() const;

	/**
	 * Set the 3D model filename to load
	 *
	 * @param pFilename the 3D model filename to load
	 */
	void set3DModelFilename( const char* pFilename );

	/**
	 * Get the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	void getTranslation( float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the translation
	 *
	 * @param pX the translation on x axis
	 * @param pY the translation on y axis
	 * @param pZ the translation on z axis
	 */
	void setTranslation( float pX, float pY, float pZ );

	/**
	 * Get the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	void getRotation( float& pAngle, float& pX, float& pY, float& pZ ) const;

	/**
	 * Set the rotation
	 *
	 * @param pAngle the rotation angle (in degree)
	 * @param pX the x component of the rotation vector
	 * @param pY the y component of the rotation vector
	 * @param pZ the z component of the rotation vector
	 */
	void setRotation( float pAngle, float pX, float pY, float pZ );

	/**
	 * Get the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	void getScale( float& pValue ) const;

	/**
	 * Set the uniform scale
	 *
	 * @param pValue the uniform scale
	 */
	void setScale( float pValue );

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
	float center[ 3 ];

	/**
	 * ...
	 */
	std::vector< oneMesh > _meshes;//all the meshes in the scene

	/**
	 * ...
	 */
	std::string _repository;

	/**
	 * ...
	 */
	GLuint _shaderProgram;

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

	/**
	 * 3D model filename (mesh or scene)
	 */
	std::string _modelFilename;

	/**
	 * Translation used to position the GigaVoxels data structure
	 */
	float _translation[ 3 ];

	/**
	 * Rotation used to position the GigaVoxels data structure
	 */
	float _rotation[ 4 ];

	/**
	 * Scale used to transform the GigaVoxels data structure
	 */
	float _scale;

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
