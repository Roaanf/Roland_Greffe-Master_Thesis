﻿/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "Mesh.h"
#include <GvCore/GvError.h>
#define BUFFER_OFFSET(a) ((char*)NULL + (a))

//Importer
Assimp::Importer importer;
//Assimp scene object
const aiScene* scene = NULL;

/**
 * Constructor: bounding box set to infinity
 * @param p: program identifier
 */ 
Mesh::Mesh(GLuint p) {
		lightPos[0] = 1;
		lightPos[1] = 1;
		lightPos[2] = 1;
		program = p;
		for (int k = 0; k < 3; k++) {
			boxMin[k] = numeric_limits<float>::max();
			boxMax[k] = numeric_limits<float>::min();
		}
}

/**
 * Loading and binding a texture.
 * @param filename: texture file
 * @param id: texture ID
 */
void Mesh::loadTexture(const char* filename, GLuint id) {
	string f;
	//get the right file name
	QDir d(filename);
	QDirIterator it(QDir(QString(Dir.c_str())), QDirIterator::Subdirectories);
	while (it.hasNext()) {
		it.next();
		QString file = it.fileName();
		if (file == QString(Filename(filename).c_str())) {
			f = it.filePath().toStdString();
		}
	}

	ifstream fin(f.c_str());
	if (!fin.fail()) {
		cout<<"Successfully loaded texture file "<<f<<endl;
		fin.close();
	} else {
		cout << "Couldn't open texture file: "<<f<<"." <<endl;
		return;
	}
	QImage img = QGLWidget::convertToGLFormat(QImage(f.c_str()));
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0,
		GL_RGBA, GL_UNSIGNED_BYTE, img.bits());
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glEnable(GL_TEXTURE_2D);
}

/**
 * Retrieves the directory of a filename.
 * @param filename: file path.
 */
string Directory(const string& filename) {
	size_t pos = filename.find_last_of("\\/");
	return (string::npos == pos) ? "" : filename.substr(0, pos + 1);
}

/**
 * Retrieves the base filename of a file path.
 * @param path: file path.
 */
string Filename(const string& path) {
	return path.substr(path.find_last_of("/\\") + 1);
}

/**
 * Collects all the information about the meshes 
 * (vertices, normals, textures, materials, ...)
 * @param scene: assimp loaded meshes.
 */
void Mesh::InitFromScene(const aiScene* scene) {
	aiColor4D coltemp;
    int materialIndex;
    aiReturn texFound;
    int nbT;
    aiString file;
    float shininess;
	/**Vertices, Normals and Textures**/
	for (int i = 0; i < scene->mNumMeshes; i++) {
		const aiMesh* mesh = scene->mMeshes[i];
		oneMesh M;
		M.hasATextures = false;
		M.hasDTextures = false;
		M.hasSTextures = false;
		//default material values
		M.shininess = 20;
		for (int a = 0; a < 3; a++) {
			M.ambient[a] = 0.2;
			M.diffuse[a] = 0.8;
			M.specular[a] = 0.75;
		}	
		M.ambient[3] = 1.0;
		M.diffuse[3] = 1.0;
		M.specular[3] = 1.0;

		for (int j = 0; j < mesh->mNumVertices; j++) {
			if (mesh->HasPositions()) {
				const aiVector3D* pos = &(mesh->mVertices[j]);
				M.Vertices.push_back(pos->x);
				if (boxMax[0] < pos->x) {
					boxMax[0] = pos->x;
				}
				if (boxMin[0] > pos->x) {
					boxMin[0] = pos->x;
				}
				M.Vertices.push_back(pos->y);
				if (boxMax[1] < pos->y) {
					boxMax[1] = pos->y;
				}
				if (boxMin[1] > pos->y) {
					boxMin[1] = pos->y;
				}
				M.Vertices.push_back(pos->z);
				if (boxMax[2] < pos->z) {
					boxMax[2] = pos->z;
				}
				if (boxMin[2] > pos->z) {
					boxMin[2] = pos->z;
				}
			}
			if (mesh->HasNormals()) {
				const aiVector3D* normal = &(mesh->mNormals[j]);
				M.Normals.push_back(normal->x);
				M.Normals.push_back(normal->y);
				M.Normals.push_back(normal->z);
			}
			if (mesh->HasTextureCoords(0)) {
	        		M.Textures.push_back(mesh->mTextureCoords[0][j].x);
	        		M.Textures.push_back(mesh->mTextureCoords[0][j].y);
	        	}
		}
		/**Indices**/
		for (int k = 0 ; k < mesh->mNumFaces ; k++) {
	        const aiFace& Face = mesh->mFaces[k];
	    		M.mode = GL_TRIANGLES;
	        	M.Indices.push_back(Face.mIndices[0]);
	        	M.Indices.push_back(Face.mIndices[1]);
	        	M.Indices.push_back(Face.mIndices[2]);
	    }
	    /**Materials**/
	    if (scene->HasMaterials()) {
		    materialIndex = mesh->mMaterialIndex;
	        aiMaterial* material = scene->mMaterials[materialIndex];

			nbT = material->GetTextureCount(aiTextureType_AMBIENT);
			if (nbT > 0) {
				M.hasATextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_AMBIENT, j, &file);
				M.texFiles[0].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[0].push_back(id);
				loadTexture(file.data, id);
			}
	        material->Get(AI_MATKEY_COLOR_AMBIENT, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.ambient[0] = coltemp.r;
				M.ambient[1] = coltemp.g;
				M.ambient[2] = coltemp.b;
				M.ambient[3] = coltemp.a;
			}

			nbT = material->GetTextureCount(aiTextureType_DIFFUSE);
			if (nbT > 0) {
				M.hasDTextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_DIFFUSE, j, &file);
				M.texFiles[1].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[1].push_back(id);
				loadTexture(file.data, id);
			}
			material->Get(AI_MATKEY_COLOR_DIFFUSE, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.diffuse[0] = coltemp.r;
				M.diffuse[1] = coltemp.g;
				M.diffuse[2] = coltemp.b;
				M.diffuse[3] = coltemp.a;
			}

			nbT = material->GetTextureCount(aiTextureType_SPECULAR);
			if (nbT > 0) {
				M.hasSTextures = true;
			} 
			for (int j = 0; j < nbT; j++) {
				material->GetTexture(aiTextureType_SPECULAR, j, &file);
				M.texFiles[2].push_back(file.data); 
				GLuint id;
				glGenTextures(1, &id);
				M.texIDs[2].push_back(id);
				loadTexture(file.data, id);
			}
			material->Get(AI_MATKEY_COLOR_SPECULAR, coltemp);
			if (!(coltemp.r ==0 && coltemp.g == 0 && coltemp.b ==0)) {
				M.specular[0] = coltemp.r;
				M.specular[1] = coltemp.g;
				M.specular[2] = coltemp.b;
				M.specular[3] = coltemp.a;
			}
			material->Get(AI_MATKEY_SHININESS, shininess);
			if (shininess != 0.f) {
				M.shininess = shininess;
			}
		}
		meshes.push_back(M);
	}
	//creating the object's bounding box
	boundingBoxSide = max(boxMax[0] - boxMin[0], max(boxMax[1] - boxMin[1], boxMax[2] - boxMin[2]));
	for (int k = 0; k < 3; k++) {
		center[k] = 0.5*(boxMax[k] + boxMin[k]);
		boxMax[k] = center[k] + boundingBoxSide*0.5;
		boxMin[k] = center[k] - boundingBoxSide*0.5;
	}
}

bool Mesh::chargerMesh(const string& filename) {
	//check if file exists
	ifstream fin(filename.c_str());
	if (!fin.fail()) {
		fin.close();
	} else {
		cout << "Couldn't open file." <<endl;
		return false;
	}
	Dir = Directory(filename);
	scene = importer.ReadFile(filename, aiProcessPreset_TargetRealtime_MaxQuality);
	QString s(Dir.c_str());
	QDir d(s);
	Dir = d.absolutePath().toStdString();
	if (!scene) {
		cout << "Import failed." << endl;
		return false;
	}
	InitFromScene(scene);
	cout << "Import scene succeeded.\n" << endl;
	return true;
}

void Mesh::creerVBO() {
	for (int i = 0; i < meshes.size(); i++) {
		glGenBuffers(1, &(meshes[i].VB));
		glGenBuffers(1, &(meshes[i].IB));
		glBindBuffer(GL_ARRAY_BUFFER, meshes[i].VB); 
		glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*meshes[i].Vertices.size() 
									+ sizeof(GLfloat)*meshes[i].Normals.size()
									+ sizeof(GLfloat)*meshes[i].Textures.size(), NULL, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(GLfloat)*meshes[i].Vertices.size(),meshes[i].Vertices.data());
		glBufferSubData(GL_ARRAY_BUFFER,sizeof(GLfloat)*meshes[i].Vertices.size(),sizeof(GLfloat)*meshes[i].Normals.size(),meshes[i].Normals.data());
		glBufferSubData(GL_ARRAY_BUFFER,sizeof(GLfloat)*meshes[i].Vertices.size() 
										+ sizeof(GLfloat)*meshes[i].Normals.size(),sizeof(GLfloat)*meshes[i].Textures.size(),meshes[i].Textures.data());
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshes[i].IB);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*meshes[i].Indices.size(), meshes[i].Indices.data(), GL_STATIC_DRAW);
	}
}

void Mesh::renderMesh(int i) {
	glEnable(GL_TEXTURE_2D);
	/*if (meshes[i].hasATextures) {
		//cout << "hasATextures mesh num"<<i<<endl;
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[0][0]);
	}*/
	if (meshes[i].hasDTextures) {
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[1][0]);
		GV_CHECK_GL_ERROR();
	} 
	/*if (meshes[i].hasSTextures) {
		//cout << "hasSTextures mesh num"<<i<<endl;
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, meshes[i].texIDs[2][0]);
	}*/
	glBindBuffer(GL_ARRAY_BUFFER, meshes[i].VB);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshes[i].IB);
GV_CHECK_GL_ERROR();
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	GV_CHECK_GL_ERROR();
	glVertexPointer(3, GL_FLOAT, 0, 0);
	GV_CHECK_GL_ERROR();
	glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(sizeof(GLfloat)*meshes[i].Vertices.size()));
	GV_CHECK_GL_ERROR();
	glTexCoordPointer(2, GL_FLOAT,0,BUFFER_OFFSET(sizeof(GLfloat)*meshes[i].Vertices.size() + sizeof(GLfloat)*meshes[i].Normals.size()));
	GV_CHECK_GL_ERROR();
	glDrawElements(GL_TRIANGLES, meshes[i].Indices.size(), GL_UNSIGNED_INT,0);
	GV_CHECK_GL_ERROR();
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glBindTexture(GL_TEXTURE_2D, 0); 
	glDisable(GL_TEXTURE_2D);
	GV_CHECK_GL_ERROR();
}

void Mesh::render() {	
	glUseProgram(program);
	glUniform1i(glGetUniformLocation(program, "samplerd"), 2);
	for (int i = 0; i < meshes.size(); i++) {	
		if (hasTexture(i)) {
		glUniform1i(glGetUniformLocation(program, "hasTex"), 1);
		 	GV_CHECK_GL_ERROR();
	} else {
		glUniform1i(glGetUniformLocation(program, "hasTex"), 0);
	}
		glUniform3f(glGetUniformLocation(program, "lightPos"), lightPos[0], lightPos[1], lightPos[2]);
		GV_CHECK_GL_ERROR();
		glUniform4f(glGetUniformLocation(program, "ambientLight"), meshes[i].ambient[0], meshes[i].ambient[1], meshes[i].ambient[2], meshes[i].ambient[3]);
		GV_CHECK_GL_ERROR();
		glUniform4f(glGetUniformLocation(program, "specularColor"), meshes[i].specular[0], meshes[i].specular[1], meshes[i].specular[2], meshes[i].ambient[3]);
		GV_CHECK_GL_ERROR();
		glUniform1f(glGetUniformLocation(program, "shininess"), meshes[i].shininess);		
		GV_CHECK_GL_ERROR();
    	renderMesh(i);	
    	GV_CHECK_GL_ERROR();			
	}
	glUseProgram(0);
	GV_CHECK_GL_ERROR();
}

vector<oneMesh> Mesh::getMeshes() {
	return meshes;
}

int Mesh::getNumberOfMeshes() {
	return meshes.size();
}

void Mesh::getAmbient(float tab[4], int i) {
	tab[0] = meshes[i].ambient[0];
	tab[1] = meshes[i].ambient[1];
	tab[2] = meshes[i].ambient[2];
	tab[3] = meshes[i].ambient[3];
}

void Mesh::getDiffuse(float tab[4], int i) {
	tab[0] = meshes[i].diffuse[0];
	tab[1] = meshes[i].diffuse[1];
	tab[2] = meshes[i].diffuse[2];
	tab[3] = meshes[i].diffuse[3];	
}

void Mesh::getSpecular(float tab[4], int i) {
	tab[0] = meshes[i].specular[0];
	tab[1] = meshes[i].specular[1];
	tab[2] = meshes[i].specular[2];
	tab[3] = meshes[i].specular[3];
}

void Mesh::getShininess(float &s, int i) {
	s = meshes[i].shininess;
}

bool Mesh::hasTexture(int i) {
	return (meshes[i].hasATextures || meshes[i].hasDTextures || meshes[i].hasSTextures);
}

void Mesh::setLightPosition(float x, float y, float z) {
	lightPos[0] = x;
	lightPos[1] = y;
	lightPos[2] = z;
}

float Mesh::getScaleFactor() {
	return 1.05*boundingBoxSide;
}

void Mesh::getTranslationFactors(float translation[3]) {
	for (int k = 0; k < 3; k++) {
		translation[k] = center[k];
	}
}

Mesh::~Mesh() {}
