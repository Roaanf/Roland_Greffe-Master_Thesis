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

#include "Skybox.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// QGLViewer
#include <QGLViewer/qglviewer.h>

// Qt
#include <QCoreApplication>
#include <QString>
#include <QDir>
#include <QFileInfo>
#include <QImage>

// STL
#include <iostream>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// STL
using namespace std;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/


/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Constructor
 ******************************************************************************/
Skybox::Skybox()
{
}

/******************************************************************************
 * Initialization
 ******************************************************************************/
void Skybox::init()
{
	glEnable( GL_TEXTURE_2D );

	/* Taille de la skybox */
	size = 100;

	// Data repository
	QString dataRepository = QCoreApplication::applicationDirPath() + QDir::separator() + QString( "Data" );

	// Load the proxy geometry
	//
	// @todo : check for file availability
	QString model3D = dataRepository + QDir::separator() + QString( "3DModels" ) + QDir::separator() + QString( "bunny.obj" );
	QString _filename = model3D.toLatin1().constData();

	QString SkyboxFacesFilename;
	QString fragmentShaderFilename;
	// Depth Peeling's initialization shader program
	SkyboxFacesFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_init_vertex.glsl" );
	fragmentShaderFilename = dataRepository + QDir::separator() + QString( "Shaders" ) + QDir::separator() + QString( "GvDepthPeeling" ) + QDir::separator() + QString( "front_peeling_init_fragment.glsl" );
	// Depth Peeling's core shader program
	QString leftFilename = dataRepository + QDir::separator() + QString( "skyboxes" ) + QDir::separator() +  QString( "left.jpg" );
	QString rightFilename = dataRepository + QDir::separator() + QString( "skyboxes" ) + QDir::separator() +  QString( "right.jpg" );
	QString topFilename = dataRepository + QDir::separator() + QString( "skyboxes" ) + QDir::separator() +  QString( "top.jpg" );
	QString backFilename = dataRepository + QDir::separator() + QString( "skyboxes" ) + QDir::separator() +  QString( "back.jpg" );
	QString frontFilename = dataRepository + QDir::separator() + QString( "skyboxes" ) + QDir::separator() +  QString( "front.jpg" );

	std::cout << leftFilename.toLatin1().constData() << std::endl;

	/* Texture de la skybox */
	CreateGLTexture(SKY_LEFT, leftFilename.toLatin1().constData());
	CreateGLTexture(SKY_BACK, backFilename.toLatin1().constData());
	CreateGLTexture(SKY_RIGHT, rightFilename.toLatin1().constData());
	CreateGLTexture(SKY_FRONT, frontFilename.toLatin1().constData());
	CreateGLTexture(SKY_TOP, topFilename.toLatin1().constData());
	//CreateGLTexture(SKY_BOTTOM,"../data/skyboxes17/bottom.jpg");

	glDisable( GL_TEXTURE_2D );
}

/******************************************************************************
 * Draw
 ******************************************************************************/
void Skybox::draw()
{
	glMatrixMode( GL_MODELVIEW );

	/* Affichage de la skybox*/
	glPushMatrix();
	glTranslatef(0, 0, size/2);
	this->drawSkybox();
	glPopMatrix();
}

/******************************************************************************
 * Fonction d'affichage de la skybox
 ******************************************************************************/
void Skybox::drawSkybox()
{
	// Activation des textures
	glEnable( GL_TEXTURE_2D );

	// Pas de teinte
	glColor3ub(255, 255, 255);

	// Sélection de la texture    
	glBindTexture(GL_TEXTURE_2D, skybox[SKY_BACK]);
	glBegin(GL_QUADS);
	glTexCoord2d(1, 0); glVertex3d(+size, -size, -size);
	glTexCoord2d(1, 1); glVertex3d(+size, -size, +size);
	glTexCoord2d(0, 1); glVertex3d(-size, -size, +size);
	glTexCoord2d(0, 0); glVertex3d(-size, -size, -size);
	glEnd();

	// glBindTexture(GL_TEXTURE_2D, skybox[SKY_BOTTOM]);
	// glBegin(GL_QUADS);
	// glTexCoord2d(0, 0); glVertex3d(+size, -size, -size);
	// glTexCoord2d(0, 1); glVertex3d(+size, +size, -size);
	// glTexCoord2d(1, 1); glVertex3d(-size, +size, -size);
	// glTexCoord2d(1, 0); glVertex3d(-size, -size, -size);   
	// glEnd();

	glBindTexture(GL_TEXTURE_2D, skybox[SKY_LEFT]);
	glBegin(GL_QUADS);
	glTexCoord2d(1, 0); glVertex3d(+size, +size, -size);
	glTexCoord2d(1, 1); glVertex3d(+size, +size, +size);
	glTexCoord2d(0, 1); glVertex3d(+size, -size, +size);
	glTexCoord2d(0, 0); glVertex3d(+size, -size, -size);
	glEnd();

	// glBindTexture(GL_TEXTURE_2D, skybox[SKY_FRONT]);
	// glBegin(GL_QUADS);
	// glTexCoord2d(1, 0); glVertex3d(-size, +size, -size);
	// glTexCoord2d(1, 1); glVertex3d(-size, +size, +size);
	// glTexCoord2d(0, 1); glVertex3d(+size, +size, +size);
	// glTexCoord2d(0, 0); glVertex3d(+size, +size, -size);
	// glEnd();

	glBindTexture(GL_TEXTURE_2D, skybox[SKY_RIGHT]);
	glBegin(GL_QUADS);
	glTexCoord2d(1, 0); glVertex3d(-size, -size, -size);
	glTexCoord2d(1, 1); glVertex3d(-size, -size, +size);
	glTexCoord2d(0, 1); glVertex3d(-size, +size, +size);
	glTexCoord2d(0, 0); glVertex3d(-size, +size, -size);
	glEnd();

	glBindTexture(GL_TEXTURE_2D, skybox[SKY_TOP]);
	glBegin(GL_QUADS);
	glTexCoord2d(0, 1); glVertex3d(+size, -size, +size);
	glTexCoord2d(0, 0); glVertex3d(+size, +size, +size);
	glTexCoord2d(1, 0); glVertex3d(-size, +size, +size);
	glTexCoord2d(1, 1); glVertex3d(-size, -size, +size);
	glEnd();
	
	glDisable( GL_TEXTURE_2D );
}

/******************************************************************************
 * Fonction de creation de la Texture
 *
 * @param pTexId ...
 * @param pFilename ...
 ******************************************************************************/
void Skybox::CreateGLTexture( SkyId pTexId, const char* pFilename )
{
	// generates an OpenGL texture id, and store it in the map
	GLuint id;
	glGenTextures( 1, &id );
	skybox[ pTexId ] = id;

	// load a texture file as a QImage
	QImage img = QGLWidget::convertToGLFormat( QImage( pFilename ) );

	// specify the texture (2D texture, rgba, single file)
	glBindTexture( GL_TEXTURE_2D, skybox[ pTexId ] );
	glTexImage2D ( GL_TEXTURE_2D, 0, GL_RGBA, img.width(), img.height(), 0,	GL_RGBA, GL_UNSIGNED_BYTE, img.bits() );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
}
