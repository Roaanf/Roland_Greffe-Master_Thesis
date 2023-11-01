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

#ifndef _GV_COLLISION_BBOX_CONFIG_H_
#define _GV_COLLISION_BBOX_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVCOLLISIONBBOX_MAKELIB	// Create a static library.
#		define GVCOLLISIONBBOX_EXPORT
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT
#	elif defined GVCOLLISIONBBOX_USELIB	// Use a static library.
#		define GVCOLLISIONBBOX_EXPORT
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT

#	elif defined GVCOLLISIONBBOX_MAKEDLL	// Create a DLL library.
#		define GVCOLLISIONBBOX_EXPORT	__declspec(dllexport)
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVCOLLISIONBBOX_EXPORT	__declspec(dllimport)
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVCOLLISIONBBOX_MAKEDLL) || defined(GVCOLLISIONBBOX_MAKELIB)
#		define GVCOLLISIONBBOX_EXPORT
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT
#	else
#		define GVCOLLISIONBBOX_EXPORT
#		define GVCOLLISIONBBOX_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
