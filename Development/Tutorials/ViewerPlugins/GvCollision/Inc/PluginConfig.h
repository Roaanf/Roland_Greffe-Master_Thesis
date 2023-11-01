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

#ifndef _GV_COLLISION_CONFIG_H_
#define _GV_COLLISION_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVCOLLISION_MAKELIB	// Create a static library.
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT
#	elif defined GVCOLLISION_USELIB	// Use a static library.
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT

#	elif defined GVCOLLISION_MAKEDLL	// Create a DLL library.
#		define GVCOLLISION_EXPORT	__declspec(dllexport)
#		define GVCOLLISION_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVCOLLISION_EXPORT	__declspec(dllimport)
#		define GVCOLLISION_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVCOLLISION_MAKEDLL) || defined(GVCOLLISION_MAKELIB)
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT
#	else
#		define GVCOLLISION_EXPORT
#		define GVCOLLISION_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
