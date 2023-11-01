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

#ifndef _GV_AMPLIFIED_SURFACE_CONFIG_H_
#define _GV_AMPLIFIED_SURFACE_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVAMPLIFIEDSURFACE_MAKELIB	// Create a static library.
#		define GVAMPLIFIEDSURFACE_EXPORT
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT
#	elif defined GVAMPLIFIEDSURFACE_USELIB	// Use a static library.
#		define GVAMPLIFIEDSURFACE_EXPORT
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT

#	elif defined GVAMPLIFIEDSURFACE_MAKEDLL	// Create a DLL library.
#		define GVAMPLIFIEDSURFACE_EXPORT	__declspec(dllexport)
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVAMPLIFIEDSURFACE_EXPORT	__declspec(dllimport)
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVAMPLIFIEDSURFACE_MAKEDLL) || defined(GVAMPLIFIEDSURFACE_MAKELIB)
#		define GVAMPLIFIEDSURFACE_EXPORT
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT
#	else
#		define GVAMPLIFIEDSURFACE_EXPORT
#		define GVAMPLIFIEDSURFACE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
