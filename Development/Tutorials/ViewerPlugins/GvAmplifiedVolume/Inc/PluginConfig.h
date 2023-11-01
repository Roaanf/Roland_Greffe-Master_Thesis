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

/**
 * ...
 */
#ifndef GVAMPLIFIEDVOLUME_CONFIG_H
#define GVAMPLIFIEDVOLUME_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVAMPLIFIEDVOLUME_MAKELIB	// Create a static library.
#		define GVAMPLIFIEDVOLUME_EXPORT
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT
#	elif defined GVAMPLIFIEDVOLUME_USELIB	// Use a static library.
#		define GVAMPLIFIEDVOLUME_EXPORT
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT

#	elif defined GVAMPLIFIEDVOLUME_MAKEDLL	// Create a DLL library.
#		define GVAMPLIFIEDVOLUME_EXPORT	__declspec(dllexport)
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVAMPLIFIEDVOLUME_EXPORT	__declspec(dllimport)
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVAMPLIFIEDVOLUME_MAKEDLL) || defined(GVAMPLIFIEDVOLUME_MAKELIB)
#		define GVAMPLIFIEDVOLUME_EXPORT
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT
#	else
#		define GVAMPLIFIEDVOLUME_EXPORT
#		define GVAMPLIFIEDVOLUME_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
