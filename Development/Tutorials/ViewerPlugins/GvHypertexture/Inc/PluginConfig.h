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
#ifndef GVHYPERTEXTURE_CONFIG_H
#define GVHYPERTEXTURE_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVHYPERTEXTURE_MAKELIB	// Create a static library.
#		define GVHYPERTEXTURE_EXPORT
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT
#	elif defined GVHYPERTEXTURE_USELIB	// Use a static library.
#		define GVHYPERTEXTURE_EXPORT
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT

#	elif defined GVHYPERTEXTURE_MAKEDLL	// Create a DLL library.
#		define GVHYPERTEXTURE_EXPORT	__declspec(dllexport)
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVHYPERTEXTURE_EXPORT	__declspec(dllimport)
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVHYPERTEXTURE_MAKEDLL) || defined(GVHYPERTEXTURE_MAKELIB)
#		define GVHYPERTEXTURE_EXPORT
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT
#	else
#		define GVHYPERTEXTURE_EXPORT
#		define GVHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
