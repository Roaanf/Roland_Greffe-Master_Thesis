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
#ifndef _GVANIMATEDSNAKE_CONFIG_H_
#define _GVANIMATEDSNAKE_CONFIG_H_

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVANIMATEDSNAKE_MAKELIB	// Create a static library.
#		define GVANIMATEDSNAKE_EXPORT
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT
#	elif defined GVANIMATEDSNAKE_USELIB	// Use a static library.
#		define GVANIMATEDSNAKE_EXPORT
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT

#	elif defined GVANIMATEDSNAKE_MAKEDLL	// Create a DLL library.
#		define GVANIMATEDSNAKE_EXPORT	__declspec(dllexport)
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVANIMATEDSNAKE_EXPORT	__declspec(dllimport)
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVANIMATEDSNAKE_MAKEDLL) || defined(GVANIMATEDSNAKE_MAKELIB)
#		define GVANIMATEDSNAKE_EXPORT
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT
#	else
#		define GVANIMATEDSNAKE_EXPORT
#		define GVANIMATEDSNAKE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
