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
#ifndef GVLAZYHYPERTEXTURE_CONFIG_H
#define GVLAZYHYPERTEXTURE_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVLAZYHYPERTEXTURE_MAKELIB	// Create a static library.
#		define GVLAZYHYPERTEXTURE_EXPORT
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT
#	elif defined GVLAZYHYPERTEXTURE_USELIB	// Use a static library.
#		define GVLAZYHYPERTEXTURE_EXPORT
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT

#	elif defined GVLAZYHYPERTEXTURE_MAKEDLL	// Create a DLL library.
#		define GVLAZYHYPERTEXTURE_EXPORT	__declspec(dllexport)
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVLAZYHYPERTEXTURE_EXPORT	__declspec(dllimport)
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVLAZYHYPERTEXTURE_MAKEDLL) || defined(GVLAZYHYPERTEXTURE_MAKELIB)
#		define GVLAZYHYPERTEXTURE_EXPORT
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT
#	else
#		define GVLAZYHYPERTEXTURE_EXPORT
#		define GVLAZYHYPERTEXTURE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
