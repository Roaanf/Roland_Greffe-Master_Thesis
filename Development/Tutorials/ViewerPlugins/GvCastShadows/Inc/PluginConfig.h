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
#ifndef GVCASTSHADOWS_CONFIG_H
#define GVCASTSHADOWS_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVCASTSHADOWS_MAKELIB	// Create a static library.
#		define GVCASTSHADOWS_EXPORT
#		define GVCASTSHADOWS_TEMPLATE_EXPORT
#	elif defined GVCASTSHADOWS_USELIB	// Use a static library.
#		define GVCASTSHADOWS_EXPORT
#		define GVCASTSHADOWS_TEMPLATE_EXPORT

#	elif defined GVCASTSHADOWS_MAKEDLL	// Create a DLL library.
#		define GVCASTSHADOWS_EXPORT	__declspec(dllexport)
#		define GVCASTSHADOWS_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVCASTSHADOWS_EXPORT	__declspec(dllimport)
#		define GVCASTSHADOWS_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVCASTSHADOWS_MAKEDLL) || defined(GVCASTSHADOWS_MAKELIB)
#		define GVCASTSHADOWS_EXPORT
#		define GVCASTSHADOWS_TEMPLATE_EXPORT
#	else
#		define GVCASTSHADOWS_EXPORT
#		define GVCASTSHADOWS_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
