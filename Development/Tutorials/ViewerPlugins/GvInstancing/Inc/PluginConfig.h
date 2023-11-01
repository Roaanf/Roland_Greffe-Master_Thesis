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
#ifndef GVINSTANCING_CONFIG_H
#define GVINSTANCING_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVINSTANCING_MAKELIB	// Create a static library.
#		define GVINSTANCING_EXPORT
#		define GVINSTANCING_TEMPLATE_EXPORT
#	elif defined GVINSTANCING_USELIB	// Use a static library.
#		define GVINSTANCING_EXPORT
#		define GVINSTANCING_TEMPLATE_EXPORT

#	elif defined GVINSTANCING_MAKEDLL	// Create a DLL library.
#		define GVINSTANCING_EXPORT	__declspec(dllexport)
#		define GVINSTANCING_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVINSTANCING_EXPORT	__declspec(dllimport)
#		define GVINSTANCING_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVINSTANCING_MAKEDLL) || defined(GVINSTANCING_MAKELIB)
#		define GVINSTANCING_EXPORT
#		define GVINSTANCING_TEMPLATE_EXPORT
#	else
#		define GVINSTANCING_EXPORT
#		define GVINSTANCING_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
