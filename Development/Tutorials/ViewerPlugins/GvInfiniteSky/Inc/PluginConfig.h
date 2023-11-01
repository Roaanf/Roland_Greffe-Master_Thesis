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
#ifndef _GVINFINITESKY_CONFIG_H_
#define _GVINFINITESKY_CONFIG_H_

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVINFINITESKY_MAKELIB	// Create a static library.
#		define GVINFINITESKY_EXPORT
#		define GVINFINITESKY_TEMPLATE_EXPORT
#	elif defined GVINFINITESKY_USELIB	// Use a static library.
#		define GVINFINITESKY_EXPORT
#		define GVINFINITESKY_TEMPLATE_EXPORT

#	elif defined GVINFINITESKY_MAKEDLL	// Create a DLL library.
#		define GVINFINITESKY_EXPORT	__declspec(dllexport)
#		define GVINFINITESKY_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVINFINITESKY_EXPORT	__declspec(dllimport)
#		define GVINFINITESKY_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVINFINITESKY_MAKEDLL) || defined(GVINFINITESKY_MAKELIB)
#		define GVINFINITESKY_EXPORT
#		define GVINFINITESKY_TEMPLATE_EXPORT
#	else
#		define GVINFINITESKY_EXPORT
#		define GVINFINITESKY_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
