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
#ifndef GVRENDERERGLSL_CONFIG_H
#define GVRENDERERGLSL_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVRENDERERGLSL_MAKELIB	// Create a static library.
#		define GVRENDERERGLSL_EXPORT
#		define GVRENDERERGLSL_TEMPLATE_EXPORT
#	elif defined GVRENDERERGLSL_USELIB	// Use a static library.
#		define GVRENDERERGLSL_EXPORT
#		define GVRENDERERGLSL_TEMPLATE_EXPORT

#	elif defined GVRENDERERGLSL_MAKEDLL	// Create a DLL library.
#		define GVRENDERERGLSL_EXPORT	__declspec(dllexport)
#		define GVRENDERERGLSL_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVRENDERERGLSL_EXPORT	__declspec(dllimport)
#		define GVRENDERERGLSL_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVRENDERERGLSL_MAKEDLL) || defined(GVRENDERERGLSL_MAKELIB)
#		define GVRENDERERGLSL_EXPORT
#		define GVRENDERERGLSL_TEMPLATE_EXPORT
#	else
#		define GVRENDERERGLSL_EXPORT
#		define GVRENDERERGLSL_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
