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

#ifndef _GV_MANDELBROT_SET_CONFIG_H_
#define _GV_MANDELBROT_SET_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVMANDELBROTSET_MAKELIB	// Create a static library.
#		define GVMANDELBROTSET_EXPORT
#		define GVMANDELBROTSET_TEMPLATE_EXPORT
#	elif defined GVMANDELBROTSET_USELIB	// Use a static library.
#		define GVMANDELBROTSET_EXPORT
#		define GVMANDELBROTSET_TEMPLATE_EXPORT

#	elif defined GVMANDELBROTSET_MAKEDLL	// Create a DLL library.
#		define GVMANDELBROTSET_EXPORT	__declspec(dllexport)
#		define GVMANDELBROTSET_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVMANDELBROTSET_EXPORT	__declspec(dllimport)
#		define GVMANDELBROTSET_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVMANDELBROTSET_MAKEDLL) || defined(GVMANDELBROTSET_MAKELIB)
#		define GVMANDELBROTSET_EXPORT
#		define GVMANDELBROTSET_TEMPLATE_EXPORT
#	else
#		define GVMANDELBROTSET_EXPORT
#		define GVMANDELBROTSET_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
