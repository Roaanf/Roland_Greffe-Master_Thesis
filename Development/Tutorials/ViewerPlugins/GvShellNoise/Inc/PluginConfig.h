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

#ifndef _GV_SHELLNOISE_CONFIG_H_
#define _GV_SHELLNOISE_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSHELLNOISE_MAKELIB	// Create a static library.
#		define GVSHELLNOISE_EXPORT
#		define GVSHELLNOISE_TEMPLATE_EXPORT
#	elif defined GVSHELLNOISE_USELIB	// Use a static library.
#		define GVSHELLNOISE_EXPORT
#		define GVSHELLNOISE_TEMPLATE_EXPORT

#	elif defined GVSHELLNOISE_MAKEDLL	// Create a DLL library.
#		define GVSHELLNOISE_EXPORT	__declspec(dllexport)
#		define GVSHELLNOISE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSHELLNOISE_EXPORT	__declspec(dllimport)
#		define GVSHELLNOISE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSHELLNOISE_MAKEDLL) || defined(GVSHELLNOISE_MAKELIB)
#		define GVSHELLNOISE_EXPORT
#		define GVSHELLNOISE_TEMPLATE_EXPORT
#	else
#		define GVSHELLNOISE_EXPORT
#		define GVSHELLNOISE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
