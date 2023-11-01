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
#ifndef GVSLISESIX_CONFIG_H
#define GVSLISESIX_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSLISESIX_MAKELIB	// Create a static library.
#		define GVSLISESIX_EXPORT
#		define GVSLISESIX_TEMPLATE_EXPORT
#	elif defined GVSLISESIX_USELIB	// Use a static library.
#		define GVSLISESIX_EXPORT
#		define GVSLISESIX_TEMPLATE_EXPORT

#	elif defined GVSLISESIX_MAKEDLL	// Create a DLL library.
#		define GVSLISESIX_EXPORT	__declspec(dllexport)
#		define GVSLISESIX_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSLISESIX_EXPORT	__declspec(dllimport)
#		define GVSLISESIX_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSLISESIX_MAKEDLL) || defined(GVSLISESIX_MAKELIB)
#		define GVSLISESIX_EXPORT
#		define GVSLISESIX_TEMPLATE_EXPORT
#	else
#		define GVSLISESIX_EXPORT
#		define GVSLISESIX_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
