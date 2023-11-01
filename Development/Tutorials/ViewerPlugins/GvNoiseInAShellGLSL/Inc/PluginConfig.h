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
#ifndef GVNOISEINASHELLGLSL_CONFIG_H
#define GVNOISEINASHELLGLSL_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVNOISEINASHELLGLSL_MAKELIB	// Create a static library.
#		define GVNOISEINASHELLGLSL_EXPORT
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT
#	elif defined GVNOISEINASHELLGLSL_USELIB	// Use a static library.
#		define GVNOISEINASHELLGLSL_EXPORT
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT

#	elif defined GVNOISEINASHELLGLSL_MAKEDLL	// Create a DLL library.
#		define GVNOISEINASHELLGLSL_EXPORT	__declspec(dllexport)
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVNOISEINASHELLGLSL_EXPORT	__declspec(dllimport)
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVNOISEINASHELLGLSL_MAKEDLL) || defined(GVNOISEINASHELLGLSL_MAKELIB)
#		define GVNOISEINASHELLGLSL_EXPORT
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT
#	else
#		define GVNOISEINASHELLGLSL_EXPORT
#		define GVNOISEINASHELLGLSL_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
