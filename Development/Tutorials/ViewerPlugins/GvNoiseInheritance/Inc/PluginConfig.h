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
#ifndef GVNOISEINHERITANCE_CONFIG_H
#define GVNOISEINHERITANCE_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVNOISEINHERITANCE_MAKELIB	// Create a static library.
#		define GVNOISEINHERITANCE_EXPORT
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT
#	elif defined GVNOISEINHERITANCE_USELIB	// Use a static library.
#		define GVNOISEINHERITANCE_EXPORT
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT

#	elif defined GVNOISEINHERITANCE_MAKEDLL	// Create a DLL library.
#		define GVNOISEINHERITANCE_EXPORT	__declspec(dllexport)
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVNOISEINHERITANCE_EXPORT	__declspec(dllimport)
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVNOISEINHERITANCE_MAKEDLL) || defined(GVNOISEINHERITANCE_MAKELIB)
#		define GVNOISEINHERITANCE_EXPORT
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT
#	else
#		define GVNOISEINHERITANCE_EXPORT
#		define GVNOISEINHERITANCE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
