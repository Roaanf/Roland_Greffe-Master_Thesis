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
#ifndef GVVBOGENERATOR_CONFIG_H
#define GVVBOGENERATOR_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVVBOGENERATOR_MAKELIB	// Create a static library.
#		define GVVBOGENERATOR_EXPORT
#		define GVVBOGENERATOR_TEMPLATE_EXPORT
#	elif defined GVVBOGENERATOR_USELIB	// Use a static library.
#		define GVVBOGENERATOR_EXPORT
#		define GVVBOGENERATOR_TEMPLATE_EXPORT

#	elif defined GVVBOGENERATOR_MAKEDLL	// Create a DLL library.
#		define GVVBOGENERATOR_EXPORT	__declspec(dllexport)
#		define GVVBOGENERATOR_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVVBOGENERATOR_EXPORT	__declspec(dllimport)
#		define GVVBOGENERATOR_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVVBOGENERATOR_MAKEDLL) || defined(GVVBOGENERATOR_MAKELIB)
#		define GVVBOGENERATOR_EXPORT
#		define GVVBOGENERATOR_TEMPLATE_EXPORT
#	else
#		define GVVBOGENERATOR_EXPORT
#		define GVVBOGENERATOR_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
