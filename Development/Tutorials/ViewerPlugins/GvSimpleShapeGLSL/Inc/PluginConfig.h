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
#ifndef GVSIMPLESHAPEGLSL_CONFIG_H
#define GVSIMPLESHAPEGLSL_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSIMPLESHAPEGLSL_MAKELIB	// Create a static library.
#		define GVSIMPLESHAPEGLSL_EXPORT
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT
#	elif defined GVSIMPLESHAPEGLSL_USELIB	// Use a static library.
#		define GVSIMPLESHAPEGLSL_EXPORT
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT

#	elif defined GVSIMPLESHAPEGLSL_MAKEDLL	// Create a DLL library.
#		define GVSIMPLESHAPEGLSL_EXPORT	__declspec(dllexport)
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSIMPLESHAPEGLSL_EXPORT	__declspec(dllimport)
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSIMPLESHAPEGLSL_MAKEDLL) || defined(GVSIMPLESHAPEGLSL_MAKELIB)
#		define GVSIMPLESHAPEGLSL_EXPORT
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT
#	else
#		define GVSIMPLESHAPEGLSL_EXPORT
#		define GVSIMPLESHAPEGLSL_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
