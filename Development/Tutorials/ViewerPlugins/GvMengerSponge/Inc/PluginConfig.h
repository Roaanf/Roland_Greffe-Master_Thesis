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
#ifndef GVMENGERSPONGE_CONFIG_H
#define GVMENGERSPONGE_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVMENGERSPONGE_MAKELIB	// Create a static library.
#		define GVMENGERSPONGE_EXPORT
#		define GVMENGERSPONGE_TEMPLATE_EXPORT
#	elif defined GVMENGERSPONGE_USELIB	// Use a static library.
#		define GVMENGERSPONGE_EXPORT
#		define GVMENGERSPONGE_TEMPLATE_EXPORT

#	elif defined GVMENGERSPONGE_MAKEDLL	// Create a DLL library.
#		define GVMENGERSPONGE_EXPORT	__declspec(dllexport)
#		define GVMENGERSPONGE_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVMENGERSPONGE_EXPORT	__declspec(dllimport)
#		define GVMENGERSPONGE_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVMENGERSPONGE_MAKEDLL) || defined(GVMENGERSPONGE_MAKELIB)
#		define GVMENGERSPONGE_EXPORT
#		define GVMENGERSPONGE_TEMPLATE_EXPORT
#	else
#		define GVMENGERSPONGE_EXPORT
#		define GVMENGERSPONGE_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
