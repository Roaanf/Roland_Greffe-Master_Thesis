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
#ifndef GVPROXYGEOMETRYMANAGER_CONFIG_H
#define GVPROXYGEOMETRYMANAGER_CONFIG_H

//*** Plugin Library 

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVPROXYGEOMETRYMANAGER_MAKELIB	// Create a static library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	elif defined GVPROXYGEOMETRYMANAGER_USELIB	// Use a static library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT

#	elif defined GVPROXYGEOMETRYMANAGER_MAKEDLL	// Create a DLL library.
#		define GVPROXYGEOMETRYMANAGER_EXPORT	__declspec(dllexport)
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVPROXYGEOMETRYMANAGER_EXPORT	__declspec(dllimport)
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVPROXYGEOMETRYMANAGER_MAKEDLL) || defined(GVPROXYGEOMETRYMANAGER_MAKELIB)
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT
#	else
#		define GVPROXYGEOMETRYMANAGER_EXPORT
#		define GVPROXYGEOMETRYMANAGER_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
