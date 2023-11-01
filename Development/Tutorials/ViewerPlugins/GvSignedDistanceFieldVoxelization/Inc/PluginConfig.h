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

#ifndef _GV_SIGNED_DISTANCE_FIELD_VOXELIZATION_CONFIG_H_
#define _GV_SIGNED_DISTANCE_FIELD_VOXELIZATION_CONFIG_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/**
 * Plugin Library 
 */

// Static or dynamic link configuration
#ifdef WIN32
#	ifdef GVSIGNEDDISTANCEFIELDVOXELIZATION_MAKELIB	// Create a static library.
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT
#	elif defined GVSIGNEDDISTANCEFIELDVOXELIZATION_USELIB	// Use a static library.
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT

#	elif defined GVSIGNEDDISTANCEFIELDVOXELIZATION_MAKEDLL	// Create a DLL library.
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT	__declspec(dllexport)
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT
#	else	// Use DLL library
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT	__declspec(dllimport)
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT	extern
#	endif
#else
#	 if defined(GVSIGNEDDISTANCEFIELDVOXELIZATION_MAKEDLL) || defined(GVSIGNEDDISTANCEFIELDVOXELIZATION_MAKELIB)
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT
#	else
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_EXPORT
#		define GVSIGNEDDISTANCEFIELDVOXELIZATION_TEMPLATE_EXPORT	extern
#	endif
#endif

#endif
