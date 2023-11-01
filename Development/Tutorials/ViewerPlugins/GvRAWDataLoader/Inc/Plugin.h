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

#ifndef GVMYPLUGIN_H
#define GVMYPLUGIN_H

#include <GvUtils/GsPluginInterface.h>

namespace GvUtils
{
    class GsPluginManager;
}

class SampleCore;

class GvMyPlugin : public GvUtils::GsPluginInterface
{

  public:

    // Constructors and 
    GvMyPlugin( GvUtils::GsPluginManager& pManager );

	/**
     * Destructor
     */
    virtual ~GvMyPlugin();

    virtual const std::string& getName();
	
  private:

      GvUtils::GsPluginManager& mManager;

      std::string mName;

      std::string mExportName;

	  SampleCore* _pipeline;

	  void initialize();

	  void finalize();

};

#endif  // GVMYPLUGIN_H
