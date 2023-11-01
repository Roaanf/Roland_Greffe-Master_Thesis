/*
 * GigaVoxels - GigaSpace
 *
 * Website: http://gigavoxels.inrialpes.fr/
 *
 * Contributors: GigaVoxels Team
 *
 * BSD 3-Clause License:
 *
 * Copyright (C) 2007-2015 INRIA - LJK (CNRS - Grenoble University), All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the organization nor the names  of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDER BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//
//#include "GvvPipelineManager.h"
//
///******************************************************************************
// ******************************* INCLUDE SECTION ******************************
// ******************************************************************************/
//
//// GvViewer
//#include "GvvPipelineInterface.h"
//#include "GvvPipelineManagerListener.h"
//
//// System
//#include <cassert>
//#include <cstdio>
//#include <cstdlib>
//#include <iostream>
//#include <cstring>
//
//// STL
//#include <algorithm>
//
///******************************************************************************
// ****************************** NAMESPACE SECTION *****************************
// ******************************************************************************/
//
//// GvViewer
//using namespace GvViewerCore;
//
//// STL
//using namespace std;
//
///******************************************************************************
// ************************* DEFINE AND CONSTANT SECTION ************************
// ******************************************************************************/
//
///**
// * The unique instance of the singleton.
// */
//GvvPipelineManager* GvvPipelineManager::msInstance = NULL;
//
///******************************************************************************
// ***************************** TYPE DEFINITION ********************************
// ******************************************************************************/
//
///******************************************************************************
// ***************************** METHOD DEFINITION ******************************
// ******************************************************************************/
//
///******************************************************************************
// * Get the unique instance.
// *
// * @return the unique instance
// ******************************************************************************/
//GvvPipelineManager& GvvPipelineManager::get()
//{
//    if ( msInstance == NULL )
//    {
//        msInstance = new GvvPipelineManager();
//    }
//
//    return *msInstance;
//}
//
///******************************************************************************
// * Constructor
// ******************************************************************************/
//GvvPipelineManager::GvvPipelineManager()
//:	mPipelines()
//,	mListeners()
//{
//}
//
///******************************************************************************
// * Add a pipeline.
// *
// * @param the pipeline to add
// ******************************************************************************/
//void GvvPipelineManager::addPipeline( GvvPipelineInterface* pPipeline )
//{
//	assert( pPipeline != NULL );
//	if ( pPipeline != NULL )
//	{
//		// Add pipeline
//		mPipelines.push_back( pPipeline );
//
//		// Inform listeners that a pipeline has been added
//		vector< GvvPipelineManagerListener* >::iterator it = mListeners.begin();
//		for ( ; it != mListeners.end(); ++it )
//		{
//			GvvPipelineManagerListener* listener = *it;
//			if ( listener != NULL )
//			{
//				listener->onPipelineAdded( pPipeline );
//			}
//		}
//	}
//}
//
///******************************************************************************
// * Remove a pipeline.
// *
// * @param the pipeline to remove
// ******************************************************************************/
//void GvvPipelineManager::removePipeline( GvvPipelineInterface* pPipeline )
//{
//	assert( pPipeline != NULL );
//	if ( pPipeline != NULL )
//	{
//		vector< GvvPipelineInterface * >::iterator itPipeline;
//		itPipeline = find( mPipelines.begin(), mPipelines.end(), pPipeline );
//		if ( itPipeline != mPipelines.end() )
//		{
//			// Remove pipeline
//			mPipelines.erase( itPipeline );
//
//			// Inform listeners that a pipeline has been removed
//			vector< GvvPipelineManagerListener* >::iterator itListener = mListeners.begin();
//			for ( ; itListener != mListeners.end(); ++itListener )
//			{
//				GvvPipelineManagerListener* listener = *itListener;
//				if ( listener != NULL )
//				{
//					listener->onPipelineAdded( pPipeline );
//				}
//			}
//		}
//	}
//}
//
///******************************************************************************
// * Register a listener.
// *
// * @param pListener the listener to register
// ******************************************************************************/
//void GvvPipelineManager::registerListener( GvvPipelineManagerListener* pListener )
//{
//	assert( pListener != NULL );
//	if ( pListener != NULL )
//	{
//		// Add listener
//		mListeners.push_back( pListener );
//	}
//}
//
///******************************************************************************
// * Unregister a listener.
// *
// * @param pListener the listener to unregister
// ******************************************************************************/
//void GvvPipelineManager::unregisterListener( GvvPipelineManagerListener* pListener )
//{
//	assert( pListener != NULL );
//	if ( pListener != NULL )
//	{
//		vector< GvvPipelineManagerListener * >::iterator it;
//		it = find( mListeners.begin(), mListeners.end(), pListener );
//		if ( it != mListeners.end() )
//		{
//			// Remove pipeline
//			mListeners.erase( it );
//		}
//	}
//}
