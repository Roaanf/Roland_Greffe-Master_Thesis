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
#include "GvvTransferFunctionEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvPipelineInterface.h"
#include "GvvContextManager.h"
//#include "GvvTransferFunctionInterface.h"

// Qtfe
#include "Qtfe.h"

// System
#include <cassert>

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
using namespace GvViewerCore;
using namespace GvViewerGui;

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ***************************** METHOD DEFINITION ******************************
 ******************************************************************************/

/******************************************************************************
 * Default constructor.
 ******************************************************************************/
GvvTransferFunctionEditor::GvvTransferFunctionEditor( QWidget* pParent, Qt::WindowFlags pFlags )
:	QObject( pParent )
,	GvvContextListener()
,	_editor( NULL )
,	_pipeline( NULL )
{
	// Cache editor
	_editor = new Qtfe();

	// Default input/output initialization
	_editor->addChannels( 4 );
	_editor->addOutputs( 1 );
	_editor->bindChannelToOutputR( 0, 0 );
	_editor->bindChannelToOutputG( 1, 0 );
	_editor->bindChannelToOutputB( 2, 0 );
	_editor->bindChannelToOutputA( 3, 0 );

	// Modify transfer function window flags to always stay on top
	Qt::WindowFlags windowFlags = _editor->windowFlags();
	windowFlags |= Qt::WindowStaysOnTopHint;
#ifndef WIN32
	windowFlags |= Qt::X11BypassWindowManagerHint;
#endif
	_editor->setWindowFlags( windowFlags );

	// Do connection(s)
	QObject::connect( _editor, SIGNAL( functionChanged() ), SLOT( onFunctionChanged() ) );
}

/******************************************************************************
 * Destructor.
 ******************************************************************************/
GvvTransferFunctionEditor::~GvvTransferFunctionEditor()
{
}

/******************************************************************************
 * Set the pipeline.
 *
 * @param pPipeline The pipeline
 ******************************************************************************/
void GvvTransferFunctionEditor::setPipeline( GvViewerCore::GvvPipelineInterface* pPipeline )
{
	//assert( pPipeline != NULL );

	// Pipeline BEGIN
	//assert( mPipeline == NULL );
	_pipeline = pPipeline;
	// Pipeline END
}

/******************************************************************************
 * ...
 ******************************************************************************/
void GvvTransferFunctionEditor::show()
{
	assert( _editor != NULL );
	if ( _editor != NULL )
	{
		_editor->show();
	}
}

///******************************************************************************
// * ...
// *
// * @param pPipeline ...
// * @param pFlag ...
// ******************************************************************************/
//void GvvTransferFunctionEditor::connect( GvViewerCore::GvvPipelineInterface* pPipeline, bool pFlag )
//{
//	// ...
//}

/******************************************************************************
 * Get the transfer function editor.
 *
 * return the transfer function editor
 ******************************************************************************/
Qtfe* GvvTransferFunctionEditor::getTransferFunction()
{
	return _editor;
}

/******************************************************************************
 * Slot called when at least one canal changed
 ******************************************************************************/
void GvvTransferFunctionEditor::onFunctionChanged()
{
	if ( _pipeline != NULL )
	{
		assert( _editor != NULL );
		if ( _editor != NULL )
		{
			float* tab = new float[ 256 * 4 ];
			for ( int i = 0; i < 256 ; ++i )
			{
				float x = i / 256.0f;
				float alpha = _editor->evalf( 3, x );

				tab[ 4 * i + 0 ] = _editor->evalf( 0, x ) * alpha;
				tab[ 4 * i + 1 ] = _editor->evalf( 1, x ) * alpha;
				tab[ 4 * i + 2 ] = _editor->evalf( 2, x ) * alpha;
				tab[ 4 * i + 3 ] = alpha;
			}

			_pipeline->updateTransferFunction( tab, 256 );

			delete[] tab;
		}
	}
}


/******************************************************************************
 * This slot is called when the current editable changed
 ******************************************************************************/
void GvvTransferFunctionEditor::onCurrentBrowsableChanged()
{
	GvvPipelineInterface* currentPipeline = dynamic_cast< GvvPipelineInterface* >( GvvContextManager::get()->editCurrentBrowsable() );
	if ( currentPipeline != NULL )
	{
		if ( currentPipeline->hasTransferFunction() )
		{
			_pipeline = currentPipeline;

			//GvvTransferFunctionInterface* transferFunction = _pipeline->getTransferFunction();
			//if ( transferFunction != NULL )
			//if ( _pipeline->getTransferFunctionFilename() != NULL )
			//{
			//	_editor->load( _pipeline->getTransferFunctionFilename() );
			//}
		}
		else
		{
			_pipeline = NULL;
		}
	}
	else
	{
		_pipeline = NULL;
	}
}
