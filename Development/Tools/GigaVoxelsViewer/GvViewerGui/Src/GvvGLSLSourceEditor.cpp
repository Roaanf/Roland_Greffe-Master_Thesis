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

#include "GvvGLSLSourceEditor.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// Viewer
#include "GvvGLSLSyntaxHighlighter.h"
#include "GvvPipelineInterface.h"

// System
#include <cassert>

// Qt
#include <QFile>
#include <QTextStream>

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
 * Default constructor
 ******************************************************************************/
GvvGLSLSourceEditor::GvvGLSLSourceEditor( QWidget* pWidget )
:	QWidget( 0, 0 )
,	_pipeline( NULL )
{
	_ui.setupUi( this );

	// TO DO
	// - store these syntax highlighter to be able to delete them
	GvvGLSLSyntaxHighlighter* lvertexHighLight = new GvvGLSLSyntaxHighlighter( _ui._vertexShaderTextEdit->document() );
	GvvGLSLSyntaxHighlighter* ltesselationControlHighLight = new GvvGLSLSyntaxHighlighter( _ui._tesselationControlShaderTextEdit->document() );
	GvvGLSLSyntaxHighlighter* ltesselationEvaluationHighLight = new GvvGLSLSyntaxHighlighter( _ui._tesselationEvaluationShaderTextEdit->document() );
	GvvGLSLSyntaxHighlighter* lgeometryHighLight = new GvvGLSLSyntaxHighlighter( _ui._geometryShaderTextEdit->document() );
	GvvGLSLSyntaxHighlighter* lfragmentHighLight = new GvvGLSLSyntaxHighlighter( _ui._fragmentShaderTextEdit->document() );
	GvvGLSLSyntaxHighlighter* lcomputeHighLight = new GvvGLSLSyntaxHighlighter( _ui._computeShaderTextEdit->document() );

	//connect( _ui._applyButton, SIGNAL( clicked() ), this, SLOT( onApply() ) );
	//connect( _ui._reloadButton, SIGNAL( clicked() ), this, SLOT( onReload() ) );

	setAcceptDrops( true );

	//_ui.tabWidget->setTabEnabled( 0, true );
	//_ui.tabWidget->setTabEnabled( 1, false );
	//_ui.tabWidget->setTabEnabled( 2, false );
	//_ui.tabWidget->setTabEnabled( 3, false );
	//_ui.tabWidget->setTabEnabled( 4, true );
	//_ui.tabWidget->setTabEnabled( 5, false );

	// If you want to create a source code editor using QTextEdit, you should first assign a fixed-width (monospace) font.
	// This ensures that all characters have the same width:
	QFont font;
	font.setFamily( "Courier" );
	font.setStyleHint( QFont::Monospace );
	font.setFixedPitch( true );
	font.setPointSize( 10 );	// TO DO : add user interface editor to be able to customize text size
	_ui._vertexShaderTextEdit->setFont( font );
	_ui._tesselationControlShaderTextEdit->setFont( font );
	_ui._tesselationEvaluationShaderTextEdit->setFont( font );
	_ui._geometryShaderTextEdit->setFont( font );
	_ui._fragmentShaderTextEdit->setFont( font );
	_ui._computeShaderTextEdit->setFont( font );
	
	// If you want to set a tab width to certain amount of spaces,
	// as it is typically done in text editors,
	// use QFontMetrics to compute the size of one space in pixels:
	const int tabStop = 4;  // 4 characters
	QFontMetrics metrics( font );
	_ui._vertexShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
	_ui._tesselationControlShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
	_ui._tesselationEvaluationShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
	_ui._geometryShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
	_ui._fragmentShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
	_ui._computeShaderTextEdit->setTabStopWidth( tabStop * metrics.width( ' ' ) );
}

/******************************************************************************
 * Apply action
 ******************************************************************************/
void GvvGLSLSourceEditor::onApply()
{
}

/******************************************************************************
 * Compile action
 ******************************************************************************/
void GvvGLSLSourceEditor::onReload()
{
}

/******************************************************************************
 * ...
 *
 * @param ...
 ******************************************************************************/
void GvvGLSLSourceEditor::populate( GvvPipelineInterface* pPipeline )
{
	_pipeline = pPipeline;

	if ( pPipeline != NULL )
	{
		_ui.tabWidget->setTabEnabled( 0, pPipeline->hasShaderType( 0 ) );
		_ui.tabWidget->setTabEnabled( 1, pPipeline->hasShaderType( 1 ) );
		_ui.tabWidget->setTabEnabled( 2, pPipeline->hasShaderType( 2 ) );
		_ui.tabWidget->setTabEnabled( 3, pPipeline->hasShaderType( 3 ) );
		_ui.tabWidget->setTabEnabled( 4, pPipeline->hasShaderType( 4 ) );
		_ui.tabWidget->setTabEnabled( 5, pPipeline->hasShaderType( 5 ) );

		if ( pPipeline->hasShaderType( 0 ) )
		{
			_ui._vertexShaderTextEdit->setText( pPipeline->getShaderSourceCode( 0 ).c_str() );
		}

		if ( pPipeline->hasShaderType( 1 ) )
		{
			_ui._tesselationControlShaderTextEdit->setText( pPipeline->getShaderSourceCode( 1 ).c_str() );
		}

		if ( pPipeline->hasShaderType( 2 ) )
		{
			_ui._tesselationEvaluationShaderTextEdit->setText( pPipeline->getShaderSourceCode( 2 ).c_str() );
		}

		if ( pPipeline->hasShaderType( 3 ) )
		{
			_ui._geometryShaderTextEdit->setText( pPipeline->getShaderSourceCode( 3 ).c_str() );
		}

		if ( pPipeline->hasShaderType( 4 ) )
		{
			_ui._fragmentShaderTextEdit->setText( pPipeline->getShaderSourceCode( 4 ).c_str() );
		}

		if ( pPipeline->hasShaderType( 5 ) )
		{
			_ui._computeShaderTextEdit->setText( pPipeline->getShaderSourceCode( 5 ).c_str() );
		}

		int currentShaderIndex = _ui.tabWidget->currentIndex();
		if ( currentShaderIndex != -1 )
		{
			_ui._shaderFilenameLineEdit->setText( pPipeline->getShaderFilename( static_cast< unsigned int >( currentShaderIndex ) ).c_str() );
		}
	}
}

/******************************************************************************
 * Slot called when current page index has changed
 *
 * @param pIndex ...
 ******************************************************************************/
void GvvGLSLSourceEditor::on_tabWidget_currentChanged( int pIndex )
{
	if ( pIndex != -1 )
	{
		if ( _pipeline != NULL )
		{
			_ui._shaderFilenameLineEdit->setText( _pipeline->getShaderFilename( static_cast< unsigned int >( pIndex ) ).c_str() );
		}
	}
	else
	{
		_ui._shaderFilenameLineEdit->setText( "" );
	}
}

/******************************************************************************
 * Slot called when apply button has been released
 ******************************************************************************/
void GvvGLSLSourceEditor::on__applyButton_released()
{
	if ( _pipeline != NULL )
	{
		int currentShaderIndex = _ui.tabWidget->currentIndex();
		if ( currentShaderIndex != -1 )
		{
			QString shaderSource;
			switch ( currentShaderIndex )
			{
			case 0:
				shaderSource = _ui._vertexShaderTextEdit->toPlainText();
				break;

			case 1:
				shaderSource = _ui._tesselationControlShaderTextEdit->toPlainText();
				break;

			case 2:
				shaderSource = _ui._tesselationEvaluationShaderTextEdit->toPlainText();
				break;

			case 3:
				shaderSource = _ui._geometryShaderTextEdit->toPlainText();
				break;

			case 4:
				shaderSource = _ui._fragmentShaderTextEdit->toPlainText();
				break;

			case 5:
				shaderSource = _ui._computeShaderTextEdit->toPlainText();
				break;

			default:

				assert( false );

				break;
			}

			QString shaderFilename = _pipeline->getShaderFilename( static_cast< unsigned int >( currentShaderIndex ) ).c_str();
			QFile shaderFile( shaderFilename );
			if ( ! shaderFile.open( QIODevice::WriteOnly | QIODevice::Text ) )
			{
				return;
			}
			QTextStream out( &shaderFile );
			out << shaderSource;
			shaderFile.close();

			// Reload shader
			_pipeline->reloadShader( static_cast< unsigned int >( currentShaderIndex ) );
		}
	}
}
