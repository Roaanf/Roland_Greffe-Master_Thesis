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

#ifndef _GVV_CUDA_SYNTAX_HIGHLIGHTER_H_
#define _GVV_CUDA_SYNTAX_HIGHLIGHTER_H_

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

// GvViewer
#include "GvvGuiConfig.h"

// Qt
#include <QSyntaxHighlighter>
#include <QHash>
#include <QTextCharFormat>

/******************************************************************************
 ************************* DEFINE AND CONSTANT SECTION ************************
 ******************************************************************************/

/******************************************************************************
 ***************************** TYPE DEFINITION ********************************
 ******************************************************************************/

/******************************************************************************
 ******************************** CLASS USED **********************************
 ******************************************************************************/

// class Qt
class QTextDocument;

/******************************************************************************
 ****************************** CLASS DEFINITION ******************************
 ******************************************************************************/

namespace GvViewerGui
{

/**
 * GLSL syntax highligther
 */
class GVVIEWERGUI_EXPORT GvvCUDASyntaxHighlighter : public QSyntaxHighlighter
{
	// Qt macro
    Q_OBJECT

	/**************************************************************************
	 ***************************** PUBLIC SECTION *****************************
	 **************************************************************************/
public:

	/***************************** INNER CLASSES ******************************/

	/**
	 * A highlight rule
	 */
    struct GvvHighlightingRule
    {
        QRegExp _pattern;
        QTextCharFormat _format;
    };
	
	/******************************** METHODS *********************************/

	/**
	 * Default constructor
	 *
	 * @param pParent the parent text document
	 */
    GvvCUDASyntaxHighlighter( QTextDocument* pParent = 0 );

	/**
	 * Add a highlight rule to the highlighter
	 *
	 * @param pHighlightRule the highlight rule
	 */
	void appendRule( const GvvHighlightingRule& pHighlightRule );

	/**************************************************************************
	 **************************** PROTECTED SECTION ***************************
	 **************************************************************************/

protected:
	
	/******************************** METHODS *********************************/
	
	/**
	 * Highlights the given text block
	 *
	 * @param pText the text block to highlight.
	 */
    virtual void highlightBlock( const QString& pText );

	/**************************************************************************
	 ***************************** PRIVATE SECTION ****************************
	 **************************************************************************/

private:
	
	/******************************* ATTRIBUTES *******************************/

	/**
	 * The highlight rule container
	 */
    QVector< GvvHighlightingRule > _highlightingRules;

	/**
	 * The comment expressions
	 */
    QRegExp _commentStartExpression;
    QRegExp _commentEndExpression;

	/**
	 * The keywords
	 */
    QTextCharFormat _singleLineCommentFormat;
    QTextCharFormat _multiLineCommentFormat;

	/******************************** METHODS *********************************/
};

} // namespace GvViewerGui

#endif // _GVV_GLSL_SYNTAX_HIGHLIGHTER_H_
