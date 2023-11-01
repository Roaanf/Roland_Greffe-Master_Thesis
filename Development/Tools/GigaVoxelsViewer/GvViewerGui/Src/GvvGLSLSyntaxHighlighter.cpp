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

#include "GvvGLSLSyntaxHighlighter.h"

/******************************************************************************
 ******************************* INCLUDE SECTION ******************************
 ******************************************************************************/

/******************************************************************************
 ****************************** NAMESPACE SECTION *****************************
 ******************************************************************************/

// GvViewer
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
 *
 * @param pParent the parent text document
 ******************************************************************************/
GvvGLSLSyntaxHighlighter::GvvGLSLSyntaxHighlighter( QTextDocument* pParent )
:	QSyntaxHighlighter( pParent )
{
    GvvHighlightingRule rule;

	// Keywords
	// --------
    QStringList keywords;
    keywords 	<< "\\bif\\b" << "\\bthen\\b" << "\\belse\\b" << "\\bvoid\\b" 
				<< "\\bfor\\b" << "\\bdo\\b" << "\\bwhile\\b" 
				<< "\\bdiscard\\b" << "\\breturn\\b" << "\\bbreak\\b" << "\\bcontinue\\b"
				<< "\\bbool\\b" << "\\btrue\\b" << "\\bfalse\\b" 
				<< "\\bint\\b" << "\\buint\\b" 
				<< "\\bfloat\\b"
				<< "\\bvec2\\b" << "\\bvec3\\b" << "\\bvec4\\b"
				<< "\\buvec2\\b" << "\\buvec3\\b" << "\\buvec4\\b"
				<< "\\buimageBuffer\\b"
				<< "\\bmat4\\b" << "\\bmat3\\b"
				<< "\\buniform\\b"
				<< "\\blayout\\b"
				<< "\\bstruct\\b"
				<< "\\bsampler2D\\b" << "\\bsampler3D \\b" << "\\bsampler2DShadow\\b"
				//<< "\\bversion\\b" << "\\bextension\\b"
				<< "\\bin\\b" << "\\bout\\b" << "\\binout\\b";
				
	QTextCharFormat keywordFormat;
	keywordFormat.setForeground( Qt::darkCyan );
    keywordFormat.setFontWeight( QFont::Bold );

    foreach( QString pattern, keywords )
	{
        rule._pattern = QRegExp( pattern );
        rule._format = keywordFormat;
        _highlightingRules.append( rule );
    }

	// Statement
	// ---------
 /*   QStringList lStatements;
	lStatements << "\\breturn\\b" << "\\blocal\\b" << "\\bbreak\\b";

	QTextCharFormat lStatementFormat;
    lStatementFormat.setForeground(Qt::darkYellow);

    foreach (QString lPattern, lStatements) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lStatementFormat;
        _highlightingRules.append(rule);
    }
*/

	// Operators
	// ---------
    QStringList lOperators;
	lOperators << "\\band\\b" << "\\bor\\b" << "\\bnot\\b";

	QTextCharFormat lOperatorFormat;
    lOperatorFormat.setForeground(Qt::darkMagenta);
    lOperatorFormat.setFontWeight(QFont::Bold);

    foreach (QString lPattern, lOperators) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lOperatorFormat;
        _highlightingRules.append(rule);
    }

	// Functions
	// ---------
    QStringList lFunctions;
	lFunctions	<< "\\btexture\\b"
				<< "\\bimageLoad\\b" << "\\bimageStore\\b"
				<< "\\bmin\\b" << "\\bmax\\b"
				<< "\\bstep\\b"
				<< "\\bfract\\b" << "\\blog2\\b" << "\\bceil\\b" << "\\btan\\b" << "\\bradians\\b" << "\\bcos\\b" << "\\bsin\\b";

	QTextCharFormat lFunctionFormat;
    lFunctionFormat.setForeground( Qt::darkMagenta );
    lFunctionFormat.setFontWeight( QFont::Bold );

    foreach (QString lPattern, lFunctions) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lFunctionFormat;
        _highlightingRules.append(rule);
    }
	
	// GLSL keywords
	// ---------
    QStringList lGLSLkeywords;
	lGLSLkeywords	<< "\\bgl_FragCoord\\b"
					<< "\\bgl_Position\\b";

	QTextCharFormat lGLSLkeywordFormat;
    lGLSLkeywordFormat.setForeground( Qt::darkBlue );
    lGLSLkeywordFormat.setFontWeight( QFont::Bold );

    foreach (QString lPattern, lGLSLkeywords) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lGLSLkeywordFormat;
        _highlightingRules.append(rule);
    }
	
	// Todo
	// ----
    QStringList lTodo;
	lTodo << "\\bTODO\\b" << "\\bFIXME\\b";

	QTextCharFormat lTodoFormat;
    lTodoFormat.setForeground(Qt::red);
    lTodoFormat.setFontWeight(QFont::Bold);
    lTodoFormat.setBackground(QBrush(Qt::yellow));

    foreach (QString lPattern, lTodo) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lTodoFormat;
        _highlightingRules.append(rule);
    }

	rule._pattern = QRegExp("\\b([A-Z])\\1{2,}\\b");
    rule._format = lTodoFormat;
    _highlightingRules.append(rule);

	// Numbers
	// -------
	QTextCharFormat lNumberFormat;
    lNumberFormat.setForeground(Qt::red);
    rule._format = lNumberFormat;
    rule._pattern = QRegExp("\\b[0-9]+(\\.[0-9]*)?(e[-+]?[0-9]+)?\\b");
    _highlightingRules.append(rule);
    
    lNumberFormat.setFontWeight(QFont::Bold);
    lNumberFormat.setForeground(Qt::red);
    rule._pattern = QRegExp("\\#[^0-9][A-Za-z0-9_]+");
    _highlightingRules.append(rule);
	
	// Tables
	// ------
	QTextCharFormat lTableFormat;
    lTableFormat.setForeground(Qt::darkGreen);
    rule._pattern = QRegExp("[{}]");
    rule._format = lTableFormat;
    _highlightingRules.append(rule);

	// Quotations
	// ----------
	QTextCharFormat lQuotationFormat;
    lQuotationFormat.setForeground(Qt::darkRed);
    rule._pattern = QRegExp("['][^']*[']");
    rule._format = lQuotationFormat;
    _highlightingRules.append(rule);
    rule._pattern = QRegExp("[\"][^\"]*[\"]");
    _highlightingRules.append(rule);
	
	// Comments
	// --------
    //_singleLineCommentFormat.setForeground(Qt::darkGray);
	 _singleLineCommentFormat.setForeground(Qt::green);
    rule._pattern = QRegExp("--[^\n]*");
    rule._format = _singleLineCommentFormat;
    _highlightingRules.append(rule);

    //_multiLineCommentFormat.setForeground(Qt::darkGray);
	_multiLineCommentFormat.setForeground(Qt::green);

    _commentStartExpression = QRegExp("--\\[=*\\[");
	_commentEndExpression = QRegExp("\\]=*\\]--");
}

/******************************************************************************
 * Add a highlight rule to the highlighter
 *
 * @param pHighlightRule the highlight rule
 ******************************************************************************/
void GvvGLSLSyntaxHighlighter::appendRule( const GvvHighlightingRule& pHighlightRule )
{
	_highlightingRules.append( pHighlightRule );
}

/******************************************************************************
 * Highlights the given text block
 *
 * @param pText the text block to highlight.
 ******************************************************************************/
void GvvGLSLSyntaxHighlighter::highlightBlock( const QString& pText )
{
   // Code inspired from the Qt documentation of the class QSyntaxHighlighter
	// to write a simple C++ syntax highlighter.

    foreach ( GvvHighlightingRule rule, _highlightingRules )
	{
        QRegExp expression( rule._pattern );
        int index = pText.indexOf( expression );
        while ( index >= 0 ) 
		{
            int length = expression.matchedLength();
            setFormat( index, length, rule._format );
            index = pText.indexOf( expression, index + length );
        }
    }
    setCurrentBlockState( 0 );

    int startIndex = 0;
    if ( previousBlockState() != 1 )
	{
        startIndex = pText.indexOf( _commentStartExpression );
	}

    while ( startIndex >= 0 )
	{
        int endIndex = pText.indexOf( _commentEndExpression, startIndex );
        int commentLength;
        if ( endIndex == -1 )
		{
            setCurrentBlockState( 1 );
            commentLength = pText.length() - startIndex;
        }
		else
		{
            commentLength = endIndex - startIndex + _commentEndExpression.matchedLength();
        }
        setFormat( startIndex, commentLength, _multiLineCommentFormat );
        startIndex = pText.indexOf( _commentStartExpression, startIndex + commentLength );
    }
}
