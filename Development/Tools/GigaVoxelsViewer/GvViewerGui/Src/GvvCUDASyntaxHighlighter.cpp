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

#include "GvvCUDASyntaxHighlighter.h"

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
GvvCUDASyntaxHighlighter::GvvCUDASyntaxHighlighter( QTextDocument* pParent )
:	QSyntaxHighlighter( pParent )
{
    GvvHighlightingRule rule;

	// Keywords
	// --------
    QStringList lKeywords;
 	lKeywords 	<< "\\bif\\b" << "\\bthen\\b" << "\\belse\\b" 
				<< "\\bfor\\b" << "\\bdo\\b" << "\\bwhile\\b" 
				<< "\\breturn\\b" << "\\bbreak\\b" << "\\bcontinue\\b"
				<< "\\btrue\\b" << "\\bfalse\\b" 
				<< "\\bfloat\\b" << "\\bfloat2\\b" << "\\bfloat3\\b" << "\\bfloat4\\b"
				<< "\\buchar\\b" << "\\buchar4\\b";
	
	QTextCharFormat lKeywordFormat;
	lKeywordFormat.setForeground(Qt::darkCyan);
    lKeywordFormat.setFontWeight(QFont::Bold);

    foreach (QString lPattern, lKeywords) {
        rule._pattern = QRegExp(lPattern);
        rule._format = lKeywordFormat;
        _highlightingRules.append(rule);
    }

	//-------------------------------------------
	// Qualifier Keywords
	// --------
	{
		QStringList lQualifierKeywords;
 		lQualifierKeywords 	<< "\\b__global__\\b" << "\\b__device__\\b" << "\\b__shared__\\b" << "\\b__constant__\\b";
		QTextCharFormat lQualifierKeywordFormat;
		lQualifierKeywordFormat.setForeground( Qt::red );
		lQualifierKeywordFormat.setFontWeight( QFont::Bold );

		foreach (QString lPattern, lQualifierKeywords)
		{
			rule._pattern = QRegExp( lPattern );
			rule._format = lQualifierKeywordFormat;
			_highlightingRules.append(rule);
		}
	}
	//-------------------------------------------

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
    _highlightingRules.append( rule );

    //_multiLineCommentFormat.setForeground(Qt::darkGray);
	_multiLineCommentFormat.setForeground( Qt::green );

    _commentStartExpression = QRegExp("--\\[=*\\[");
	_commentEndExpression = QRegExp("\\]=*\\]--");
}

/******************************************************************************
 * Add a highlight rule to the highlighter
 *
 * @param pHighlightRule the highlight rule
 ******************************************************************************/
void GvvCUDASyntaxHighlighter::appendRule( const GvvHighlightingRule& pHighlightRule )
{
	_highlightingRules.append( pHighlightRule );
}

/******************************************************************************
 * Highlights the given text block
 *
 * @param pText the text block to highlight.
 ******************************************************************************/
void GvvCUDASyntaxHighlighter::highlightBlock( const QString& pText )
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
