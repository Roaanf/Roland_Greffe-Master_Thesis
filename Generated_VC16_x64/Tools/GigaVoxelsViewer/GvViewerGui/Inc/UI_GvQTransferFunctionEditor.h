/********************************************************************************
** Form generated from reading UI file 'GvQTransferFunctionEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQTRANSFERFUNCTIONEDITOR_H
#define UI_GVQTRANSFERFUNCTIONEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QToolButton>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvQTransferFunctionEditor
{
public:
    QGridLayout *gridLayout;
    QGroupBox *groupBox_2;
    QHBoxLayout *horizontalLayout;
    QLineEdit *_filenameLineEdit;
    QToolButton *_loadToolButton;
    QGroupBox *_transferFunctionGroupBox;
    QPushButton *_savePushButton;
    QPushButton *_saveAsPushButton;
    QSpacerItem *horizontalSpacer_2;
    QPushButton *_quitPushButton;

    void setupUi(QWidget *GvQTransferFunctionEditor)
    {
        if (GvQTransferFunctionEditor->objectName().isEmpty())
            GvQTransferFunctionEditor->setObjectName(QString::fromUtf8("GvQTransferFunctionEditor"));
        GvQTransferFunctionEditor->resize(356, 451);
        gridLayout = new QGridLayout(GvQTransferFunctionEditor);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        groupBox_2 = new QGroupBox(GvQTransferFunctionEditor);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(groupBox_2->sizePolicy().hasHeightForWidth());
        groupBox_2->setSizePolicy(sizePolicy);
        horizontalLayout = new QHBoxLayout(groupBox_2);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        _filenameLineEdit = new QLineEdit(groupBox_2);
        _filenameLineEdit->setObjectName(QString::fromUtf8("_filenameLineEdit"));
        _filenameLineEdit->setReadOnly(true);

        horizontalLayout->addWidget(_filenameLineEdit);

        _loadToolButton = new QToolButton(groupBox_2);
        _loadToolButton->setObjectName(QString::fromUtf8("_loadToolButton"));

        horizontalLayout->addWidget(_loadToolButton);


        gridLayout->addWidget(groupBox_2, 0, 0, 1, 4);

        _transferFunctionGroupBox = new QGroupBox(GvQTransferFunctionEditor);
        _transferFunctionGroupBox->setObjectName(QString::fromUtf8("_transferFunctionGroupBox"));

        gridLayout->addWidget(_transferFunctionGroupBox, 1, 0, 1, 4);

        _savePushButton = new QPushButton(GvQTransferFunctionEditor);
        _savePushButton->setObjectName(QString::fromUtf8("_savePushButton"));

        gridLayout->addWidget(_savePushButton, 2, 0, 1, 1);

        _saveAsPushButton = new QPushButton(GvQTransferFunctionEditor);
        _saveAsPushButton->setObjectName(QString::fromUtf8("_saveAsPushButton"));

        gridLayout->addWidget(_saveAsPushButton, 2, 1, 1, 1);

        horizontalSpacer_2 = new QSpacerItem(92, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer_2, 2, 2, 1, 1);

        _quitPushButton = new QPushButton(GvQTransferFunctionEditor);
        _quitPushButton->setObjectName(QString::fromUtf8("_quitPushButton"));

        gridLayout->addWidget(_quitPushButton, 2, 3, 1, 1);


        retranslateUi(GvQTransferFunctionEditor);

        QMetaObject::connectSlotsByName(GvQTransferFunctionEditor);
    } // setupUi

    void retranslateUi(QWidget *GvQTransferFunctionEditor)
    {
        GvQTransferFunctionEditor->setWindowTitle(QApplication::translate("GvQTransferFunctionEditor", "Transfer Function Editor", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("GvQTransferFunctionEditor", "File", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _loadToolButton->setToolTip(QApplication::translate("GvQTransferFunctionEditor", "Open a Qtfe transfer function's file.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        _loadToolButton->setText(QApplication::translate("GvQTransferFunctionEditor", "...", 0, QApplication::UnicodeUTF8));
        _transferFunctionGroupBox->setTitle(QApplication::translate("GvQTransferFunctionEditor", "Transfer Functions", 0, QApplication::UnicodeUTF8));
        _savePushButton->setText(QApplication::translate("GvQTransferFunctionEditor", "Save", 0, QApplication::UnicodeUTF8));
        _saveAsPushButton->setText(QApplication::translate("GvQTransferFunctionEditor", "Save As", 0, QApplication::UnicodeUTF8));
        _quitPushButton->setText(QApplication::translate("GvQTransferFunctionEditor", "Quit", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvQTransferFunctionEditor: public Ui_GvQTransferFunctionEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQTRANSFERFUNCTIONEDITOR_H
