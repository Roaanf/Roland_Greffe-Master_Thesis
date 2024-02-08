/********************************************************************************
** Form generated from reading UI file 'GvQCUDASourceEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQCUDASOURCEEDITOR_H
#define UI_GVQCUDASOURCEEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvQCUDASourceEditor
{
public:
    QVBoxLayout *vboxLayout;
    QTextEdit *_textEdit;
    QHBoxLayout *hboxLayout;
    QPushButton *_applyButton;
    QPushButton *_compileButton;
    QSpacerItem *spacerItem;

    void setupUi(QWidget *GvQCUDASourceEditor)
    {
        if (GvQCUDASourceEditor->objectName().isEmpty())
            GvQCUDASourceEditor->setObjectName(QString::fromUtf8("GvQCUDASourceEditor"));
        GvQCUDASourceEditor->setWindowModality(Qt::WindowModal);
        GvQCUDASourceEditor->resize(398, 548);
        vboxLayout = new QVBoxLayout(GvQCUDASourceEditor);
        vboxLayout->setObjectName(QString::fromUtf8("vboxLayout"));
        _textEdit = new QTextEdit(GvQCUDASourceEditor);
        _textEdit->setObjectName(QString::fromUtf8("_textEdit"));

        vboxLayout->addWidget(_textEdit);

        hboxLayout = new QHBoxLayout();
        hboxLayout->setObjectName(QString::fromUtf8("hboxLayout"));
        _applyButton = new QPushButton(GvQCUDASourceEditor);
        _applyButton->setObjectName(QString::fromUtf8("_applyButton"));

        hboxLayout->addWidget(_applyButton);

        _compileButton = new QPushButton(GvQCUDASourceEditor);
        _compileButton->setObjectName(QString::fromUtf8("_compileButton"));

        hboxLayout->addWidget(_compileButton);

        spacerItem = new QSpacerItem(201, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        hboxLayout->addItem(spacerItem);


        vboxLayout->addLayout(hboxLayout);


        retranslateUi(GvQCUDASourceEditor);

        QMetaObject::connectSlotsByName(GvQCUDASourceEditor);
    } // setupUi

    void retranslateUi(QWidget *GvQCUDASourceEditor)
    {
        GvQCUDASourceEditor->setWindowTitle(QApplication::translate("GvQCUDASourceEditor", "CUDA - Program Source Editor", 0, QApplication::UnicodeUTF8));
        _applyButton->setText(QApplication::translate("GvQCUDASourceEditor", "Apply", 0, QApplication::UnicodeUTF8));
        _compileButton->setText(QApplication::translate("GvQCUDASourceEditor", "Compile", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvQCUDASourceEditor: public Ui_GvQCUDASourceEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQCUDASOURCEEDITOR_H
