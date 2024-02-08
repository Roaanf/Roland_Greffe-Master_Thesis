/********************************************************************************
** Form generated from reading UI file 'GvvQEditorWidget.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQEDITORWIDGET_H
#define UI_GVVQEDITORWIDGET_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QTabWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQEditorWidget
{
public:
    QVBoxLayout *verticalLayout_3;
    QTabWidget *tabWidget;

    void setupUi(QWidget *GvvQEditorWidget)
    {
        if (GvvQEditorWidget->objectName().isEmpty())
            GvvQEditorWidget->setObjectName(QString::fromUtf8("GvvQEditorWidget"));
        GvvQEditorWidget->resize(202, 357);
        verticalLayout_3 = new QVBoxLayout(GvvQEditorWidget);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        tabWidget = new QTabWidget(GvvQEditorWidget);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setTabShape(QTabWidget::Rounded);

        verticalLayout_3->addWidget(tabWidget);


        retranslateUi(GvvQEditorWidget);

        tabWidget->setCurrentIndex(-1);


        QMetaObject::connectSlotsByName(GvvQEditorWidget);
    } // setupUi

    void retranslateUi(QWidget *GvvQEditorWidget)
    {
        GvvQEditorWidget->setWindowTitle(QApplication::translate("GvvQEditorWidget", "Form", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQEditorWidget: public Ui_GvvQEditorWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQEDITORWIDGET_H
