/********************************************************************************
** Form generated from reading UI file 'GvvQTimeBudgetView.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQTIMEBUDGETVIEW_H
#define UI_GVVQTIMEBUDGETVIEW_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQTimeBudgetView
{
public:
    QHBoxLayout *horizontalLayout;
    QGroupBox *_frameTimeViewGroupBox;

    void setupUi(QWidget *GvvQTimeBudgetView)
    {
        if (GvvQTimeBudgetView->objectName().isEmpty())
            GvvQTimeBudgetView->setObjectName(QString::fromUtf8("GvvQTimeBudgetView"));
        GvvQTimeBudgetView->resize(417, 172);
        horizontalLayout = new QHBoxLayout(GvvQTimeBudgetView);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        _frameTimeViewGroupBox = new QGroupBox(GvvQTimeBudgetView);
        _frameTimeViewGroupBox->setObjectName(QString::fromUtf8("_frameTimeViewGroupBox"));

        horizontalLayout->addWidget(_frameTimeViewGroupBox);


        retranslateUi(GvvQTimeBudgetView);

        QMetaObject::connectSlotsByName(GvvQTimeBudgetView);
    } // setupUi

    void retranslateUi(QWidget *GvvQTimeBudgetView)
    {
        GvvQTimeBudgetView->setWindowTitle(QApplication::translate("GvvQTimeBudgetView", "Time Budget Monitoring", 0, QApplication::UnicodeUTF8));
        _frameTimeViewGroupBox->setTitle(QString());
    } // retranslateUi

};

namespace Ui {
    class GvvQTimeBudgetView: public Ui_GvvQTimeBudgetView {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQTIMEBUDGETVIEW_H
