/********************************************************************************
** Form generated from reading UI file 'GvvQMonitoringEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQMONITORINGEDITOR_H
#define UI_GVVQMONITORINGEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQMonitoringEditor
{
public:
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *_timeBudgetLineEdit;
    QSpinBox *_timeBudgetSpinBox;
    QGroupBox *groupBox_2;
    QVBoxLayout *verticalLayout;
    QCheckBox *checkBox_2;
    QCheckBox *checkBox;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQMonitoringEditor)
    {
        if (GvvQMonitoringEditor->objectName().isEmpty())
            GvvQMonitoringEditor->setObjectName(QString::fromUtf8("GvvQMonitoringEditor"));
        GvvQMonitoringEditor->resize(310, 250);
        verticalLayout_2 = new QVBoxLayout(GvvQMonitoringEditor);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        groupBox = new QGroupBox(GvvQMonitoringEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setCheckable(true);
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        _timeBudgetLineEdit = new QLineEdit(groupBox);
        _timeBudgetLineEdit->setObjectName(QString::fromUtf8("_timeBudgetLineEdit"));
        _timeBudgetLineEdit->setEnabled(false);
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(_timeBudgetLineEdit->sizePolicy().hasHeightForWidth());
        _timeBudgetLineEdit->setSizePolicy(sizePolicy);
        _timeBudgetLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _timeBudgetLineEdit->setReadOnly(true);

        horizontalLayout->addWidget(_timeBudgetLineEdit);

        _timeBudgetSpinBox = new QSpinBox(groupBox);
        _timeBudgetSpinBox->setObjectName(QString::fromUtf8("_timeBudgetSpinBox"));
        sizePolicy.setHeightForWidth(_timeBudgetSpinBox->sizePolicy().hasHeightForWidth());
        _timeBudgetSpinBox->setSizePolicy(sizePolicy);
        _timeBudgetSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _timeBudgetSpinBox->setMinimum(1);
        _timeBudgetSpinBox->setMaximum(120);
        _timeBudgetSpinBox->setValue(60);

        horizontalLayout->addWidget(_timeBudgetSpinBox);


        verticalLayout_2->addWidget(groupBox);

        groupBox_2 = new QGroupBox(GvvQMonitoringEditor);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setCheckable(true);
        verticalLayout = new QVBoxLayout(groupBox_2);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        checkBox_2 = new QCheckBox(groupBox_2);
        checkBox_2->setObjectName(QString::fromUtf8("checkBox_2"));

        verticalLayout->addWidget(checkBox_2);

        checkBox = new QCheckBox(groupBox_2);
        checkBox->setObjectName(QString::fromUtf8("checkBox"));

        verticalLayout->addWidget(checkBox);


        verticalLayout_2->addWidget(groupBox_2);

        verticalSpacer = new QSpacerItem(20, 91, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_2->addItem(verticalSpacer);


        retranslateUi(GvvQMonitoringEditor);

        QMetaObject::connectSlotsByName(GvvQMonitoringEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQMonitoringEditor)
    {
        GvvQMonitoringEditor->setWindowTitle(QApplication::translate("GvvQMonitoringEditor", "Form", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQMonitoringEditor", "Rendering", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvvQMonitoringEditor", "Time Budget", 0, QApplication::UnicodeUTF8));
        _timeBudgetSpinBox->setSuffix(QApplication::translate("GvvQMonitoringEditor", " fps", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("GvvQMonitoringEditor", "Data Production Management", 0, QApplication::UnicodeUTF8));
        checkBox_2->setText(QApplication::translate("GvvQMonitoringEditor", "Production", 0, QApplication::UnicodeUTF8));
        checkBox->setText(QApplication::translate("GvvQMonitoringEditor", "Usage", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQMonitoringEditor: public Ui_GvvQMonitoringEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQMONITORINGEDITOR_H
