/********************************************************************************
** Form generated from reading UI file 'GvvQLightEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQLIGHTEDITOR_H
#define UI_GVVQLIGHTEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQLightEditor
{
public:
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QLabel *label_2;
    QDoubleSpinBox *doubleSpinBox;
    QSpacerItem *horizontalSpacer_2;
    QLabel *label_3;
    QDoubleSpinBox *doubleSpinBox_2;
    QSpacerItem *horizontalSpacer;
    QLabel *label_4;
    QDoubleSpinBox *doubleSpinBox_3;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQLightEditor)
    {
        if (GvvQLightEditor->objectName().isEmpty())
            GvvQLightEditor->setObjectName(QString::fromUtf8("GvvQLightEditor"));
        GvvQLightEditor->resize(298, 213);
        gridLayout = new QGridLayout(GvvQLightEditor);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(GvvQLightEditor);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineEdit = new QLineEdit(GvvQLightEditor);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        groupBox = new QGroupBox(GvvQLightEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        horizontalLayout->addWidget(label_2);

        doubleSpinBox = new QDoubleSpinBox(groupBox);
        doubleSpinBox->setObjectName(QString::fromUtf8("doubleSpinBox"));
        doubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout->addWidget(doubleSpinBox);

        horizontalSpacer_2 = new QSpacerItem(7, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        horizontalLayout->addWidget(label_3);

        doubleSpinBox_2 = new QDoubleSpinBox(groupBox);
        doubleSpinBox_2->setObjectName(QString::fromUtf8("doubleSpinBox_2"));
        doubleSpinBox_2->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout->addWidget(doubleSpinBox_2);

        horizontalSpacer = new QSpacerItem(8, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        horizontalLayout->addWidget(label_4);

        doubleSpinBox_3 = new QDoubleSpinBox(groupBox);
        doubleSpinBox_3->setObjectName(QString::fromUtf8("doubleSpinBox_3"));
        doubleSpinBox_3->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        horizontalLayout->addWidget(doubleSpinBox_3);


        gridLayout->addWidget(groupBox, 1, 0, 1, 2);

        verticalSpacer = new QSpacerItem(20, 112, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 2, 1, 1, 1);


        retranslateUi(GvvQLightEditor);

        QMetaObject::connectSlotsByName(GvvQLightEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQLightEditor)
    {
        GvvQLightEditor->setWindowTitle(QApplication::translate("GvvQLightEditor", "Light Editor", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvvQLightEditor", "Name", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQLightEditor", "Position", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("GvvQLightEditor", "X", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQLightEditor", "Y", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQLightEditor", "Z", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQLightEditor: public Ui_GvvQLightEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQLIGHTEDITOR_H
