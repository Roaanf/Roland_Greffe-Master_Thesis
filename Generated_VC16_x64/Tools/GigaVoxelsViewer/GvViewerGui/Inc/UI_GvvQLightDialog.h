/********************************************************************************
** Form generated from reading UI file 'GvvQLightDialog.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQLIGHTDIALOG_H
#define UI_GVVQLIGHTDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_GvvQLightDialog
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
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *GvvQLightDialog)
    {
        if (GvvQLightDialog->objectName().isEmpty())
            GvvQLightDialog->setObjectName(QString::fromUtf8("GvvQLightDialog"));
        GvvQLightDialog->resize(277, 166);
        gridLayout = new QGridLayout(GvvQLightDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(GvvQLightDialog);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        lineEdit = new QLineEdit(GvvQLightDialog);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        gridLayout->addWidget(lineEdit, 0, 1, 1, 1);

        groupBox = new QGroupBox(GvvQLightDialog);
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

        verticalSpacer = new QSpacerItem(20, 37, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 2, 1, 1, 1);

        buttonBox = new QDialogButtonBox(GvvQLightDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        gridLayout->addWidget(buttonBox, 3, 0, 1, 2);


        retranslateUi(GvvQLightDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GvvQLightDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GvvQLightDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(GvvQLightDialog);
    } // setupUi

    void retranslateUi(QDialog *GvvQLightDialog)
    {
        GvvQLightDialog->setWindowTitle(QApplication::translate("GvvQLightDialog", "Light Dialog", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvvQLightDialog", "Name", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQLightDialog", "Position", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("GvvQLightDialog", "X", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQLightDialog", "Y", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQLightDialog", "Z", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQLightDialog: public Ui_GvvQLightDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQLIGHTDIALOG_H
