/********************************************************************************
** Form generated from reading UI file 'GvQDataLoaderDialog.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQDATALOADERDIALOG_H
#define UI_GVQDATALOADERDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QToolButton>

QT_BEGIN_NAMESPACE

class Ui_GvQDataLoaderDialog
{
public:
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *_3DModelLineEdit;
    QToolButton *_3DModelToolButton;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *GvQDataLoaderDialog)
    {
        if (GvQDataLoaderDialog->objectName().isEmpty())
            GvQDataLoaderDialog->setObjectName(QString::fromUtf8("GvQDataLoaderDialog"));
        GvQDataLoaderDialog->resize(516, 94);
        gridLayout_2 = new QGridLayout(GvQDataLoaderDialog);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        groupBox = new QGroupBox(GvQDataLoaderDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        _3DModelLineEdit = new QLineEdit(groupBox);
        _3DModelLineEdit->setObjectName(QString::fromUtf8("_3DModelLineEdit"));
        _3DModelLineEdit->setEnabled(true);

        gridLayout->addWidget(_3DModelLineEdit, 0, 2, 1, 2);

        _3DModelToolButton = new QToolButton(groupBox);
        _3DModelToolButton->setObjectName(QString::fromUtf8("_3DModelToolButton"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/Icons/fileopen.png"), QSize(), QIcon::Normal, QIcon::Off);
        _3DModelToolButton->setIcon(icon);

        gridLayout->addWidget(_3DModelToolButton, 0, 4, 1, 1);


        gridLayout_2->addWidget(groupBox, 0, 0, 1, 1);

        buttonBox = new QDialogButtonBox(GvQDataLoaderDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        gridLayout_2->addWidget(buttonBox, 4, 0, 1, 1);


        retranslateUi(GvQDataLoaderDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GvQDataLoaderDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GvQDataLoaderDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(GvQDataLoaderDialog);
    } // setupUi

    void retranslateUi(QDialog *GvQDataLoaderDialog)
    {
        GvQDataLoaderDialog->setWindowTitle(QApplication::translate("GvQDataLoaderDialog", "GigaSpace - Model Loader", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvQDataLoaderDialog", "Model", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvQDataLoaderDialog", "Filename", 0, QApplication::UnicodeUTF8));
        _3DModelToolButton->setText(QApplication::translate("GvQDataLoaderDialog", "...", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvQDataLoaderDialog: public Ui_GvQDataLoaderDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQDATALOADERDIALOG_H
