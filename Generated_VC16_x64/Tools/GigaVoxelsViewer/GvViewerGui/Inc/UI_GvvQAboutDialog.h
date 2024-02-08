/********************************************************************************
** Form generated from reading UI file 'GvvQAboutDialog.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQABOUTDIALOG_H
#define UI_GVVQABOUTDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>

QT_BEGIN_NAMESPACE

class Ui_GvvQAboutDialog
{
public:
    QGridLayout *gridLayout;
    QLabel *label;
    QLabel *label_2;
    QLabel *label_3;
    QLabel *label_4;
    QLabel *label_5;
    QSpacerItem *verticalSpacer;
    QPushButton *_creditsPushButton;
    QPushButton *_licensePushButton;
    QSpacerItem *horizontalSpacer;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *GvvQAboutDialog)
    {
        if (GvvQAboutDialog->objectName().isEmpty())
            GvvQAboutDialog->setObjectName(QString::fromUtf8("GvvQAboutDialog"));
        GvvQAboutDialog->resize(322, 226);
        GvvQAboutDialog->setSizeGripEnabled(false);
        gridLayout = new QGridLayout(GvvQAboutDialog);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(GvvQAboutDialog);
        label->setObjectName(QString::fromUtf8("label"));
        label->setPixmap(QPixmap(QString::fromUtf8(":/icons/Icons/GigaVoxelsLogo_div2.png")));
        label->setScaledContents(true);

        gridLayout->addWidget(label, 0, 0, 1, 4);

        label_2 = new QLabel(GvvQAboutDialog);
        label_2->setObjectName(QString::fromUtf8("label_2"));
        label_2->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_2, 1, 0, 1, 4);

        label_3 = new QLabel(GvvQAboutDialog);
        label_3->setObjectName(QString::fromUtf8("label_3"));
        label_3->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_3, 2, 0, 1, 4);

        label_4 = new QLabel(GvvQAboutDialog);
        label_4->setObjectName(QString::fromUtf8("label_4"));
        label_4->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_4, 3, 0, 1, 4);

        label_5 = new QLabel(GvvQAboutDialog);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setAlignment(Qt::AlignCenter);

        gridLayout->addWidget(label_5, 4, 0, 1, 4);

        verticalSpacer = new QSpacerItem(20, 2, QSizePolicy::Minimum, QSizePolicy::Expanding);

        gridLayout->addItem(verticalSpacer, 5, 2, 1, 1);

        _creditsPushButton = new QPushButton(GvvQAboutDialog);
        _creditsPushButton->setObjectName(QString::fromUtf8("_creditsPushButton"));

        gridLayout->addWidget(_creditsPushButton, 6, 0, 1, 1);

        _licensePushButton = new QPushButton(GvvQAboutDialog);
        _licensePushButton->setObjectName(QString::fromUtf8("_licensePushButton"));

        gridLayout->addWidget(_licensePushButton, 6, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(52, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 6, 2, 1, 1);

        buttonBox = new QDialogButtonBox(GvvQAboutDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Close);
        buttonBox->setCenterButtons(false);

        gridLayout->addWidget(buttonBox, 6, 3, 1, 1);


        retranslateUi(GvvQAboutDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GvvQAboutDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GvvQAboutDialog, SLOT(reject()));

        QMetaObject::connectSlotsByName(GvvQAboutDialog);
    } // setupUi

    void retranslateUi(QDialog *GvvQAboutDialog)
    {
        GvvQAboutDialog->setWindowTitle(QApplication::translate("GvvQAboutDialog", "About GigaVoxels", 0, QApplication::UnicodeUTF8));
        label->setText(QString());
        label_2->setText(QApplication::translate("GvvQAboutDialog", "Version 1.0.0", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQAboutDialog", "Efficient rendering of highly detailed volumetric scenes", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQAboutDialog", "Copyright \302\251 2011-2015", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("GvvQAboutDialog", "CNRS, INRIA, LJK", 0, QApplication::UnicodeUTF8));
        _creditsPushButton->setText(QApplication::translate("GvvQAboutDialog", "Credits", 0, QApplication::UnicodeUTF8));
        _licensePushButton->setText(QApplication::translate("GvvQAboutDialog", "License", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQAboutDialog: public Ui_GvvQAboutDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQABOUTDIALOG_H
