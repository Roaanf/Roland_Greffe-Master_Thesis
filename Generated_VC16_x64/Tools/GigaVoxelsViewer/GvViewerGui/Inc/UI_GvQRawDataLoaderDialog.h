/********************************************************************************
** Form generated from reading UI file 'GvQRawDataLoaderDialog.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQRAWDATALOADERDIALOG_H
#define UI_GVQRAWDATALOADERDIALOG_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QComboBox>
#include <QtGui/QDialog>
#include <QtGui/QDialogButtonBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QToolButton>
#include <QtGui/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_GvQRawDataLoaderDialog
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *_3DModelFilenameLineEdit;
    QToolButton *_3DModelDirectoryToolButton;
    QLabel *label_5;
    QComboBox *_3DModelDataTypeComboBox;
    QSpacerItem *horizontalSpacer;
    QLabel *label_4;
    QComboBox *_maxResolutionComboBox;
    QLabel *label_2;
    QComboBox *_brickSize;
    QLabel *label_21;
    QSpinBox *_trueXSpinBox;
    QSpinBox *_trueYSpinBox;
    QSpinBox *_trueZSpinBox;
    QDialogButtonBox *buttonBox;

    void setupUi(QDialog *GvQRawDataLoaderDialog)
    {
        if (GvQRawDataLoaderDialog->objectName().isEmpty())
            GvQRawDataLoaderDialog->setObjectName(QString::fromUtf8("GvQRawDataLoaderDialog"));
        GvQRawDataLoaderDialog->resize(319, 252);
        verticalLayout = new QVBoxLayout(GvQRawDataLoaderDialog);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(GvQRawDataLoaderDialog);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        _3DModelFilenameLineEdit = new QLineEdit(groupBox);
        _3DModelFilenameLineEdit->setObjectName(QString::fromUtf8("_3DModelFilenameLineEdit"));
        _3DModelFilenameLineEdit->setReadOnly(true);

        gridLayout->addWidget(_3DModelFilenameLineEdit, 0, 1, 1, 3);

        _3DModelDirectoryToolButton = new QToolButton(groupBox);
        _3DModelDirectoryToolButton->setObjectName(QString::fromUtf8("_3DModelDirectoryToolButton"));
        QIcon icon;
        icon.addFile(QString::fromUtf8(":/icons/Icons/fileopen.png"), QSize(), QIcon::Normal, QIcon::Off);
        _3DModelDirectoryToolButton->setIcon(icon);

        gridLayout->addWidget(_3DModelDirectoryToolButton, 0, 4, 1, 1);

        label_5 = new QLabel(groupBox);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 1, 0, 1, 1);

        _3DModelDataTypeComboBox = new QComboBox(groupBox);
        _3DModelDataTypeComboBox->setObjectName(QString::fromUtf8("_3DModelDataTypeComboBox"));

        gridLayout->addWidget(_3DModelDataTypeComboBox, 1, 1, 1, 1);

        horizontalSpacer = new QSpacerItem(158, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout->addItem(horizontalSpacer, 1, 2, 1, 3);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 2, 0, 1, 1);

        _maxResolutionComboBox = new QComboBox(groupBox);
        _maxResolutionComboBox->setObjectName(QString::fromUtf8("_maxResolutionComboBox"));
        _maxResolutionComboBox->setLayoutDirection(Qt::RightToLeft);

        gridLayout->addWidget(_maxResolutionComboBox, 2, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 3, 0, 1, 1);

        _brickSize = new QComboBox(groupBox);
        _brickSize->setObjectName(QString::fromUtf8("_brickSize"));
        _brickSize->setLayoutDirection(Qt::RightToLeft);

        gridLayout->addWidget(_brickSize, 3, 1, 1, 1);

        label_21 = new QLabel(groupBox);
        label_21->setObjectName(QString::fromUtf8("label_21"));

        gridLayout->addWidget(label_21, 4, 0, 1, 1);

        _trueXSpinBox = new QSpinBox(groupBox);
        _trueXSpinBox->setObjectName(QString::fromUtf8("_trueXSpinBox"));
        _trueXSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueXSpinBox->setMinimum(0);
        _trueXSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueXSpinBox, 4, 1, 1, 1);

        _trueYSpinBox = new QSpinBox(groupBox);
        _trueYSpinBox->setObjectName(QString::fromUtf8("_trueYSpinBox"));
        _trueYSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueYSpinBox->setMinimum(0);
        _trueYSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueYSpinBox, 4, 2, 1, 1);

        _trueZSpinBox = new QSpinBox(groupBox);
        _trueZSpinBox->setObjectName(QString::fromUtf8("_trueZSpinBox"));
        _trueZSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueZSpinBox->setMinimum(0);
        _trueZSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueZSpinBox, 4, 3, 1, 1);


        verticalLayout->addWidget(groupBox);

        buttonBox = new QDialogButtonBox(GvQRawDataLoaderDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(GvQRawDataLoaderDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GvQRawDataLoaderDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GvQRawDataLoaderDialog, SLOT(reject()));

        _maxResolutionComboBox->setCurrentIndex(0);
        _brickSize->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(GvQRawDataLoaderDialog);
    } // setupUi

    void retranslateUi(QDialog *GvQRawDataLoaderDialog)
    {
        GvQRawDataLoaderDialog->setWindowTitle(QApplication::translate("GvQRawDataLoaderDialog", "RAW Data Loader", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvQRawDataLoaderDialog", "3D Model", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvQRawDataLoaderDialog", "Filename", 0, QApplication::UnicodeUTF8));
        _3DModelDirectoryToolButton->setText(QApplication::translate("GvQRawDataLoaderDialog", "...", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("GvQRawDataLoaderDialog", "Data Type", 0, QApplication::UnicodeUTF8));
        _3DModelDataTypeComboBox->clear();
        _3DModelDataTypeComboBox->insertItems(0, QStringList()
         << QApplication::translate("GvQRawDataLoaderDialog", "USHORT", 0, QApplication::UnicodeUTF8)
        );
#ifndef QT_NO_TOOLTIP
        label_4->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Number of levels of resolution of the generated data strucutre.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_4->setText(QApplication::translate("GvQRawDataLoaderDialog", "Model Resolution", 0, QApplication::UnicodeUTF8));
        _maxResolutionComboBox->clear();
        _maxResolutionComboBox->insertItems(0, QStringList()
         << QApplication::translate("GvQRawDataLoaderDialog", "8", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "16", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "32", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "64", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "128", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "256", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "512", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "1024", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "2048", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "4096", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "8192", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "16384", 0, QApplication::UnicodeUTF8)
        );
#ifndef QT_NO_TOOLTIP
        _maxResolutionComboBox->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Number of levels of resolution of the generated data strucutre.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        label_2->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Number of levels of resolution of the generated data strucutre.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_2->setText(QApplication::translate("GvQRawDataLoaderDialog", "BrickRes", 0, QApplication::UnicodeUTF8));
        _brickSize->clear();
        _brickSize->insertItems(0, QStringList()
         << QApplication::translate("GvQRawDataLoaderDialog", "4", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "8", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "16", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "32", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "64", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "128", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("GvQRawDataLoaderDialog", "256", 0, QApplication::UnicodeUTF8)
        );
#ifndef QT_NO_TOOLTIP
        _brickSize->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Size of the brick", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        label_21->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "True res.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_21->setText(QApplication::translate("GvQRawDataLoaderDialog", "X/Y/Z", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _trueXSpinBox->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        _trueYSpinBox->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        _trueZSpinBox->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
    } // retranslateUi

};

namespace Ui {
    class GvQRawDataLoaderDialog: public Ui_GvQRawDataLoaderDialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQRAWDATALOADERDIALOG_H
