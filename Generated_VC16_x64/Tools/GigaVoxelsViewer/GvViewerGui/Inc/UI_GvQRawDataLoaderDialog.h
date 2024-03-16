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
    QLabel *label_2;
    QSpinBox *_radius;
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

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 2, 0, 1, 1);

        _radius = new QSpinBox(groupBox);
        _radius->setObjectName(QString::fromUtf8("_radius"));
        _radius->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _radius->setMinimum(0);
        _radius->setMaximum(1e+06);

        gridLayout->addWidget(_radius, 2, 1, 1, 1);

        label_21 = new QLabel(groupBox);
        label_21->setObjectName(QString::fromUtf8("label_21"));

        gridLayout->addWidget(label_21, 3, 0, 1, 1);

        _trueXSpinBox = new QSpinBox(groupBox);
        _trueXSpinBox->setObjectName(QString::fromUtf8("_trueXSpinBox"));
        _trueXSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueXSpinBox->setMinimum(0);
        _trueXSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueXSpinBox, 3, 1, 1, 1);

        _trueYSpinBox = new QSpinBox(groupBox);
        _trueYSpinBox->setObjectName(QString::fromUtf8("_trueYSpinBox"));
        _trueYSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueYSpinBox->setMinimum(0);
        _trueYSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueYSpinBox, 3, 2, 1, 1);

        _trueZSpinBox = new QSpinBox(groupBox);
        _trueZSpinBox->setObjectName(QString::fromUtf8("_trueZSpinBox"));
        _trueZSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _trueZSpinBox->setMinimum(0);
        _trueZSpinBox->setMaximum(1e+06);

        gridLayout->addWidget(_trueZSpinBox, 3, 3, 1, 1);


        verticalLayout->addWidget(groupBox);

        buttonBox = new QDialogButtonBox(GvQRawDataLoaderDialog);
        buttonBox->setObjectName(QString::fromUtf8("buttonBox"));
        buttonBox->setOrientation(Qt::Horizontal);
        buttonBox->setStandardButtons(QDialogButtonBox::Cancel|QDialogButtonBox::Ok);

        verticalLayout->addWidget(buttonBox);


        retranslateUi(GvQRawDataLoaderDialog);
        QObject::connect(buttonBox, SIGNAL(accepted()), GvQRawDataLoaderDialog, SLOT(accept()));
        QObject::connect(buttonBox, SIGNAL(rejected()), GvQRawDataLoaderDialog, SLOT(reject()));

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
        label_2->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "Radius to remove.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_2->setText(QApplication::translate("GvQRawDataLoaderDialog", "Radius", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _radius->setToolTip(QApplication::translate("GvQRawDataLoaderDialog", "\n"
"              defines the outer radius that will be raplced by empty voxels -> 0 does nothing.\n"
"            ", 0, QApplication::UnicodeUTF8));
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
