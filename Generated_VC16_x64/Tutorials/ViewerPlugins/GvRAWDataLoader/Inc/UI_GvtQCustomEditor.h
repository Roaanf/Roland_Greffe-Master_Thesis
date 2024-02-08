/********************************************************************************
** Form generated from reading UI file 'GvtQCustomEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVTQCUSTOMEDITOR_H
#define UI_GVTQCUSTOMEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvtQCustomEditor
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout_2;
    QLabel *label_3;
    QLineEdit *_filenameLineEdit;
    QLabel *label_4;
    QLineEdit *_dataTypeLineEdit;
    QLabel *label;
    QLineEdit *_minDataValueLineEdit;
    QLabel *label_2;
    QLineEdit *_maxDataValueLineEdit;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout;
    QLabel *label_9;
    QDoubleSpinBox *_producerThresholdDoubleSpinBoxLow;
    QDoubleSpinBox *_producerThresholdDoubleSpinBoxHigh;
    QLabel *label_8;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout_1;
    QLabel *label_91;
    QDoubleSpinBox *_shaderThresholdDoubleSpinBoxLow;
    QLabel *label_92;
    QDoubleSpinBox *_shaderThresholdDoubleSpinBoxHigh;
    QDoubleSpinBox *_shaderFullOpacityDistanceDoubleSpinBox;
    QLabel *label_81;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvtQCustomEditor)
    {
        if (GvtQCustomEditor->objectName().isEmpty())
            GvtQCustomEditor->setObjectName(QString::fromUtf8("GvtQCustomEditor"));
        GvtQCustomEditor->resize(216, 331);
        verticalLayout = new QVBoxLayout(GvtQCustomEditor);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(GvtQCustomEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout_2 = new QGridLayout(groupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 0, 0, 1, 1);

        _filenameLineEdit = new QLineEdit(groupBox);
        _filenameLineEdit->setObjectName(QString::fromUtf8("_filenameLineEdit"));
        _filenameLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _filenameLineEdit->setReadOnly(true);

        gridLayout_2->addWidget(_filenameLineEdit, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 1, 0, 1, 1);

        _dataTypeLineEdit = new QLineEdit(groupBox);
        _dataTypeLineEdit->setObjectName(QString::fromUtf8("_dataTypeLineEdit"));
        _dataTypeLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);

        gridLayout_2->addWidget(_dataTypeLineEdit, 1, 1, 1, 1);

        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 2, 0, 1, 1);

        _minDataValueLineEdit = new QLineEdit(groupBox);
        _minDataValueLineEdit->setObjectName(QString::fromUtf8("_minDataValueLineEdit"));
        _minDataValueLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _minDataValueLineEdit->setReadOnly(true);

        gridLayout_2->addWidget(_minDataValueLineEdit, 2, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout_2->addWidget(label_2, 3, 0, 1, 1);

        _maxDataValueLineEdit = new QLineEdit(groupBox);
        _maxDataValueLineEdit->setObjectName(QString::fromUtf8("_maxDataValueLineEdit"));
        _maxDataValueLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _maxDataValueLineEdit->setReadOnly(true);

        gridLayout_2->addWidget(_maxDataValueLineEdit, 3, 1, 1, 1);


        verticalLayout->addWidget(groupBox);

        groupBox_4 = new QGroupBox(GvtQCustomEditor);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        gridLayout = new QGridLayout(groupBox_4);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_9 = new QLabel(groupBox_4);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout->addWidget(label_9, 1, 0, 1, 1);

        _producerThresholdDoubleSpinBoxLow = new QDoubleSpinBox(groupBox_4);
        _producerThresholdDoubleSpinBoxLow->setObjectName(QString::fromUtf8("_producerThresholdDoubleSpinBoxLow"));
        _producerThresholdDoubleSpinBoxLow->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _producerThresholdDoubleSpinBoxLow->setMinimum(-1e+06);
        _producerThresholdDoubleSpinBoxLow->setMaximum(1e+06);
        _producerThresholdDoubleSpinBoxLow->setSingleStep(1);

        gridLayout->addWidget(_producerThresholdDoubleSpinBoxLow, 1, 1, 1, 1);

        _producerThresholdDoubleSpinBoxHigh = new QDoubleSpinBox(groupBox_4);
        _producerThresholdDoubleSpinBoxHigh->setObjectName(QString::fromUtf8("_producerThresholdDoubleSpinBoxHigh"));
        _producerThresholdDoubleSpinBoxHigh->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _producerThresholdDoubleSpinBoxHigh->setMinimum(-1e+06);
        _producerThresholdDoubleSpinBoxHigh->setMaximum(1e+06);
        _producerThresholdDoubleSpinBoxHigh->setSingleStep(1);

        gridLayout->addWidget(_producerThresholdDoubleSpinBoxHigh, 3, 1, 1, 1);

        label_8 = new QLabel(groupBox_4);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout->addWidget(label_8, 3, 0, 1, 1);


        verticalLayout->addWidget(groupBox_4);

        groupBox_3 = new QGroupBox(GvtQCustomEditor);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout_1 = new QGridLayout(groupBox_3);
        gridLayout_1->setObjectName(QString::fromUtf8("gridLayout_1"));
        label_91 = new QLabel(groupBox_3);
        label_91->setObjectName(QString::fromUtf8("label_91"));

        gridLayout_1->addWidget(label_91, 1, 0, 1, 1);

        _shaderThresholdDoubleSpinBoxLow = new QDoubleSpinBox(groupBox_3);
        _shaderThresholdDoubleSpinBoxLow->setObjectName(QString::fromUtf8("_shaderThresholdDoubleSpinBoxLow"));
        _shaderThresholdDoubleSpinBoxLow->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _shaderThresholdDoubleSpinBoxLow->setMinimum(0);
        _shaderThresholdDoubleSpinBoxLow->setMaximum(1000);
        _shaderThresholdDoubleSpinBoxLow->setSingleStep(1);

        gridLayout_1->addWidget(_shaderThresholdDoubleSpinBoxLow, 1, 1, 1, 1);

        label_92 = new QLabel(groupBox_3);
        label_92->setObjectName(QString::fromUtf8("label_92"));

        gridLayout_1->addWidget(label_92, 2, 0, 1, 1);

        _shaderThresholdDoubleSpinBoxHigh = new QDoubleSpinBox(groupBox_3);
        _shaderThresholdDoubleSpinBoxHigh->setObjectName(QString::fromUtf8("_shaderThresholdDoubleSpinBoxHigh"));
        _shaderThresholdDoubleSpinBoxHigh->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _shaderThresholdDoubleSpinBoxHigh->setMinimum(0);
        _shaderThresholdDoubleSpinBoxHigh->setMaximum(1000);
        _shaderThresholdDoubleSpinBoxHigh->setSingleStep(1);

        gridLayout_1->addWidget(_shaderThresholdDoubleSpinBoxHigh, 2, 1, 1, 1);

        _shaderFullOpacityDistanceDoubleSpinBox = new QDoubleSpinBox(groupBox_3);
        _shaderFullOpacityDistanceDoubleSpinBox->setObjectName(QString::fromUtf8("_shaderFullOpacityDistanceDoubleSpinBox"));
        _shaderFullOpacityDistanceDoubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _shaderFullOpacityDistanceDoubleSpinBox->setDecimals(2);
        _shaderFullOpacityDistanceDoubleSpinBox->setMinimum(0);
        _shaderFullOpacityDistanceDoubleSpinBox->setMaximum(1e+06);
        _shaderFullOpacityDistanceDoubleSpinBox->setSingleStep(10);

        gridLayout_1->addWidget(_shaderFullOpacityDistanceDoubleSpinBox, 3, 1, 1, 1);

        label_81 = new QLabel(groupBox_3);
        label_81->setObjectName(QString::fromUtf8("label_81"));

        gridLayout_1->addWidget(label_81, 3, 0, 1, 1);


        verticalLayout->addWidget(groupBox_3);

        verticalSpacer = new QSpacerItem(20, 122, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(GvtQCustomEditor);

        QMetaObject::connectSlotsByName(GvtQCustomEditor);
    } // setupUi

    void retranslateUi(QWidget *GvtQCustomEditor)
    {
        GvtQCustomEditor->setWindowTitle(QApplication::translate("GvtQCustomEditor", "Custom Editor", 0, QApplication::UnicodeUTF8));
        GvtQCustomEditor->setStyleSheet(QApplication::translate("GvtQCustomEditor", "QGroupBox\n"
"{\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                       stop: 0 #E0E0E0, stop: 1 #FFFFFF);\n"
"     border: 2px solid gray;\n"
"     border-radius: 5px;\n"
"     margin-top: 1ex; /* leave space at the top for the title */\n"
" }\n"
"\n"
" QGroupBox::title\n"
"{\n"
"	 subcontrol-origin: margin;\n"
"     subcontrol-position: top center; /* position at the top center */\n"
"     padding: 0 3px;\n"
"     background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,\n"
"                                       stop: 0 #FFOECE, stop: 1 #FFFFFF);\n"
" }\n"
"\n"
"QLabel\n"
"{\n"
"	border-radius: 4px;\n"
"	padding: 2px;\n"
" }\n"
"", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvtQCustomEditor", "Data", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvtQCustomEditor", "File", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvtQCustomEditor", "Data Type", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvtQCustomEditor", "Min", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("GvtQCustomEditor", "Max", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("GvtQCustomEditor", "Provider", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        label_9->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_9->setText(QApplication::translate("GvtQCustomEditor", "Threshold Low", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _producerThresholdDoubleSpinBoxLow->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        _producerThresholdDoubleSpinBoxHigh->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when streaming data from Host.\n"
"It can be used to optimize the internal data structure by flagging nodes as empty.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_8->setText(QApplication::translate("GvtQCustomEditor", "Threshold High", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("GvtQCustomEditor", "Shader", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        label_91->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when rendering bricks of voxels.\n"
"It ranges from 0 to 1.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_91->setText(QApplication::translate("GvtQCustomEditor", "Threshold", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _shaderThresholdDoubleSpinBoxLow->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when rendering bricks of voxels.\n"
"It ranges from 0 to 1.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
#ifndef QT_NO_TOOLTIP
        label_92->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when rendering bricks of voxels.\n"
"It ranges from 0 to 1.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_92->setText(QApplication::translate("GvtQCustomEditor", "Threshold", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _shaderThresholdDoubleSpinBoxHigh->setToolTip(QApplication::translate("GvtQCustomEditor", "Scalar value used when rendering bricks of voxels.\n"
"It ranges from 0 to 1.", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_81->setText(QApplication::translate("GvtQCustomEditor", "Opacity Distance", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvtQCustomEditor: public Ui_GvtQCustomEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVTQCUSTOMEDITOR_H
