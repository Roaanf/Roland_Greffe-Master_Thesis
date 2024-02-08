/********************************************************************************
** Form generated from reading UI file 'GvvQTransformationEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQTRANSFORMATIONEDITOR_H
#define UI_GVVQTRANSFORMATIONEDITOR_H

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
#include <QtGui/QSpacerItem>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQTransformationEditor
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_9;
    QVBoxLayout *verticalLayout_2;
    QGroupBox *groupBox_7;
    QHBoxLayout *horizontalLayout;
    QLabel *label_8;
    QDoubleSpinBox *_xTranslationSpinBox;
    QLabel *label_9;
    QDoubleSpinBox *_yTranslationSpinBox;
    QLabel *label_10;
    QDoubleSpinBox *_zTranslationSpinBox;
    QGroupBox *groupBox_6;
    QGridLayout *gridLayout_5;
    QLabel *label_11;
    QDoubleSpinBox *_xRotationSpinBox;
    QLabel *label_12;
    QDoubleSpinBox *_yRotationSpinBox;
    QLabel *label_13;
    QDoubleSpinBox *_zRotationSpinBox;
    QSpacerItem *horizontalSpacer;
    QLabel *label_14;
    QDoubleSpinBox *_angleRotationSpinBox;
    QGroupBox *groupBox_8;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label_15;
    QDoubleSpinBox *_uniformScaleSpinBox;
    QSpacerItem *horizontalSpacer_2;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQTransformationEditor)
    {
        if (GvvQTransformationEditor->objectName().isEmpty())
            GvvQTransformationEditor->setObjectName(QString::fromUtf8("GvvQTransformationEditor"));
        GvvQTransformationEditor->resize(316, 344);
        verticalLayout = new QVBoxLayout(GvvQTransformationEditor);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox_9 = new QGroupBox(GvvQTransformationEditor);
        groupBox_9->setObjectName(QString::fromUtf8("groupBox_9"));
        verticalLayout_2 = new QVBoxLayout(groupBox_9);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        groupBox_7 = new QGroupBox(groupBox_9);
        groupBox_7->setObjectName(QString::fromUtf8("groupBox_7"));
        horizontalLayout = new QHBoxLayout(groupBox_7);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        label_8 = new QLabel(groupBox_7);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        horizontalLayout->addWidget(label_8);

        _xTranslationSpinBox = new QDoubleSpinBox(groupBox_7);
        _xTranslationSpinBox->setObjectName(QString::fromUtf8("_xTranslationSpinBox"));
        _xTranslationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _xTranslationSpinBox->setMinimum(-100);
        _xTranslationSpinBox->setMaximum(100);

        horizontalLayout->addWidget(_xTranslationSpinBox);

        label_9 = new QLabel(groupBox_7);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        horizontalLayout->addWidget(label_9);

        _yTranslationSpinBox = new QDoubleSpinBox(groupBox_7);
        _yTranslationSpinBox->setObjectName(QString::fromUtf8("_yTranslationSpinBox"));
        _yTranslationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _yTranslationSpinBox->setMinimum(-100);
        _yTranslationSpinBox->setMaximum(100);

        horizontalLayout->addWidget(_yTranslationSpinBox);

        label_10 = new QLabel(groupBox_7);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        horizontalLayout->addWidget(label_10);

        _zTranslationSpinBox = new QDoubleSpinBox(groupBox_7);
        _zTranslationSpinBox->setObjectName(QString::fromUtf8("_zTranslationSpinBox"));
        _zTranslationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _zTranslationSpinBox->setMinimum(-100);
        _zTranslationSpinBox->setMaximum(100);

        horizontalLayout->addWidget(_zTranslationSpinBox);


        verticalLayout_2->addWidget(groupBox_7);

        groupBox_6 = new QGroupBox(groupBox_9);
        groupBox_6->setObjectName(QString::fromUtf8("groupBox_6"));
        gridLayout_5 = new QGridLayout(groupBox_6);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        label_11 = new QLabel(groupBox_6);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_5->addWidget(label_11, 0, 0, 1, 1);

        _xRotationSpinBox = new QDoubleSpinBox(groupBox_6);
        _xRotationSpinBox->setObjectName(QString::fromUtf8("_xRotationSpinBox"));
        _xRotationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _xRotationSpinBox->setMinimum(-1);
        _xRotationSpinBox->setMaximum(1);

        gridLayout_5->addWidget(_xRotationSpinBox, 0, 1, 1, 1);

        label_12 = new QLabel(groupBox_6);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout_5->addWidget(label_12, 0, 2, 1, 1);

        _yRotationSpinBox = new QDoubleSpinBox(groupBox_6);
        _yRotationSpinBox->setObjectName(QString::fromUtf8("_yRotationSpinBox"));
        _yRotationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _yRotationSpinBox->setMinimum(-1);
        _yRotationSpinBox->setMaximum(1);

        gridLayout_5->addWidget(_yRotationSpinBox, 0, 3, 1, 1);

        label_13 = new QLabel(groupBox_6);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_5->addWidget(label_13, 0, 4, 1, 1);

        _zRotationSpinBox = new QDoubleSpinBox(groupBox_6);
        _zRotationSpinBox->setObjectName(QString::fromUtf8("_zRotationSpinBox"));
        _zRotationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _zRotationSpinBox->setMinimum(-1);
        _zRotationSpinBox->setMaximum(1);

        gridLayout_5->addWidget(_zRotationSpinBox, 0, 5, 1, 1);

        horizontalSpacer = new QSpacerItem(141, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_5->addItem(horizontalSpacer, 1, 0, 1, 4);

        label_14 = new QLabel(groupBox_6);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout_5->addWidget(label_14, 1, 4, 1, 1);

        _angleRotationSpinBox = new QDoubleSpinBox(groupBox_6);
        _angleRotationSpinBox->setObjectName(QString::fromUtf8("_angleRotationSpinBox"));
        _angleRotationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _angleRotationSpinBox->setMinimum(-360);
        _angleRotationSpinBox->setMaximum(360);

        gridLayout_5->addWidget(_angleRotationSpinBox, 1, 5, 1, 1);


        verticalLayout_2->addWidget(groupBox_6);

        groupBox_8 = new QGroupBox(groupBox_9);
        groupBox_8->setObjectName(QString::fromUtf8("groupBox_8"));
        horizontalLayout_2 = new QHBoxLayout(groupBox_8);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label_15 = new QLabel(groupBox_8);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        horizontalLayout_2->addWidget(label_15);

        _uniformScaleSpinBox = new QDoubleSpinBox(groupBox_8);
        _uniformScaleSpinBox->setObjectName(QString::fromUtf8("_uniformScaleSpinBox"));
        _uniformScaleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _uniformScaleSpinBox->setMinimum(-100);
        _uniformScaleSpinBox->setMaximum(100);
        _uniformScaleSpinBox->setSingleStep(0.1);

        horizontalLayout_2->addWidget(_uniformScaleSpinBox);

        horizontalSpacer_2 = new QSpacerItem(124, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_2->addItem(horizontalSpacer_2);


        verticalLayout_2->addWidget(groupBox_8);


        verticalLayout->addWidget(groupBox_9);

        verticalSpacer = new QSpacerItem(20, 119, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(GvvQTransformationEditor);

        QMetaObject::connectSlotsByName(GvvQTransformationEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQTransformationEditor)
    {
        GvvQTransformationEditor->setWindowTitle(QApplication::translate("GvvQTransformationEditor", "Form", 0, QApplication::UnicodeUTF8));
        groupBox_9->setTitle(QString());
        groupBox_7->setTitle(QApplication::translate("GvvQTransformationEditor", "Translation", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("GvvQTransformationEditor", "x", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("GvvQTransformationEditor", "y", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("GvvQTransformationEditor", "z", 0, QApplication::UnicodeUTF8));
        groupBox_6->setTitle(QApplication::translate("GvvQTransformationEditor", "Rotation", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("GvvQTransformationEditor", "x", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("GvvQTransformationEditor", "y", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("GvvQTransformationEditor", "z", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("GvvQTransformationEditor", "angle", 0, QApplication::UnicodeUTF8));
        groupBox_8->setTitle(QApplication::translate("GvvQTransformationEditor", "Scale", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("GvvQTransformationEditor", "Uniform", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQTransformationEditor: public Ui_GvvQTransformationEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQTRANSFORMATIONEDITOR_H
