/********************************************************************************
** Form generated from reading UI file 'GvvQCameraEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQCAMERAEDITOR_H
#define UI_GVVQCAMERAEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQCameraEditor
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox;
    QGridLayout *gridLayout;
    QLabel *label;
    QDoubleSpinBox *_fieldOfViewDoubleSpinBox;
    QLabel *label_2;
    QDoubleSpinBox *_sceneRadiusDoubleSpinBox;
    QLabel *label_3;
    QDoubleSpinBox *_zNearCoefficientDoubleSpinBox;
    QLabel *label_4;
    QDoubleSpinBox *_zClippingCoefficientDoubleSpinBox;

    void setupUi(QWidget *GvvQCameraEditor)
    {
        if (GvvQCameraEditor->objectName().isEmpty())
            GvvQCameraEditor->setObjectName(QString::fromUtf8("GvvQCameraEditor"));
        GvvQCameraEditor->resize(247, 149);
        verticalLayout = new QVBoxLayout(GvvQCameraEditor);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox = new QGroupBox(GvvQCameraEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        gridLayout = new QGridLayout(groupBox);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        _fieldOfViewDoubleSpinBox = new QDoubleSpinBox(groupBox);
        _fieldOfViewDoubleSpinBox->setObjectName(QString::fromUtf8("_fieldOfViewDoubleSpinBox"));
        _fieldOfViewDoubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _fieldOfViewDoubleSpinBox->setDecimals(6);
        _fieldOfViewDoubleSpinBox->setMaximum(7);
        _fieldOfViewDoubleSpinBox->setSingleStep(0.1);
        _fieldOfViewDoubleSpinBox->setValue(0.785398);

        gridLayout->addWidget(_fieldOfViewDoubleSpinBox, 0, 1, 1, 1);

        label_2 = new QLabel(groupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 1, 0, 1, 1);

        _sceneRadiusDoubleSpinBox = new QDoubleSpinBox(groupBox);
        _sceneRadiusDoubleSpinBox->setObjectName(QString::fromUtf8("_sceneRadiusDoubleSpinBox"));
        _sceneRadiusDoubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _sceneRadiusDoubleSpinBox->setDecimals(6);
        _sceneRadiusDoubleSpinBox->setMaximum(100000);
        _sceneRadiusDoubleSpinBox->setValue(1);

        gridLayout->addWidget(_sceneRadiusDoubleSpinBox, 1, 1, 1, 1);

        label_3 = new QLabel(groupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout->addWidget(label_3, 2, 0, 1, 1);

        _zNearCoefficientDoubleSpinBox = new QDoubleSpinBox(groupBox);
        _zNearCoefficientDoubleSpinBox->setObjectName(QString::fromUtf8("_zNearCoefficientDoubleSpinBox"));
        _zNearCoefficientDoubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _zNearCoefficientDoubleSpinBox->setDecimals(6);
        _zNearCoefficientDoubleSpinBox->setMaximum(1);
        _zNearCoefficientDoubleSpinBox->setSingleStep(0.1);
        _zNearCoefficientDoubleSpinBox->setValue(0.005);

        gridLayout->addWidget(_zNearCoefficientDoubleSpinBox, 2, 1, 1, 1);

        label_4 = new QLabel(groupBox);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout->addWidget(label_4, 3, 0, 1, 1);

        _zClippingCoefficientDoubleSpinBox = new QDoubleSpinBox(groupBox);
        _zClippingCoefficientDoubleSpinBox->setObjectName(QString::fromUtf8("_zClippingCoefficientDoubleSpinBox"));
        _zClippingCoefficientDoubleSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _zClippingCoefficientDoubleSpinBox->setDecimals(6);
        _zClippingCoefficientDoubleSpinBox->setMaximum(10);
        _zClippingCoefficientDoubleSpinBox->setSingleStep(0.1);
        _zClippingCoefficientDoubleSpinBox->setValue(1.73205);

        gridLayout->addWidget(_zClippingCoefficientDoubleSpinBox, 3, 1, 1, 1);


        verticalLayout->addWidget(groupBox);


        retranslateUi(GvvQCameraEditor);

        QMetaObject::connectSlotsByName(GvvQCameraEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQCameraEditor)
    {
        GvvQCameraEditor->setWindowTitle(QApplication::translate("GvvQCameraEditor", "Form", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvvQCameraEditor", "Camera", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        label->setToolTip(QApplication::translate("GvvQCameraEditor", "Field of view", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label->setText(QApplication::translate("GvvQCameraEditor", "fieldOfView", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("GvvQCameraEditor", "sceneRadius", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQCameraEditor", "zNearCoefficient", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQCameraEditor", "zClippingCoefficient", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQCameraEditor: public Ui_GvvQCameraEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQCAMERAEDITOR_H
