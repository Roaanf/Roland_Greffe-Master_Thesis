/********************************************************************************
** Form generated from reading UI file 'GvvQCacheEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQCACHEEDITOR_H
#define UI_GVVQCACHEEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFormLayout>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QTableWidget>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQCacheEditor
{
public:
    QVBoxLayout *verticalLayout_3;
    QGroupBox *_dataTypeGroupBox;
    QVBoxLayout *verticalLayout;
    QTableWidget *_dataTableWidget;
    QGroupBox *groupBox_6;
    QGridLayout *gridLayout_5;
    QLabel *label_7;
    QLabel *label_8;
    QLineEdit *_nodeCacheMemoryLineEdit;
    QLabel *label_9;
    QLineEdit *_nodeCacheCapacityLineEdit;
    QLineEdit *_nodeTileResolutionLineEdit;
    QGroupBox *groupBox_7;
    QGridLayout *gridLayout_6;
    QLabel *label_10;
    QLabel *label_11;
    QLineEdit *_brickBorderSizeLineEdit;
    QLabel *label_12;
    QLineEdit *_brickCacheMemoryLineEdit;
    QLabel *label_13;
    QLineEdit *_brickCacheCapacityLineEdit;
    QLineEdit *_brickResolutionLineEdit;
    QGroupBox *groupBox;
    QVBoxLayout *verticalLayout_2;
    QCheckBox *_preventReplacingUsedElementsCachePolicyCheckBox;
    QGroupBox *_smoothLoadingCachePolicyGroupBox;
    QFormLayout *formLayout_2;
    QLabel *label;
    QSpinBox *_nbSubdivisionsSpinBox;
    QLabel *label_2;
    QSpinBox *_nbLoadsSpinBox;
    QGroupBox *_timeLimitGroupBox;
    QFormLayout *formLayout;
    QLabel *label_3;
    QDoubleSpinBox *_timeLimitDoubleSpinBox;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQCacheEditor)
    {
        if (GvvQCacheEditor->objectName().isEmpty())
            GvvQCacheEditor->setObjectName(QString::fromUtf8("GvvQCacheEditor"));
        GvvQCacheEditor->resize(270, 772);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(GvvQCacheEditor->sizePolicy().hasHeightForWidth());
        GvvQCacheEditor->setSizePolicy(sizePolicy);
        verticalLayout_3 = new QVBoxLayout(GvvQCacheEditor);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        _dataTypeGroupBox = new QGroupBox(GvvQCacheEditor);
        _dataTypeGroupBox->setObjectName(QString::fromUtf8("_dataTypeGroupBox"));
        sizePolicy.setHeightForWidth(_dataTypeGroupBox->sizePolicy().hasHeightForWidth());
        _dataTypeGroupBox->setSizePolicy(sizePolicy);
        verticalLayout = new QVBoxLayout(_dataTypeGroupBox);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        _dataTableWidget = new QTableWidget(_dataTypeGroupBox);
        _dataTableWidget->setObjectName(QString::fromUtf8("_dataTableWidget"));
        QSizePolicy sizePolicy1(QSizePolicy::Expanding, QSizePolicy::Preferred);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(_dataTableWidget->sizePolicy().hasHeightForWidth());
        _dataTableWidget->setSizePolicy(sizePolicy1);
        _dataTableWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
        _dataTableWidget->setAlternatingRowColors(true);

        verticalLayout->addWidget(_dataTableWidget);


        verticalLayout_3->addWidget(_dataTypeGroupBox);

        groupBox_6 = new QGroupBox(GvvQCacheEditor);
        groupBox_6->setObjectName(QString::fromUtf8("groupBox_6"));
        gridLayout_5 = new QGridLayout(groupBox_6);
        gridLayout_5->setObjectName(QString::fromUtf8("gridLayout_5"));
        label_7 = new QLabel(groupBox_6);
        label_7->setObjectName(QString::fromUtf8("label_7"));

        gridLayout_5->addWidget(label_7, 0, 0, 1, 1);

        label_8 = new QLabel(groupBox_6);
        label_8->setObjectName(QString::fromUtf8("label_8"));

        gridLayout_5->addWidget(label_8, 1, 0, 1, 1);

        _nodeCacheMemoryLineEdit = new QLineEdit(groupBox_6);
        _nodeCacheMemoryLineEdit->setObjectName(QString::fromUtf8("_nodeCacheMemoryLineEdit"));
        _nodeCacheMemoryLineEdit->setLayoutDirection(Qt::LeftToRight);
        _nodeCacheMemoryLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeCacheMemoryLineEdit->setReadOnly(true);

        gridLayout_5->addWidget(_nodeCacheMemoryLineEdit, 1, 1, 1, 1);

        label_9 = new QLabel(groupBox_6);
        label_9->setObjectName(QString::fromUtf8("label_9"));

        gridLayout_5->addWidget(label_9, 2, 0, 1, 1);

        _nodeCacheCapacityLineEdit = new QLineEdit(groupBox_6);
        _nodeCacheCapacityLineEdit->setObjectName(QString::fromUtf8("_nodeCacheCapacityLineEdit"));
        _nodeCacheCapacityLineEdit->setLayoutDirection(Qt::LeftToRight);
        _nodeCacheCapacityLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeCacheCapacityLineEdit->setReadOnly(true);

        gridLayout_5->addWidget(_nodeCacheCapacityLineEdit, 2, 1, 1, 1);

        _nodeTileResolutionLineEdit = new QLineEdit(groupBox_6);
        _nodeTileResolutionLineEdit->setObjectName(QString::fromUtf8("_nodeTileResolutionLineEdit"));
        _nodeTileResolutionLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nodeTileResolutionLineEdit->setReadOnly(true);

        gridLayout_5->addWidget(_nodeTileResolutionLineEdit, 0, 1, 1, 1);


        verticalLayout_3->addWidget(groupBox_6);

        groupBox_7 = new QGroupBox(GvvQCacheEditor);
        groupBox_7->setObjectName(QString::fromUtf8("groupBox_7"));
        gridLayout_6 = new QGridLayout(groupBox_7);
        gridLayout_6->setObjectName(QString::fromUtf8("gridLayout_6"));
        label_10 = new QLabel(groupBox_7);
        label_10->setObjectName(QString::fromUtf8("label_10"));

        gridLayout_6->addWidget(label_10, 0, 0, 1, 1);

        label_11 = new QLabel(groupBox_7);
        label_11->setObjectName(QString::fromUtf8("label_11"));

        gridLayout_6->addWidget(label_11, 1, 0, 1, 1);

        _brickBorderSizeLineEdit = new QLineEdit(groupBox_7);
        _brickBorderSizeLineEdit->setObjectName(QString::fromUtf8("_brickBorderSizeLineEdit"));
        _brickBorderSizeLineEdit->setLayoutDirection(Qt::LeftToRight);
        _brickBorderSizeLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _brickBorderSizeLineEdit->setReadOnly(true);

        gridLayout_6->addWidget(_brickBorderSizeLineEdit, 1, 1, 1, 1);

        label_12 = new QLabel(groupBox_7);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        gridLayout_6->addWidget(label_12, 2, 0, 1, 1);

        _brickCacheMemoryLineEdit = new QLineEdit(groupBox_7);
        _brickCacheMemoryLineEdit->setObjectName(QString::fromUtf8("_brickCacheMemoryLineEdit"));
        _brickCacheMemoryLineEdit->setLayoutDirection(Qt::LeftToRight);
        _brickCacheMemoryLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _brickCacheMemoryLineEdit->setReadOnly(true);

        gridLayout_6->addWidget(_brickCacheMemoryLineEdit, 2, 1, 1, 1);

        label_13 = new QLabel(groupBox_7);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        gridLayout_6->addWidget(label_13, 3, 0, 1, 1);

        _brickCacheCapacityLineEdit = new QLineEdit(groupBox_7);
        _brickCacheCapacityLineEdit->setObjectName(QString::fromUtf8("_brickCacheCapacityLineEdit"));
        _brickCacheCapacityLineEdit->setLayoutDirection(Qt::LeftToRight);
        _brickCacheCapacityLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _brickCacheCapacityLineEdit->setReadOnly(true);

        gridLayout_6->addWidget(_brickCacheCapacityLineEdit, 3, 1, 1, 1);

        _brickResolutionLineEdit = new QLineEdit(groupBox_7);
        _brickResolutionLineEdit->setObjectName(QString::fromUtf8("_brickResolutionLineEdit"));
        _brickResolutionLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _brickResolutionLineEdit->setReadOnly(true);

        gridLayout_6->addWidget(_brickResolutionLineEdit, 0, 1, 1, 1);


        verticalLayout_3->addWidget(groupBox_7);

        groupBox = new QGroupBox(GvvQCacheEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        verticalLayout_2 = new QVBoxLayout(groupBox);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        _preventReplacingUsedElementsCachePolicyCheckBox = new QCheckBox(groupBox);
        _preventReplacingUsedElementsCachePolicyCheckBox->setObjectName(QString::fromUtf8("_preventReplacingUsedElementsCachePolicyCheckBox"));

        verticalLayout_2->addWidget(_preventReplacingUsedElementsCachePolicyCheckBox);

        _smoothLoadingCachePolicyGroupBox = new QGroupBox(groupBox);
        _smoothLoadingCachePolicyGroupBox->setObjectName(QString::fromUtf8("_smoothLoadingCachePolicyGroupBox"));
        _smoothLoadingCachePolicyGroupBox->setAlignment(Qt::AlignLeading|Qt::AlignLeft|Qt::AlignVCenter);
        _smoothLoadingCachePolicyGroupBox->setFlat(false);
        _smoothLoadingCachePolicyGroupBox->setCheckable(true);
        _smoothLoadingCachePolicyGroupBox->setChecked(false);
        formLayout_2 = new QFormLayout(_smoothLoadingCachePolicyGroupBox);
        formLayout_2->setObjectName(QString::fromUtf8("formLayout_2"));
        label = new QLabel(_smoothLoadingCachePolicyGroupBox);
        label->setObjectName(QString::fromUtf8("label"));

        formLayout_2->setWidget(0, QFormLayout::LabelRole, label);

        _nbSubdivisionsSpinBox = new QSpinBox(_smoothLoadingCachePolicyGroupBox);
        _nbSubdivisionsSpinBox->setObjectName(QString::fromUtf8("_nbSubdivisionsSpinBox"));
        _nbSubdivisionsSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nbSubdivisionsSpinBox->setMaximum(10000);
        _nbSubdivisionsSpinBox->setSingleStep(100);

        formLayout_2->setWidget(0, QFormLayout::FieldRole, _nbSubdivisionsSpinBox);

        label_2 = new QLabel(_smoothLoadingCachePolicyGroupBox);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        formLayout_2->setWidget(1, QFormLayout::LabelRole, label_2);

        _nbLoadsSpinBox = new QSpinBox(_smoothLoadingCachePolicyGroupBox);
        _nbLoadsSpinBox->setObjectName(QString::fromUtf8("_nbLoadsSpinBox"));
        _nbLoadsSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nbLoadsSpinBox->setMaximum(10000);
        _nbLoadsSpinBox->setSingleStep(100);

        formLayout_2->setWidget(1, QFormLayout::FieldRole, _nbLoadsSpinBox);


        verticalLayout_2->addWidget(_smoothLoadingCachePolicyGroupBox);

        _timeLimitGroupBox = new QGroupBox(groupBox);
        _timeLimitGroupBox->setObjectName(QString::fromUtf8("_timeLimitGroupBox"));
        _timeLimitGroupBox->setFocusPolicy(Qt::StrongFocus);
        _timeLimitGroupBox->setCheckable(true);
        _timeLimitGroupBox->setChecked(false);
        formLayout = new QFormLayout(_timeLimitGroupBox);
        formLayout->setObjectName(QString::fromUtf8("formLayout"));
        label_3 = new QLabel(_timeLimitGroupBox);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        formLayout->setWidget(0, QFormLayout::LabelRole, label_3);

        _timeLimitDoubleSpinBox = new QDoubleSpinBox(_timeLimitGroupBox);
        _timeLimitDoubleSpinBox->setObjectName(QString::fromUtf8("_timeLimitDoubleSpinBox"));
        _timeLimitDoubleSpinBox->setMaximum(1000);
        _timeLimitDoubleSpinBox->setValue(10);

        formLayout->setWidget(0, QFormLayout::FieldRole, _timeLimitDoubleSpinBox);


        verticalLayout_2->addWidget(_timeLimitGroupBox);


        verticalLayout_3->addWidget(groupBox);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout_3->addItem(verticalSpacer);

        groupBox->raise();
        groupBox_6->raise();
        groupBox_7->raise();
        _dataTypeGroupBox->raise();

        retranslateUi(GvvQCacheEditor);

        QMetaObject::connectSlotsByName(GvvQCacheEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQCacheEditor)
    {
        GvvQCacheEditor->setWindowTitle(QApplication::translate("GvvQCacheEditor", "Cache Editor", 0, QApplication::UnicodeUTF8));
        _dataTypeGroupBox->setTitle(QApplication::translate("GvvQCacheEditor", "Data", 0, QApplication::UnicodeUTF8));
        groupBox_6->setTitle(QApplication::translate("GvvQCacheEditor", "Nodes Cache", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("GvvQCacheEditor", "Resolution", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("GvvQCacheEditor", "Memory", 0, QApplication::UnicodeUTF8));
        _nodeCacheMemoryLineEdit->setText(QString());
        label_9->setText(QApplication::translate("GvvQCacheEditor", "Capacity", 0, QApplication::UnicodeUTF8));
        _nodeCacheCapacityLineEdit->setText(QString());
        groupBox_7->setTitle(QApplication::translate("GvvQCacheEditor", "Bricks Cache", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("GvvQCacheEditor", "Resolution", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("GvvQCacheEditor", "Border", 0, QApplication::UnicodeUTF8));
        _brickBorderSizeLineEdit->setText(QApplication::translate("GvvQCacheEditor", "1", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("GvvQCacheEditor", "Memory", 0, QApplication::UnicodeUTF8));
        _brickCacheMemoryLineEdit->setText(QString());
        label_13->setText(QApplication::translate("GvvQCacheEditor", "Capacity", 0, QApplication::UnicodeUTF8));
        _brickCacheCapacityLineEdit->setText(QString());
        groupBox->setTitle(QApplication::translate("GvvQCacheEditor", "Cache Policy", 0, QApplication::UnicodeUTF8));
        _preventReplacingUsedElementsCachePolicyCheckBox->setText(QApplication::translate("GvvQCacheEditor", "Don't Replace Used Elements", 0, QApplication::UnicodeUTF8));
        _smoothLoadingCachePolicyGroupBox->setTitle(QApplication::translate("GvvQCacheEditor", "Smooth Loading", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvvQCacheEditor", "Nb Max Subdivisions", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("GvvQCacheEditor", "Nb Max Loads", 0, QApplication::UnicodeUTF8));
        _timeLimitGroupBox->setTitle(QApplication::translate("GvvQCacheEditor", "Limit production time", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQCacheEditor", "Time Limit (ms)", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQCacheEditor: public Ui_GvvQCacheEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQCACHEEDITOR_H
