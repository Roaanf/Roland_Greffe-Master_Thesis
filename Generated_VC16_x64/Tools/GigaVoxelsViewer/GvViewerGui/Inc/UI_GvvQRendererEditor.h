/********************************************************************************
** Form generated from reading UI file 'GvvQRendererEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVVQRENDEREREDITOR_H
#define UI_GVVQRENDEREREDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QRadioButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QSpinBox>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvvQRendererEditor
{
public:
    QVBoxLayout *verticalLayout;
    QGroupBox *groupBox_3;
    QGridLayout *gridLayout;
    QLabel *label_5;
    QSpinBox *_maxDepthSpinBox;
    QLineEdit *_maxResolutionLineEdit;
    QLabel *label_6;
    QSpinBox *_nbMaxRayCastingIterationSpinBox;
    QCheckBox *_dynamicUpdateCheckBox;
    QGroupBox *groupBox_4;
    QGridLayout *gridLayout_3;
    QRadioButton *_priorityOnNodesRadioButton;
    QRadioButton *_priorityOnBricksRadioButton;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout_7;
    QLabel *label_3;
    QSpinBox *_viewportXSpinBox;
    QLabel *label_4;
    QSpinBox *_viewportYSpinBox;
    QLabel *label_17;
    QSpinBox *_viewportWidthSpinBox;
    QLabel *label_18;
    QSpinBox *_viewportHeightSpinBox;
    QGroupBox *_viewportOffscreenSizeGroupBox;
    QGridLayout *gridLayout_4;
    QLabel *label_16;
    QComboBox *_viewportOffscreenSizeComboBox;
    QLabel *label_14;
    QSpinBox *_graphicsBufferWidthSpinBox;
    QLabel *label_15;
    QSpinBox *_graphicsBufferHeightSpinBox;
    QGroupBox *_timeBudgetParametersGroupBox;
    QGridLayout *gridLayout_2;
    QLabel *label;
    QLineEdit *_timeBudgetLineEdit;
    QSpinBox *_timeBudgetSpinBox;
    QSpacerItem *verticalSpacer;

    void setupUi(QWidget *GvvQRendererEditor)
    {
        if (GvvQRendererEditor->objectName().isEmpty())
            GvvQRendererEditor->setObjectName(QString::fromUtf8("GvvQRendererEditor"));
        GvvQRendererEditor->resize(311, 526);
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(GvvQRendererEditor->sizePolicy().hasHeightForWidth());
        GvvQRendererEditor->setSizePolicy(sizePolicy);
        verticalLayout = new QVBoxLayout(GvvQRendererEditor);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        groupBox_3 = new QGroupBox(GvvQRendererEditor);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        gridLayout = new QGridLayout(groupBox_3);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label_5 = new QLabel(groupBox_3);
        label_5->setObjectName(QString::fromUtf8("label_5"));

        gridLayout->addWidget(label_5, 0, 0, 1, 1);

        _maxDepthSpinBox = new QSpinBox(groupBox_3);
        _maxDepthSpinBox->setObjectName(QString::fromUtf8("_maxDepthSpinBox"));
        _maxDepthSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _maxDepthSpinBox->setMaximum(31);
        _maxDepthSpinBox->setSingleStep(1);

        gridLayout->addWidget(_maxDepthSpinBox, 0, 1, 1, 1);

        _maxResolutionLineEdit = new QLineEdit(groupBox_3);
        _maxResolutionLineEdit->setObjectName(QString::fromUtf8("_maxResolutionLineEdit"));
        _maxResolutionLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _maxResolutionLineEdit->setReadOnly(true);

        gridLayout->addWidget(_maxResolutionLineEdit, 0, 2, 1, 1);

        label_6 = new QLabel(groupBox_3);
        label_6->setObjectName(QString::fromUtf8("label_6"));

        gridLayout->addWidget(label_6, 1, 0, 1, 1);

        _nbMaxRayCastingIterationSpinBox = new QSpinBox(groupBox_3);
        _nbMaxRayCastingIterationSpinBox->setObjectName(QString::fromUtf8("_nbMaxRayCastingIterationSpinBox"));
        _nbMaxRayCastingIterationSpinBox->setEnabled(false);
        _nbMaxRayCastingIterationSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _nbMaxRayCastingIterationSpinBox->setMaximum(31);
        _nbMaxRayCastingIterationSpinBox->setSingleStep(1);

        gridLayout->addWidget(_nbMaxRayCastingIterationSpinBox, 1, 1, 1, 2);

        _dynamicUpdateCheckBox = new QCheckBox(groupBox_3);
        _dynamicUpdateCheckBox->setObjectName(QString::fromUtf8("_dynamicUpdateCheckBox"));

        gridLayout->addWidget(_dynamicUpdateCheckBox, 2, 0, 1, 2);


        verticalLayout->addWidget(groupBox_3);

        groupBox_4 = new QGroupBox(GvvQRendererEditor);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        gridLayout_3 = new QGridLayout(groupBox_4);
        gridLayout_3->setObjectName(QString::fromUtf8("gridLayout_3"));
        _priorityOnNodesRadioButton = new QRadioButton(groupBox_4);
        _priorityOnNodesRadioButton->setObjectName(QString::fromUtf8("_priorityOnNodesRadioButton"));

        gridLayout_3->addWidget(_priorityOnNodesRadioButton, 1, 0, 1, 1);

        _priorityOnBricksRadioButton = new QRadioButton(groupBox_4);
        _priorityOnBricksRadioButton->setObjectName(QString::fromUtf8("_priorityOnBricksRadioButton"));

        gridLayout_3->addWidget(_priorityOnBricksRadioButton, 0, 0, 1, 1);


        verticalLayout->addWidget(groupBox_4);

        groupBox_2 = new QGroupBox(GvvQRendererEditor);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        gridLayout_7 = new QGridLayout(groupBox_2);
        gridLayout_7->setObjectName(QString::fromUtf8("gridLayout_7"));
        label_3 = new QLabel(groupBox_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_7->addWidget(label_3, 0, 0, 1, 1);

        _viewportXSpinBox = new QSpinBox(groupBox_2);
        _viewportXSpinBox->setObjectName(QString::fromUtf8("_viewportXSpinBox"));
        _viewportXSpinBox->setEnabled(false);
        _viewportXSpinBox->setMaximum(1920);

        gridLayout_7->addWidget(_viewportXSpinBox, 0, 1, 1, 1);

        label_4 = new QLabel(groupBox_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_7->addWidget(label_4, 0, 2, 1, 1);

        _viewportYSpinBox = new QSpinBox(groupBox_2);
        _viewportYSpinBox->setObjectName(QString::fromUtf8("_viewportYSpinBox"));
        _viewportYSpinBox->setEnabled(false);
        _viewportYSpinBox->setMaximum(1200);

        gridLayout_7->addWidget(_viewportYSpinBox, 0, 3, 1, 1);

        label_17 = new QLabel(groupBox_2);
        label_17->setObjectName(QString::fromUtf8("label_17"));

        gridLayout_7->addWidget(label_17, 1, 0, 1, 1);

        _viewportWidthSpinBox = new QSpinBox(groupBox_2);
        _viewportWidthSpinBox->setObjectName(QString::fromUtf8("_viewportWidthSpinBox"));
        _viewportWidthSpinBox->setEnabled(false);
        _viewportWidthSpinBox->setMinimum(1);
        _viewportWidthSpinBox->setMaximum(1920);
        _viewportWidthSpinBox->setValue(512);

        gridLayout_7->addWidget(_viewportWidthSpinBox, 1, 1, 1, 1);

        label_18 = new QLabel(groupBox_2);
        label_18->setObjectName(QString::fromUtf8("label_18"));

        gridLayout_7->addWidget(label_18, 1, 2, 1, 1);

        _viewportHeightSpinBox = new QSpinBox(groupBox_2);
        _viewportHeightSpinBox->setObjectName(QString::fromUtf8("_viewportHeightSpinBox"));
        _viewportHeightSpinBox->setEnabled(false);
        _viewportHeightSpinBox->setMinimum(1);
        _viewportHeightSpinBox->setMaximum(1200);
        _viewportHeightSpinBox->setValue(512);

        gridLayout_7->addWidget(_viewportHeightSpinBox, 1, 3, 1, 1);

        _viewportOffscreenSizeGroupBox = new QGroupBox(groupBox_2);
        _viewportOffscreenSizeGroupBox->setObjectName(QString::fromUtf8("_viewportOffscreenSizeGroupBox"));
        _viewportOffscreenSizeGroupBox->setCheckable(true);
        _viewportOffscreenSizeGroupBox->setChecked(false);
        gridLayout_4 = new QGridLayout(_viewportOffscreenSizeGroupBox);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        label_16 = new QLabel(_viewportOffscreenSizeGroupBox);
        label_16->setObjectName(QString::fromUtf8("label_16"));

        gridLayout_4->addWidget(label_16, 0, 0, 1, 1);

        _viewportOffscreenSizeComboBox = new QComboBox(_viewportOffscreenSizeGroupBox);
        _viewportOffscreenSizeComboBox->setObjectName(QString::fromUtf8("_viewportOffscreenSizeComboBox"));

        gridLayout_4->addWidget(_viewportOffscreenSizeComboBox, 0, 1, 1, 2);

        label_14 = new QLabel(_viewportOffscreenSizeGroupBox);
        label_14->setObjectName(QString::fromUtf8("label_14"));

        gridLayout_4->addWidget(label_14, 1, 0, 1, 1);

        _graphicsBufferWidthSpinBox = new QSpinBox(_viewportOffscreenSizeGroupBox);
        _graphicsBufferWidthSpinBox->setObjectName(QString::fromUtf8("_graphicsBufferWidthSpinBox"));
        _graphicsBufferWidthSpinBox->setMinimum(1);
        _graphicsBufferWidthSpinBox->setMaximum(1920);
        _graphicsBufferWidthSpinBox->setValue(512);

        gridLayout_4->addWidget(_graphicsBufferWidthSpinBox, 1, 1, 1, 1);

        label_15 = new QLabel(_viewportOffscreenSizeGroupBox);
        label_15->setObjectName(QString::fromUtf8("label_15"));

        gridLayout_4->addWidget(label_15, 1, 2, 1, 1);

        _graphicsBufferHeightSpinBox = new QSpinBox(_viewportOffscreenSizeGroupBox);
        _graphicsBufferHeightSpinBox->setObjectName(QString::fromUtf8("_graphicsBufferHeightSpinBox"));
        _graphicsBufferHeightSpinBox->setMinimum(1);
        _graphicsBufferHeightSpinBox->setMaximum(1200);
        _graphicsBufferHeightSpinBox->setValue(512);

        gridLayout_4->addWidget(_graphicsBufferHeightSpinBox, 1, 3, 1, 1);


        gridLayout_7->addWidget(_viewportOffscreenSizeGroupBox, 2, 0, 1, 4);


        verticalLayout->addWidget(groupBox_2);

        _timeBudgetParametersGroupBox = new QGroupBox(GvvQRendererEditor);
        _timeBudgetParametersGroupBox->setObjectName(QString::fromUtf8("_timeBudgetParametersGroupBox"));
        _timeBudgetParametersGroupBox->setEnabled(true);
        _timeBudgetParametersGroupBox->setCheckable(true);
        _timeBudgetParametersGroupBox->setChecked(false);
        gridLayout_2 = new QGridLayout(_timeBudgetParametersGroupBox);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        label = new QLabel(_timeBudgetParametersGroupBox);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout_2->addWidget(label, 0, 0, 1, 1);

        _timeBudgetLineEdit = new QLineEdit(_timeBudgetParametersGroupBox);
        _timeBudgetLineEdit->setObjectName(QString::fromUtf8("_timeBudgetLineEdit"));
        _timeBudgetLineEdit->setEnabled(false);
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(_timeBudgetLineEdit->sizePolicy().hasHeightForWidth());
        _timeBudgetLineEdit->setSizePolicy(sizePolicy1);
        _timeBudgetLineEdit->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _timeBudgetLineEdit->setReadOnly(true);

        gridLayout_2->addWidget(_timeBudgetLineEdit, 0, 1, 1, 1);

        _timeBudgetSpinBox = new QSpinBox(_timeBudgetParametersGroupBox);
        _timeBudgetSpinBox->setObjectName(QString::fromUtf8("_timeBudgetSpinBox"));
        sizePolicy1.setHeightForWidth(_timeBudgetSpinBox->sizePolicy().hasHeightForWidth());
        _timeBudgetSpinBox->setSizePolicy(sizePolicy1);
        _timeBudgetSpinBox->setAlignment(Qt::AlignRight|Qt::AlignTrailing|Qt::AlignVCenter);
        _timeBudgetSpinBox->setMinimum(1);
        _timeBudgetSpinBox->setMaximum(120);
        _timeBudgetSpinBox->setValue(60);

        gridLayout_2->addWidget(_timeBudgetSpinBox, 0, 2, 1, 1);


        verticalLayout->addWidget(_timeBudgetParametersGroupBox);

        verticalSpacer = new QSpacerItem(20, 42, QSizePolicy::Minimum, QSizePolicy::Expanding);

        verticalLayout->addItem(verticalSpacer);


        retranslateUi(GvvQRendererEditor);

        QMetaObject::connectSlotsByName(GvvQRendererEditor);
    } // setupUi

    void retranslateUi(QWidget *GvvQRendererEditor)
    {
        GvvQRendererEditor->setWindowTitle(QApplication::translate("GvvQRendererEditor", "Renderer Editor", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("GvvQRendererEditor", "Renderer", 0, QApplication::UnicodeUTF8));
        label_5->setText(QApplication::translate("GvvQRendererEditor", "Max Depth", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _maxResolutionLineEdit->setToolTip(QApplication::translate("GvvQRendererEditor", "Resolution max, i.e. nb of voxels in each dimension", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_6->setText(QApplication::translate("GvvQRendererEditor", "Nb Max Loop", 0, QApplication::UnicodeUTF8));
        _dynamicUpdateCheckBox->setText(QApplication::translate("GvvQRendererEditor", "Dynamic Update", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("GvvQRendererEditor", "Requests Strategy", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _priorityOnNodesRadioButton->setToolTip(QApplication::translate("GvvQRendererEditor", "Priorirty on node subdivisions", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        _priorityOnNodesRadioButton->setText(QApplication::translate("GvvQRendererEditor", "Quality before", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        _priorityOnBricksRadioButton->setToolTip(QApplication::translate("GvvQRendererEditor", "Priorirty on brick loads", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        _priorityOnBricksRadioButton->setText(QApplication::translate("GvvQRendererEditor", "Peformance trade-off", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("GvvQRendererEditor", "Viewport", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("GvvQRendererEditor", "X", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("GvvQRendererEditor", "Y", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("GvvQRendererEditor", "Width", 0, QApplication::UnicodeUTF8));
        label_18->setText(QApplication::translate("GvvQRendererEditor", "Height", 0, QApplication::UnicodeUTF8));
        _viewportOffscreenSizeGroupBox->setTitle(QApplication::translate("GvvQRendererEditor", "Offsrceen Buffer", 0, QApplication::UnicodeUTF8));
        label_16->setText(QApplication::translate("GvvQRendererEditor", "Size", 0, QApplication::UnicodeUTF8));
        _viewportOffscreenSizeComboBox->clear();
        _viewportOffscreenSizeComboBox->insertItems(0, QStringList()
         << QApplication::translate("GvvQRendererEditor", "512 x 512", 0, QApplication::UnicodeUTF8)
        );
        label_14->setText(QApplication::translate("GvvQRendererEditor", "Width", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("GvvQRendererEditor", "Height", 0, QApplication::UnicodeUTF8));
        _timeBudgetParametersGroupBox->setTitle(QApplication::translate("GvvQRendererEditor", "User Requests", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("GvvQRendererEditor", "Time Budget", 0, QApplication::UnicodeUTF8));
        _timeBudgetSpinBox->setSuffix(QApplication::translate("GvvQRendererEditor", " fps", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class GvvQRendererEditor: public Ui_GvvQRendererEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVVQRENDEREREDITOR_H
