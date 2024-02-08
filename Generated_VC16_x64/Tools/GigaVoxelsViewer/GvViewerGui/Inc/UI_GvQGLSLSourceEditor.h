/********************************************************************************
** Form generated from reading UI file 'GvQGLSLSourceEditor.ui'
**
** Created by: Qt User Interface Compiler version 4.8.7
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_GVQGLSLSOURCEEDITOR_H
#define UI_GVQGLSLSOURCEEDITOR_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QDockWidget>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSpacerItem>
#include <QtGui/QTabWidget>
#include <QtGui/QTextEdit>
#include <QtGui/QVBoxLayout>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_GvQGLSLSourceEditor
{
public:
    QGridLayout *gridLayout_2;
    QGroupBox *groupBox;
    QHBoxLayout *horizontalLayout;
    QGroupBox *groupBox_3;
    QHBoxLayout *horizontalLayout_2;
    QGroupBox *groupBox_2;
    QGridLayout *gridLayout;
    QLabel *label;
    QLineEdit *_shaderFilenameLineEdit;
    QTabWidget *tabWidget;
    QWidget *tab;
    QVBoxLayout *verticalLayout_2;
    QTextEdit *_vertexShaderTextEdit;
    QWidget *tab_2;
    QVBoxLayout *verticalLayout_3;
    QTextEdit *_tesselationControlShaderTextEdit;
    QWidget *tab_3;
    QVBoxLayout *verticalLayout_4;
    QTextEdit *_tesselationEvaluationShaderTextEdit;
    QWidget *tab_4;
    QVBoxLayout *verticalLayout_5;
    QTextEdit *_geometryShaderTextEdit;
    QWidget *tab_5;
    QVBoxLayout *verticalLayout_6;
    QTextEdit *_fragmentShaderTextEdit;
    QWidget *tab_6;
    QVBoxLayout *verticalLayout_7;
    QTextEdit *_computeShaderTextEdit;
    QPushButton *_reloadButton;
    QSpacerItem *spacerItem;
    QPushButton *_applyButton;
    QDockWidget *_logWindowDockWidget;
    QWidget *dockWidgetContents_2;
    QVBoxLayout *verticalLayout_8;
    QGroupBox *groupBox_4;
    QVBoxLayout *verticalLayout;
    QTextEdit *_logWindowTextEdit;

    void setupUi(QWidget *GvQGLSLSourceEditor)
    {
        if (GvQGLSLSourceEditor->objectName().isEmpty())
            GvQGLSLSourceEditor->setObjectName(QString::fromUtf8("GvQGLSLSourceEditor"));
        GvQGLSLSourceEditor->setWindowModality(Qt::WindowModal);
        GvQGLSLSourceEditor->resize(755, 565);
        gridLayout_2 = new QGridLayout(GvQGLSLSourceEditor);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        groupBox = new QGroupBox(GvQGLSLSourceEditor);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        horizontalLayout = new QHBoxLayout(groupBox);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));

        gridLayout_2->addWidget(groupBox, 0, 0, 1, 2);

        groupBox_3 = new QGroupBox(GvQGLSLSourceEditor);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        horizontalLayout_2 = new QHBoxLayout(groupBox_3);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));

        gridLayout_2->addWidget(groupBox_3, 0, 2, 1, 2);

        groupBox_2 = new QGroupBox(GvQGLSLSourceEditor);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        gridLayout = new QGridLayout(groupBox_2);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        label = new QLabel(groupBox_2);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        _shaderFilenameLineEdit = new QLineEdit(groupBox_2);
        _shaderFilenameLineEdit->setObjectName(QString::fromUtf8("_shaderFilenameLineEdit"));
        _shaderFilenameLineEdit->setReadOnly(true);

        gridLayout->addWidget(_shaderFilenameLineEdit, 0, 1, 1, 1);

        tabWidget = new QTabWidget(groupBox_2);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setEnabled(true);
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        verticalLayout_2 = new QVBoxLayout(tab);
        verticalLayout_2->setObjectName(QString::fromUtf8("verticalLayout_2"));
        _vertexShaderTextEdit = new QTextEdit(tab);
        _vertexShaderTextEdit->setObjectName(QString::fromUtf8("_vertexShaderTextEdit"));

        verticalLayout_2->addWidget(_vertexShaderTextEdit);

        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        verticalLayout_3 = new QVBoxLayout(tab_2);
        verticalLayout_3->setObjectName(QString::fromUtf8("verticalLayout_3"));
        _tesselationControlShaderTextEdit = new QTextEdit(tab_2);
        _tesselationControlShaderTextEdit->setObjectName(QString::fromUtf8("_tesselationControlShaderTextEdit"));

        verticalLayout_3->addWidget(_tesselationControlShaderTextEdit);

        tabWidget->addTab(tab_2, QString());
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        verticalLayout_4 = new QVBoxLayout(tab_3);
        verticalLayout_4->setObjectName(QString::fromUtf8("verticalLayout_4"));
        _tesselationEvaluationShaderTextEdit = new QTextEdit(tab_3);
        _tesselationEvaluationShaderTextEdit->setObjectName(QString::fromUtf8("_tesselationEvaluationShaderTextEdit"));

        verticalLayout_4->addWidget(_tesselationEvaluationShaderTextEdit);

        tabWidget->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        verticalLayout_5 = new QVBoxLayout(tab_4);
        verticalLayout_5->setObjectName(QString::fromUtf8("verticalLayout_5"));
        _geometryShaderTextEdit = new QTextEdit(tab_4);
        _geometryShaderTextEdit->setObjectName(QString::fromUtf8("_geometryShaderTextEdit"));

        verticalLayout_5->addWidget(_geometryShaderTextEdit);

        tabWidget->addTab(tab_4, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QString::fromUtf8("tab_5"));
        verticalLayout_6 = new QVBoxLayout(tab_5);
        verticalLayout_6->setObjectName(QString::fromUtf8("verticalLayout_6"));
        _fragmentShaderTextEdit = new QTextEdit(tab_5);
        _fragmentShaderTextEdit->setObjectName(QString::fromUtf8("_fragmentShaderTextEdit"));

        verticalLayout_6->addWidget(_fragmentShaderTextEdit);

        tabWidget->addTab(tab_5, QString());
        tab_6 = new QWidget();
        tab_6->setObjectName(QString::fromUtf8("tab_6"));
        verticalLayout_7 = new QVBoxLayout(tab_6);
        verticalLayout_7->setObjectName(QString::fromUtf8("verticalLayout_7"));
        _computeShaderTextEdit = new QTextEdit(tab_6);
        _computeShaderTextEdit->setObjectName(QString::fromUtf8("_computeShaderTextEdit"));

        verticalLayout_7->addWidget(_computeShaderTextEdit);

        tabWidget->addTab(tab_6, QString());

        gridLayout->addWidget(tabWidget, 1, 0, 1, 2);


        gridLayout_2->addWidget(groupBox_2, 1, 0, 1, 4);

        _reloadButton = new QPushButton(GvQGLSLSourceEditor);
        _reloadButton->setObjectName(QString::fromUtf8("_reloadButton"));
        QSizePolicy sizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(_reloadButton->sizePolicy().hasHeightForWidth());
        _reloadButton->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(_reloadButton, 2, 0, 1, 1);

        spacerItem = new QSpacerItem(533, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        gridLayout_2->addItem(spacerItem, 2, 1, 1, 2);

        _applyButton = new QPushButton(GvQGLSLSourceEditor);
        _applyButton->setObjectName(QString::fromUtf8("_applyButton"));
        sizePolicy.setHeightForWidth(_applyButton->sizePolicy().hasHeightForWidth());
        _applyButton->setSizePolicy(sizePolicy);

        gridLayout_2->addWidget(_applyButton, 2, 3, 1, 1);

        _logWindowDockWidget = new QDockWidget(GvQGLSLSourceEditor);
        _logWindowDockWidget->setObjectName(QString::fromUtf8("_logWindowDockWidget"));
        dockWidgetContents_2 = new QWidget();
        dockWidgetContents_2->setObjectName(QString::fromUtf8("dockWidgetContents_2"));
        verticalLayout_8 = new QVBoxLayout(dockWidgetContents_2);
        verticalLayout_8->setObjectName(QString::fromUtf8("verticalLayout_8"));
        groupBox_4 = new QGroupBox(dockWidgetContents_2);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        verticalLayout = new QVBoxLayout(groupBox_4);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        _logWindowTextEdit = new QTextEdit(groupBox_4);
        _logWindowTextEdit->setObjectName(QString::fromUtf8("_logWindowTextEdit"));
        _logWindowTextEdit->setReadOnly(true);

        verticalLayout->addWidget(_logWindowTextEdit);


        verticalLayout_8->addWidget(groupBox_4);

        _logWindowDockWidget->setWidget(dockWidgetContents_2);

        gridLayout_2->addWidget(_logWindowDockWidget, 3, 0, 1, 4);

        _applyButton->raise();
        groupBox->raise();
        groupBox_3->raise();
        groupBox_2->raise();
        _reloadButton->raise();
        _logWindowDockWidget->raise();

        retranslateUi(GvQGLSLSourceEditor);

        tabWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(GvQGLSLSourceEditor);
    } // setupUi

    void retranslateUi(QWidget *GvQGLSLSourceEditor)
    {
        GvQGLSLSourceEditor->setWindowTitle(QApplication::translate("GvQGLSLSourceEditor", "GLSL - Program Source Editor", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("GvQGLSLSourceEditor", "Shader Browser", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("GvQGLSLSourceEditor", "Info", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QString());
        label->setText(QApplication::translate("GvQGLSLSourceEditor", "File", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("GvQGLSLSourceEditor", "Vertex", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("GvQGLSLSourceEditor", "Tesselation Control", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_3), QApplication::translate("GvQGLSLSourceEditor", "Tesselation Evaluation", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_4), QApplication::translate("GvQGLSLSourceEditor", "Geometry", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_5), QApplication::translate("GvQGLSLSourceEditor", "Fragment", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_6), QApplication::translate("GvQGLSLSourceEditor", "Compute", 0, QApplication::UnicodeUTF8));
        _reloadButton->setText(QApplication::translate("GvQGLSLSourceEditor", "Reload", 0, QApplication::UnicodeUTF8));
        _applyButton->setText(QApplication::translate("GvQGLSLSourceEditor", "Apply", 0, QApplication::UnicodeUTF8));
        _logWindowDockWidget->setWindowTitle(QApplication::translate("GvQGLSLSourceEditor", "LOG Window - GLSL Editor", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QString());
    } // retranslateUi

};

namespace Ui {
    class GvQGLSLSourceEditor: public Ui_GvQGLSLSourceEditor {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_GVQGLSLSOURCEEDITOR_H
