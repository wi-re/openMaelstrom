#include "propertyviewer.h"
#include "timerlistwidget.h"
#include "ui_mainwindow.h"
#include <QCommonStyle>
#include <QDebug>
#include <QDockWidget>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListView>
#include <QMainWindow>
#include <QMouseEvent>
#include <QPushButton>
#include <QRadioButton>
#include <QShortcut>
#include <QStringList>
#include <QStyleFactory>
#include <QTextEdit>
#include <QThread>
#include <QTime>
#include <QVBoxLayout>
#include <fstream>
#include <gui/qt/console.h>
#include <gui/qt/mainwindow.h>
#include <gui/qt/propertyviewer.h>
#include <gui/qt/qgraph.h>
#include <gui/qt/timerlistwidget.h>
#include <iostream>
#include <render/qGLWidget/oglwidget.h>
#include <vector>
#include<utility/helpers/pathfinder.h>
#include <utility/helpers/arguments.h>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), m_ui(new Ui::MainWindow) {
  setDockOptions(QMainWindow::AllowTabbedDocks);

  auto f = resolveFile("cfg/style.css");
  auto p = f.parent_path().string();
  if (*(p.end() - 1) == '/' || *(p.end() - 1) == '\\')
    p = p.substr(0, p.length() - 1);
  std::replace(p.begin(), p.end(), '\\', '/'); 
  std::string style_file = f.string();
  std::ifstream t(f);
  std::string str((std::istreambuf_iterator<char>(t)),
                  std::istreambuf_iterator<char>());
  QString style = QString::fromStdString(str);
  style.replace("$resource", QString::fromStdString(p));
  QApplication::setStyle(style);
  qApp->setStyleSheet(style);

  m_ui->setupUi(this);
  setMinimumSize(1920, 1080);
  setWindowFlags(Qt::FramelessWindowHint);

  createMenu();
  createHotkeys();

  setCentralWidget(m_openGLWidget = new OGLWidget);
  addDockWidget(Qt::BottomDockWidgetArea,
                m_consoleDocker = &console::instance());
  m_consoleDocker->setVisible(false);
  if (get<parameters::gl_record>() || arguments::cmd::instance().renderToFile || arguments::cmd::instance().rtx) {
    m_consoleDocker->setFloating(true);
    m_consoleDocker->setVisible(false);
  }
}
MainWindow::~MainWindow() {
  delete console::m_instance;
  delete m_ui;
}

void MainWindow::keyPressEvent(QKeyEvent *event) {
  m_openGLWidget->keyPressEvent(event);
}
void MainWindow::keyReleaseEvent(QKeyEvent *event) {
  m_openGLWidget->keyReleaseEvent(event);
}
bool MainWindow::eventFilter(QObject *watched, QEvent *event) {
  if (watched == m_ui->menuBar) {
    if (event->type() == QEvent::MouseButtonPress) {
      QMouseEvent *mouse_event = dynamic_cast<QMouseEvent *>(event);
      if (mouse_event->button() == Qt::LeftButton) {
        m_mouseClickedOnMenuBar = true;
        m_dragPosition = mouse_event->globalPos() - frameGeometry().topLeft();
        return false;
      }
    } else if (event->type() == QEvent::MouseMove) {
      QMouseEvent *mouse_event = dynamic_cast<QMouseEvent *>(event);
      if (mouse_event->buttons() & Qt::LeftButton && m_mouseClickedOnMenuBar) {
        move(mouse_event->globalPos() - m_dragPosition);
        return false;
      }
    } else if (event->type() == QEvent::MouseButtonRelease) {
      QMouseEvent *mouse_event = dynamic_cast<QMouseEvent *>(event);
      if (mouse_event->button() == Qt::LeftButton) {
        m_mouseClickedOnMenuBar = false;
        return false;
      }
    }
  }
  return false;
}

void MainWindow::createMenu() {
  QStyle *appStyle = qApp->style();
  QWidget *widget = new QWidget(menuBar());

  m_minButton = new QPushButton(widget);
  m_minButton->setObjectName("SizeButton");
  m_minButton->setIcon(appStyle->standardIcon(QStyle::SP_TitleBarMinButton));

  m_maxButton = new QPushButton(widget);
  m_maxButton->setObjectName("SizeButton");
  m_maxButton->setIcon(appStyle->standardIcon(QStyle::SP_TitleBarMaxButton));

  m_closeButton = new QPushButton(widget);
  m_closeButton->setObjectName("ExitButton");
  m_closeButton->setIcon(
      appStyle->standardIcon(QStyle::SP_TitleBarCloseButton));

  QIcon closeIcon = appStyle->standardIcon(QStyle::SP_TitleBarCloseButton);
  QIcon maxIcon = appStyle->standardIcon(QStyle::SP_TitleBarMaxButton);
  QIcon minIcon = appStyle->standardIcon(QStyle::SP_TitleBarMinButton);

  QHBoxLayout *layout = new QHBoxLayout(widget);
  layout->setSpacing(0);
  layout->addWidget(m_minButton);
  layout->addWidget(m_maxButton);
  layout->addWidget(m_closeButton);
  layout->setContentsMargins(0, 0, 0, 0);
  menuBar()->setContentsMargins(0, 0, 0, 0);
  widget->setLayout(layout);

  menuBar()->setCornerWidget(widget, Qt::TopRightCorner);

  connect(m_closeButton, &QPushButton::released,
          [=]() { QApplication::quit(); });
  connect(m_minButton, &QPushButton::released,
          [=]() { this->setWindowState(Qt::WindowMinimized); });
  connect(m_maxButton, &QPushButton::released, [=]() {
    if (!this->isMaximized()) {
      this->showMaximized();
      m_maxButton->setIcon(
          appStyle->standardIcon(QStyle::SP_TitleBarNormalButton));
    } else {
      this->showNormal();
      m_maxButton->setIcon(
          appStyle->standardIcon(QStyle::SP_TitleBarMaxButton));
    }
  });

  m_ui->menuBar->installEventFilter(this);
}
void MainWindow::createHotkeys() {
  QObject::connect(new QShortcut(QKeySequence(Qt::ALT + Qt::Key_Q), this),
                   &QShortcut::activated, [=]() { QApplication::quit(); });
  QObject::connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_G), this),
                   &QShortcut::activated, [=]() {
                     if (m_graphDocker == nullptr) {
                       m_graphDocker = new QGraph;
                       addDockWidget(Qt::BottomDockWidgetArea, m_graphDocker);
                       m_graphDocker->setFloating(true);
                       m_graphDocker->resize(1920, 1080);
                       m_graphDocker->move(QCursor::pos());
                     } else {
                       m_graphDocker->setVisible(!m_graphDocker->isVisible());
                       if (m_graphDocker->isFloating())
                         m_graphDocker->move(QCursor::pos());
                     }
                   });
  QObject::connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_T), this),
                   &QShortcut::activated, [=]() {
                     if (m_timerDocker == nullptr) {
                       m_timerDocker = new TimerListWidget;
                       addDockWidget(Qt::RightDockWidgetArea, m_timerDocker);
                       m_timerDocker->setFloating(true);
                       m_timerDocker->resize(256, 1080);
                       m_timerDocker->move(QCursor::pos());
                     } else {
                       m_timerDocker->setVisible(!m_timerDocker->isVisible());
                       if (m_timerDocker->isFloating())
                         m_timerDocker->move(QCursor::pos());
                     }
                   });
  QObject::connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_W), this),
                   &QShortcut::activated, [=]() {
                     if (m_propertyDocker == nullptr) {
                       m_propertyDocker = new PropertyViewer();
                       addDockWidget(Qt::RightDockWidgetArea, m_propertyDocker);
                       m_propertyDocker->setFloating(true);
                       m_propertyDocker->resize(256, 1080);
                       m_propertyDocker->move(QCursor::pos());
                     } else {
                       m_propertyDocker->setVisible(
                           !m_propertyDocker->isVisible());
                       if (m_propertyDocker->isFloating())
                         m_propertyDocker->move(QCursor::pos());
                     }
                   });
  QObject::connect(new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_D), this),
                   &QShortcut::activated, [=]() {
                     m_consoleDocker->setVisible(!m_consoleDocker->isVisible());
                     if (m_consoleDocker->isFloating())
                       m_consoleDocker->move(QCursor::pos());
                   });
}
