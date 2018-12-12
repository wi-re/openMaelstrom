#pragma once
#include "ui_mainwindow.h"
#include <QMainWindow>
#include <gui/qt/console.h>
#include <gui/qt/propertyviewer.h>
#include <gui/qt/qgraph.h>
#include <gui/qt/timerlistwidget.h>
#include <render/qGLWidget/oglwidget.h>

namespace Ui {
class MainWindow;
}

/** This class represents the main window of the simulation.  **/
class MainWindow : public QMainWindow {
  // !!!! Q_OBJECT !!!!!
  /** GUI Elements**/
  QPushButton *m_minButton, *m_maxButton, *m_closeButton;
  Ui::MainWindow *m_ui;

  /** OpenGL Instance for rendering the simulation **/
  OGLWidget *m_openGLWidget = nullptr;

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow() override;

protected:
  /** Key presses need to be forwarded to the openGL Instance which handles
   * interaction directly. **/
  virtual void keyPressEvent(QKeyEvent *event) override;
  virtual void keyReleaseEvent(QKeyEvent *event) override;
  /** Events that involve clicking on the menu bar are filtered out here and
   * used to drag around the window by dragging it with the menu bar.**/
  QPoint m_dragPosition;
  bool m_mouseClickedOnMenuBar = false;
  bool eventFilter(QObject *watched, QEvent *event) override;
  /** Dockers for helpful data. m_consoleDocker is a singleton and needs to be
   * deleted in this windows destructor. **/
  Docker *m_graphDocker = nullptr, *m_propertyDocker = nullptr,
         *m_consoleDocker = nullptr, *m_timerDocker = nullptr;

private:
  /** Helper functions to keep menubar and hotkey creation centralized.**/
  void createMenu();
  void createHotkeys();
};
