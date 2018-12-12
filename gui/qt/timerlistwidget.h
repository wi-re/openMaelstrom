#pragma once
#include <QTimer>
#include <QTreeWidget>
#include <QLabel>
#include <utility/helpers/timer.h>
#include <gui/qt/docker.h>

/** This class is used to store a list of all current timers and show some basic
 * statistics about them. The list is auto updated with a timer every 500ms.**/
class TimerListWidget : public Docker {
  Q_OBJECT
  /** This class is used to represent entries in the list of timers in the GUI.
   * It contains no functionality besides containing two labels containing
   * median, maxmium, average and the corresponding standard deviation of the
   * timer.**/
  class TimerEntry : public QWidget {
    QLabel *m_minmaxLabel = nullptr;
    QLabel *m_avgdevLabel = nullptr;

  public:
    Timer *m_referenceTimer = nullptr;

    explicit TimerEntry(Timer *timer, QWidget *parent = nullptr);

    void updateLabels();
  };
  /** GUI Elements **/
  QTreeWidget *m_timerList = nullptr;
  /** Stores list of all timers added to check if a new one was added globally **/
  std::vector<TimerEntry *> m_timerEntries;

  QTimer *m_updateTimer;
  /** Helper function that creages the GUI elements for a specific Timer **/
  void addTimer(Timer *t);
public:
  explicit TimerListWidget(QWidget *parent = nullptr);
  ~TimerListWidget() {}
};
