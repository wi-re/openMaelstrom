#pragma once
#include "ui_mainwindow.h"
#include <QLineEdit>
#include <QRadioButton>
#include <QTextEdit>
#include <QTimer>
#include <gui/qt/docker.h>
#include <mutex>
#include <utility/helpers/log.h>

/** Console logging window that displays all log messages with filtering by log
 * level and contains a search box. Updates the global log function! **/
class console : public Docker {
  Q_OBJECT
  /** GUI Fields **/
  QRadioButton *m_logAll, *m_logDebug, *m_logWarning, *m_logError, *m_logVerbose;
  QLineEdit *m_searchField;
  QTextEdit *m_consoleField;

  /** Timer to periodically update the log window **/
  QTimer m_timer;

  /** Constructor declared private due to Singleton**/
  console(QWidget *parent = nullptr);

public:
  /** Magic statics cannot be used here because Qt requires
   * objects to be deleted from the thread they were created from.
   * However this instance is not created from the main thread so
   * this does not work without creating an error message on closing the program
   * public so it can be deleted on closing the program **/
  static console *m_instance;
  /** Contains preprocessed log entries for quicker access **/
  std::vector<std::pair<log_level, QString>> m_logEntries;
  /** To ensure thread safe changes to entries (which are not thread safe) **/
  std::mutex m_logMutex;

  /** Singleton function to get the instance **/
  static console &instance();
  /** Converts a log entry into a log message in the format (HTML):
   * <font color="COLOR"> DESCRIPTION: </font> HH:MM:SS -> MESSAGE <br>
   * where the color and description depend on the logging level of the message:
   * level	    COLOR		DESCRIPTION
   * info	 -> Black		INFO
   * error	 -> Red			ERROR
   * debug	 -> green		DEBUG
   * warning -> aqua		WARNING
   * verbose -> grey		VERBOSE
   **/
  static QString messageToString(log_level log, std::chrono::system_clock::time_point time, std::string message);
   
public slots:
  /** Function periodically called to update the log entries **/
  void timerHit();
};
