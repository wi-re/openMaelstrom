#include <QFontDatabase>
#include <QGroupBox>
#include <QLabel>
#include <QScrollBar>
#include <QVBoxLayout>
#include <ctime>
#include <gui/qt/console.h>
#include <utility/helpers/log.h>

console *console::m_instance = nullptr;

console &console::instance() {
  if (m_instance == nullptr)
    m_instance = new console();
  return *m_instance;
}

console::console(QWidget *parent) : Docker("console", parent) {
  LOG_VERBOSE << "Creating console widget";
  std::lock_guard<std::mutex> lock(m_logMutex);
  setAllowedAreas(Qt::AllDockWidgetAreas);
  m_timer.setInterval(50);
  // Helper lambdas to remove duplicate cote
  auto radioButton = [&](auto str, bool checked = false) {
    auto temp = new QRadioButton(QString(str));
    temp->setChecked(checked);
    return temp;
  };
  auto setSpacing = [&](auto wid) {
    wid->setContentsMargins(0, 0, 0, 0);
    wid->setSpacing(0);
  };

  // read all existing log entries into the console buffer
  for (auto log : logger::logs) {
    auto [level, time, message] = log;
    auto parsed = messageToString(level, time, message);
    m_logEntries.push_back(std::make_pair(level, parsed));
  }
  // update log function to reference this console
  logger::log_fn = [](std::string message, log_level log) {
    auto time = std::chrono::system_clock::now();
    logger::logs.push_back(std::make_tuple(log, time, message));
    std::lock_guard<std::mutex> lock(console::instance().m_logMutex);

    auto parsed = messageToString(log, time, message);
    instance().m_logEntries.push_back(std::make_pair(log, parsed));
    return std::make_pair(log, parsed);
  };

  // GUI creation stuff
  QGroupBox *groupBox = new QGroupBox(tr(""));
  QHBoxLayout *vbox = new QHBoxLayout;
  auto topWidget = new QWidget();
  groupBox->setLayout(vbox);

  setWidget(m_consoleField = new QTextEdit());
  m_layOut->removeWidget(m_maxButton);
  m_layOut->removeWidget(m_closeButton);

  m_layOut->addWidget(groupBox);
  m_layOut->addWidget(new QLabel("     Search:"));
  m_layOut->addWidget(m_searchField = new QLineEdit());
  m_layOut->addWidget(m_maxButton);
  m_layOut->addWidget(m_closeButton);

  vbox->addWidget(m_logAll = radioButton("&All", true));
  vbox->addWidget(m_logVerbose = radioButton("&Verbose"));
  vbox->addWidget(m_logDebug = radioButton("&Debug"));
  vbox->addWidget(m_logWarning = radioButton("&Warning"));
  vbox->addWidget(m_logError = radioButton("&Error"));

  setSpacing(vbox);
  
  m_consoleField->setReadOnly(true);
  m_consoleField->setFont(QFontDatabase::systemFont(QFontDatabase::FixedFont));

  connect(&m_timer, SIGNAL(timeout()), this, SLOT(timerHit()),
          Qt::DirectConnection);

  m_timer.start();
}

QString console::messageToString(log_level level,
                                 std::chrono::system_clock::time_point time,
                                 std::string message) {
  auto tt = std::chrono::system_clock::to_time_t(time);
  auto tm = std::localtime(&tt);

  std::stringstream sstream;
  switch (level) {
  case log_level::info:
    sstream << R"(<font color="Black">INFO: )";
    break;
  case log_level::error:
    sstream << R"(<font color="Red">ERROR: )";
    break;
  case log_level::debug:
    sstream << R"(<font color="green">DEBUG: )";
    break;
  case log_level::warning:
    sstream << R"(<font color="aqua">WARNING: )";
    break;
  case log_level::verbose:
	  sstream << R"(<font color="grey">VERBOSE: )";
	  break;
  }
  sstream << "</font> " << tm->tm_hour << ":" << tm->tm_min << ":" << tm->tm_sec
          << " -> " << message << R"(<br>)";
  return QString::fromStdString(sstream.str());
}

void console::timerHit() {
  static uint64_t last_size = UINT64_MAX;
  static int32_t last_radio;
  static QString last_str;

  auto &con = console::instance();
  auto all = con.m_logAll->isChecked();
  auto deb = con.m_logDebug->isChecked();
  auto war = con.m_logWarning->isChecked();
  auto err = con.m_logError->isChecked();
  auto ver = con.m_logVerbose->isChecked();
  auto str = con.m_searchField->text();

  int32_t radio = 0;
  radio += all ? 1 : 0;
  radio += deb ? 2 : 0;
  radio += war ? 4 : 0;
  radio += err ? 8 : 0;
  radio += ver ? 16 : 0;

  if (last_size != console::instance().m_logEntries.size() ||
      last_radio != radio || last_str != str) {
    std::lock_guard<std::mutex> lock(console::instance().m_logMutex);
    last_size = console::instance().m_logEntries.size();
    last_radio = radio;
    last_str = str;
    QString qs;
    for (const auto &q : console::instance().m_logEntries) {
      if (str != tr(""))
        if (!q.second.toLower().contains(str.toLower()))
          continue;
      switch (q.first) {
      case log_level::info:
        if (all || ver)
          qs.append(q.second);
        break;
      case log_level::debug:
        if (all || deb || ver)
          qs.append(q.second);
        break;
      case log_level::warning:
        if (all || deb || war || ver)
          qs.append(q.second);
        break;
      case log_level::error:
        if (all || deb || war || err || ver)
          qs.append(q.second);
        break;
      case log_level::verbose:
        if (ver)
          qs.append(q.second);
        break;
      }
    }
	m_consoleField->setHtml(qs);
	m_consoleField->ensureCursorVisible();
	QTextCursor c = m_consoleField->textCursor();
	c.movePosition(QTextCursor::End);
	m_consoleField->setTextCursor(c);
	m_consoleField->verticalScrollBar()->setValue(
	m_consoleField->verticalScrollBar()->maximum());
  }
}
