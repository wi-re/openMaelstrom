#include <QListWidget>
#include <QListWidgetItem>
#include <QSpacerItem>
#include <QTreeWidget>
#include <QHeaderView>
#include <gui/qt/propertyviewer.h>
#include <gui/qt/delegate.h>

// Defined as a global variable so <gui/qt/delegate.h> does not pollute
// everything in the GUI slowing syntax highlighting down.
std::vector<value_delegate *> m_parameterDelegates;

struct test {
	float4 a;
	int b;
};

test reflect{ {1, 2, 3, 4}, 2 };


PropertyViewer::PropertyViewer(QWidget *parent)
    : Docker("Property Viewer", parent) {
  setAllowedAreas(Qt::AllDockWidgetAreas);
  setMinimumSize(256, 512);
  // Setup empty list of parameters for now
  m_parameterTree = new QTreeWidget();
  m_parameterTree->header()->close();
  setWidget(m_parameterTree);
  // Setup timer to auto refresh parameters
  m_refreshTimer = new QTimer();
  connect(m_refreshTimer, &QTimer::timeout, [&]() {
	  for (auto d : m_parameterDelegates)
		  d->update();
	  QDockWidget::update();
  });
  m_refreshTimer->setInterval(500);
  m_refreshTimer->start(500);

  // Create list of QTreeWidgetItem-s for every uniform
  std::map<std::string, QTreeWidgetItem *> top;
  for_each(uniforms_list, [&](auto y) {
	  std::string str = std::string(decltype(y)::jsonName);
	  auto tok = split(str, '.');
	  if (top.find(tok[0]) == top.end()) {
		  auto wid = new QTreeWidgetItem();
		  top[tok[0]] = wid;
	  }
  });
  // Fill up treeWidget with items according to their hierarchy
  for (auto x : top) {
	  m_parameterTree->addTopLevelItem(x.second);
	  m_parameterTree->setItemWidget(x.second, 0, new QLabel(QString::fromStdString(x.first)));
  }
  // Fill up vector of delegates so the update function can work properly
  for_each(uniforms_list, [&](auto x) {
	  auto d = addToTree<decltype(x)>(m_parameterTree, top);
	  for (auto de : d)
		  if (de != nullptr)
			  m_parameterDelegates.push_back(de);
  });
  addToTree(&reflect, "reflection test", m_parameterTree);


}
