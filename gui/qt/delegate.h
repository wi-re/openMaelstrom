#pragma once
// There is an issue in the compiler that prevents classes with members with
// alignas specifiers to be linked causes unresolved external symbol on new and
// delete
#if defined(__MSC_VER__) && defined(__clang__)
#include <host_defines.h>
#undef __builtin_align__
#define __builtin_align__(x)
#endif

#include "ui_mainwindow.h"
#include <IO/config/parser.h>
#include <QComboBox>
#include <iostream>
#include <QLineEdit>
#include <QLabel>
#include <QFormLayout>
#include <QRadioButton>
#include <QTreeWidget>
#include <type_traits>
#include <utility/helpers/color.h>
#include <utility/helpers/log.h>
#include <utility/identifier/uniform.h>
#include <utility/math.h>
#include <vector>
#include <utility/template/for_struct.h>
using namespace IO::config;

/** Function used to convert numbers and vectors into strings for display, only
 * a wrapper but useful due to QString not working nicely with std::string. **/
template <typename T> QString to_qstring(T arg, char = 'g', int32_t = 4) {
  return QString::fromStdString(IO::config::convertToString(arg));
}
/** Helper class that is used for polymorphism as we want to store all delegates
 * in a single vector which requires them all having a common baseclass. Even
 * though this could be achieved using QWidget, in practice this approach gives
 * us more freedom was we can freely add virtual methods that resolve as needed
 * at runtime instead of relying on existing methods in Qt. Additionally we can
 * make this base class a Q_OBJECT as the actual delegates are templates. **/
class value_delegate : public QWidget {
  Q_OBJECT
public:
  value_delegate(QWidget *parent = nullptr) : QWidget(parent) {}
  virtual void update() = 0;
};
/** This class handles displaying the GUI elements to display and edit
 * configuration parameters at runtime. This is achieved mostly by using if
 * constexpr expression based on the template parameter. **/
template <typename type> class template_delegate : public value_delegate {
protected:
  /** This member stores a reference to the original value. If we only used
   * existing parameters this would not be required as info::ptr could provided
   * the needed reference. However as we might want to add other variables to
   * the editor which have no underlying meta structure we need to store the
   * pointer here. **/
  type *m_data = nullptr;

  /** GUI ELements**/
  QFormLayout *m_layout = nullptr;
  QComboBox *m_presetComboBox = nullptr;
  QLineEdit *m_label = nullptr;
  std::vector<QSlider *> m_sliders;
  std::map<QSlider *, int32_t> m_lastSliders;

  /** Meta informaiton about the value, stored due to a similar reason as m_data
   * as we need to access this information but specify the template on T not
   * info. **/
  type m_minValue;
  type m_maxValue;
  type m_stepValue;
  type m_oldValue;

public:
  /** Constructor used for parameters from the simulation with the proper meta
   * type **/
  template <typename info>
  template_delegate(info, std::string prefix = "v: ")
      : value_delegate(nullptr), m_data(info::ptr) {
    // meta information, T should be equal to type usually.
    using T = typename info::type;
    constexpr uint32_t dimension = math::dimension<T>::value;
    //constexpr bool has_presets =
    //    !std::is_same_v<std::decay_t<presets_type<info>>, std::ptrdiff_t>;
    //constexpr bool has_step =
    //    !std::is_same<std::decay_t<step_type<info>>, std::ptrdiff_t>::value;

    // Setup a basic layout that displays the current value of the parameter in
    // a line edit.
    m_layout = new QFormLayout;
    m_layout->setContentsMargins(0, 0, 0, 0);
    m_layout->setSpacing(0);
    m_layout->addRow(new QLabel(QString::fromStdString(prefix)),
                     m_label = new QLineEdit(to_qstring(*info::ptr)));
    if (info::modifiable == false)
      m_label->setReadOnly(true);
    m_label->setAlignment(Qt::AlignRight);

    // Additional UI elements are only created if the value is editable.
    if constexpr (info::modifiable == true) {
      // Add a combo box if there are any preset values stored as meta data
		auto arr = info::getPresets();
		if (arr.size() != 0) {
			m_presetComboBox = new QComboBox();
			m_layout->addRow(new QLabel(""), m_presetComboBox);
			for (auto p : arr)
				m_presetComboBox->addItem(QString::fromStdString(p));
		}

      //if constexpr (!std::is_same_v<std::decay_t<presets_type<info>>, std::ptrdiff_t>) {
      //  m_presetComboBox = new QComboBox();
      //  m_layout->addRow(new QLabel(""), m_presetComboBox);
      //  constexpr auto arr = info::presets;
      //  for (auto p : arr)
      //    m_presetComboBox->addItem(p);
      //}
      // Add a slider for every dimension of data (e.g. 4 for a 4d value)
      if constexpr (min_type_v<info> && max_type_v<info>) {
        m_minValue = info::min;
        m_maxValue = info::max;
        // Helper lambda as this logic is required for every dimension
        auto add_slider = [&](auto label, auto fn) {
          auto slider = new QSlider(Qt::Horizontal);
          slider->setFocusPolicy(Qt::StrongFocus);
          slider->setTickPosition(QSlider::TicksBothSides);
          slider->setTickInterval(10);
          slider->setSingleStep(1);
          slider->blockSignals(true);
          m_layout->addRow(new QLabel(label), slider);
          if (!std::is_same<std::decay_t<step_type<info>>, std::ptrdiff_t>::value) {
            auto diff = (m_maxValue) - (m_minValue);
            auto steps = diff / (m_stepValue);
			if (std::is_integral_v<typename info::type>) {
				slider->setMaximum(fn(m_maxValue));
				slider->setMinimum(fn(m_minValue));
			}
			else
				slider->setMaximum(static_cast<int>(fn(steps)) + 1);
          }
          auto min_s = slider->minimum();
          auto max_s = slider->maximum();
          auto slope = static_cast<decltype(fn(m_maxValue))>(max_s - min_s) /
                       ((m_maxValue) - (m_minValue));
          auto output = min_s + slope * (*m_data - (m_minValue));
          slider->setSliderPosition(static_cast<int>(fn(output)));
          slider->blockSignals(false);
          m_sliders.push_back(slider);
          return slider;
        };
        // If there is meta information about a step distance set this value
        if constexpr (!std::is_same<std::decay_t<step_type<info>>, std::ptrdiff_t>::value) {
          constexpr type t_step = info::step;
          m_stepValue = t_step;
        }
        // Create the actual sliders.
        if constexpr (dimension == 0)
          add_slider("x: ", [](auto x) { return x; });
        if constexpr (dimension > 0)
          add_slider("x: ", [](auto x) { return x.x; });
        if constexpr (dimension > 1)
          add_slider("y: ", [](auto x) { return x.y; });
        if constexpr (dimension > 2)
          add_slider("z: ", [](auto x) { return x.z; });
        if constexpr (dimension > 3)
          add_slider("w: ", [](auto x) { return x.w; });
      }
    }

    // Store the current value of the parameter and update the slider values so
    // we can check for updates easily.
    m_oldValue = *info::ptr;
    for (auto slider : m_sliders)
      m_lastSliders[slider] = slider->value();

    setLayout(m_layout);

    // Create callbacks that are called on moving the sliders changing a preset
    // or when editing the line edit.
    for (auto slider : m_sliders)
      connect(slider,
              static_cast<void (QSlider::*)(int)>(&QSlider::valueChanged),
              [=]() { update_value(*m_data); });
    connect(m_label,
            static_cast<void (QLineEdit::*)()>(&QLineEdit::editingFinished),
            [=]() { edit_finished(); });
    if (m_presetComboBox != nullptr)
      connect(m_presetComboBox,
              static_cast<void (QComboBox::*)(const QString &)>(
                  &QComboBox::currentIndexChanged),
              [=](const QString &text) { update_combo(text); });
  }
  /** Constructor used for non parameters without meta information. **/
  template_delegate(type *d, std::string label = "v:")
      : value_delegate(nullptr), m_data(d) {
    // Create a basic layout
    m_layout = new QFormLayout;
    m_layout->setContentsMargins(0, 0, 0, 0);
    m_layout->setSpacing(0);
    m_layout->addRow(new QLabel(QString::fromStdString(label)),
                     m_label = new QLineEdit(to_qstring(*d)));
    m_label->setAlignment(Qt::AlignRight);

    // Update the stored value
    m_oldValue = *m_data;

    setLayout(m_layout);

    // The only callback we can create is one for editing the line edit.
    connect(m_label,
            static_cast<void (QLineEdit::*)()>(&QLineEdit::editingFinished),
            [=]() { edit_finished(); });
  }
  /** Callback for selecting a new preset value from a combo box. The combo box
   * contains the string representations which are transformed back into actual
   * values if this conversion is possible. **/
  void update_combo(QString input) {
    auto conversion =
        IO::config::convertStringChecked<type>(input.toStdString());
    if (conversion.first) {
      *m_data = conversion.second;
      m_label->setText(input);
      update_slider(*m_data);
    } else
      LOG_WARNING << "invalid input entered, could not convert "
                  << input.toStdString();
  }
  /** Callback used for reading a value from the sliders. **/
  template <typename T> void update_value(T &value) {
    constexpr auto dim = math::dimension<T>::value;
    if (m_sliders.size() > 0)
      if constexpr (dim != 0xDEADBEEF && !std::is_same<T, bool>::value) {
        auto slider_to_value = [&](auto slider, auto fn) {
			if constexpr(std::is_integral_v<decltype(fn(value))>) {
				//std::cout << value << std::endl;
				auto output = slider->value();
				//if (output == m_lastSliders[slider])
				//	return fn(value);
				return output;
			}
			else {
				auto min_s = slider->minimum();
				auto max_s = slider->maximum();
				auto new_x = slider->value();
				if (new_x == m_lastSliders[slider])
					return fn(value);
				auto slope = (m_maxValue - m_minValue) /
					(static_cast<decltype(fn(value))>(max_s - min_s));
				auto output =
					m_minValue +
					slope * (new_x - static_cast<decltype(fn(value))>(min_s));
				return fn(output);
			}
        };
        if constexpr (dim == 0)
          value = slider_to_value(m_sliders[0], [](auto v) { return v; });
        if constexpr (dim > 0)
          value.x = slider_to_value(m_sliders[0], [](auto v) { return v.x; });
        if constexpr (dim > 1)
          value.y = slider_to_value(m_sliders[1], [](auto v) { return v.y; });
        if constexpr (dim > 2)
          value.z = slider_to_value(m_sliders[2], [](auto v) { return v.z; });
        if constexpr (dim > 3)
          value.w = slider_to_value(m_sliders[3], [](auto v) { return v.w; });
        for (auto slider : m_sliders)
          m_lastSliders[slider] = slider->value();
      }
    m_oldValue = value;
    m_label->setText(to_qstring(value));
  }
  /** Callback used for moving sliders to the correct position. **/
  template <typename T> void update_slider(T &value) {
    constexpr auto dim = math::dimension<T>::value;
    if (m_sliders.size() > 0)
      if constexpr (dim != 0xDEADBEEF && !std::is_same_v<T, bool>) {
        auto slider_to_value = [&](auto slider, auto fn) {
          auto min_s = slider->minimum();
          auto max_s = slider->maximum();
          auto slope =
              static_cast<decltype(fn(std::declval<T>()))>(max_s - min_s) /
              (m_maxValue - m_minValue);
          auto output = min_s + slope * (*m_data - m_minValue);
          slider->blockSignals(true);
          slider->setSliderPosition(static_cast<int>(fn(output)));
          slider->blockSignals(false);
        };
        if constexpr (dim == 0)
          slider_to_value(m_sliders[0], [](auto v) { return v; });
        if constexpr (dim > 0)
          slider_to_value(m_sliders[0], [](auto v) { return v.x; });
        if constexpr (dim > 1)
          slider_to_value(m_sliders[1], [](auto v) { return v.y; });
        if constexpr (dim > 2)
          slider_to_value(m_sliders[2], [](auto v) { return v.z; });
        if constexpr (dim > 3)
          slider_to_value(m_sliders[3], [](auto v) { return v.w; });
        for (auto slider : m_sliders)
          m_lastSliders[slider] = slider->value();
      }
    m_oldValue = value;
  }
  /** Callback used for editing the line editor, logs a warning if the value
   * could not be converted to type. **/
  void edit_finished() {
    QString input = m_label->text();
    auto conversion =
        IO::config::convertStringChecked<type>(input.toStdString());
    if (conversion.first) {
      *m_data = conversion.second;
      update_slider(*m_data);
    } else
      LOG_WARNING << "invalid input entered, could not convert "
                  << input.toStdString();
  }

public:
  /** Callback function for Qt used to update the display along with the generic
   * UI refresh that Qt does automatically. **/
  void update() {
    if (m_oldValue == *m_data)
      return;
    update_slider(*m_data);
    update_value(*m_data);
  }
};
/** This function is used to help with the template deduction of
 * template_delegate and handles creating multiple sub entries for complex
 * values. The actual delegate itself can only handle a 1:1 relationship,
 * meaning 1 delegate for 1 parameter. Complex values however require multiple
 * entries. This function also handles generic types by using for_struct_fn. **/
template <typename T>
std::vector<value_delegate *> create_delegate(T *ptr, std::string prefix = "v: ") {
  value_delegate *del = nullptr;

  if constexpr (std::is_same<uniform_type_template<T>,
                             complex_uniform>::value) {
    std::vector<value_delegate *> delegates;
    for_struct_fn(*ptr, [&](auto &elem) {
      auto wid = create_delegate(&elem.value, elem.jsonName);
      for (auto w : wid)
        delegates.push_back(w);
    });
    return delegates;
  } else if constexpr (std::is_same_v<std::string, T>) {
    del = new template_delegate<std::decay_t<decltype(*ptr)>>(ptr, prefix);
  } else if constexpr (math::dimension<T>::value != 0xDEADBEEF) {
    del = new template_delegate<T>(ptr, prefix);
  } else if constexpr (IO::config::is_std_array<T>::value) {
    del = new template_delegate<T>(ptr, prefix);
  } else {
    std::vector<value_delegate *> delegates;
    for_struct_fn(*ptr, [&](auto &elem) {
      auto wid = create_delegate(&elem);
      for (auto w : wid)
        delegates.push_back(w);
    });
    return delegates;
  }
  return {del};
}
/** Simple wrapper around parameters that hides non visible parameters in the
 * GUI. In case a parameter is hidden this function returns an empty vector. **/
template <typename T> std::vector<value_delegate *> create_delegate() {
  if constexpr (!T::visible)
    return std::vector<value_delegate *>{};
  return {new template_delegate<std::decay_t<typename T::type>>(T{})};
}
/** This function creates a QTreeWidgetItem for a generic value. **/
template <typename T>
std::vector<value_delegate *> addToTree(T *ptr, std::string identifier,
                                        QTreeWidget *ls) {
  auto dels = create_delegate(ptr);
  std::vector<value_delegate *> delegates;
  QTreeWidgetItem *item = new QTreeWidgetItem();

  ls->addTopLevelItem(item);
  ls->setItemWidget(item, 0, new QLabel(QString::fromStdString(identifier)));
  for (auto w : dels) {
    if (w != nullptr) {
      QTreeWidgetItem *item2 = new QTreeWidgetItem();
      item->addChild(item2);
      ls->setItemWidget(item2, 0, w);
      delegates.push_back(w);
    }
  }
  return delegates;
}
/** This function creates a QTreeWidgetItem for a parameter. **/
template <typename Ty>
std::vector<value_delegate *>
addToTree(QTreeWidget *ls, std::map<std::string, QTreeWidgetItem *> &top) {
  if constexpr (is_vector<typename Ty::type>::value) {
    using elem_t = std::decay_t<decltype((*Ty::ptr)[0])>;
    std::vector<value_delegate *> delegates;
    {
      std::string str = std::string(Ty::jsonName);
      auto tok = split(str, '.');
      QTreeWidgetItem *item = new QTreeWidgetItem();
      top[tok[0]]->addChild(item);
      ls->setItemWidget(item, 0, new QLabel(QString::fromStdString(tok[1])));

      for (uint64_t i = 0; i < Ty::ptr->size(); ++i) {
        auto &elem = (*Ty::ptr)[i];
        if constexpr (std::is_same<uniform_type_template<elem_t>,
                                   complex_uniform>::value) {
          QTreeWidgetItem *nitem2 = new QTreeWidgetItem();
          item->addChild(nitem2);
          QString qs("Element ");
          qs.append(QString::number(i));
          ls->setItemWidget(nitem2, 0, new QLabel(qs));
          auto wid = create_delegate(&elem);
          for (auto w : wid) {
            if (w != nullptr) {
              QTreeWidgetItem *item2 = new QTreeWidgetItem();
              nitem2->addChild(item2);
              ls->setItemWidget(item2, 0, w);
              delegates.push_back(w);
            }
          }
        } else {
          std::string qs = std::to_string(i);
          qs.append(": ");
          auto wid = create_delegate(&elem, qs);
          for (auto w : wid) {
            if (w != nullptr) {
              QTreeWidgetItem *item2 = new QTreeWidgetItem();
              item->addChild(item2);
              ls->setItemWidget(item2, 0, w);
              delegates.push_back(w);
            }
          }
        }
      }
    }
    return delegates;
  } else if constexpr (std::is_same<uniform_type_template<typename Ty::type>,
                                    complex_uniform>::value) {
    auto wid = create_delegate<Ty>();

    std::string str = std::string(Ty::jsonName);
    auto tok = split(str, '.');

    QTreeWidgetItem *item = new QTreeWidgetItem();
    top[tok[0]]->addChild(item);
    ls->setItemWidget(item, 0, new QLabel(QString::fromStdString(tok[1])));
    for (auto w : wid) {
      if (w != nullptr) {
        QTreeWidgetItem *item2 = new QTreeWidgetItem();

        item->addChild(item2);
        ls->setItemWidget(item2, 0, w);
      }
    }
  } else {
    auto wid = create_delegate<Ty>();
    if (wid.size() > 0)
      if (wid[0] != nullptr) {

        std::string str = std::string(Ty::jsonName);
        auto tok = split(str, '.');

        QTreeWidgetItem *item = new QTreeWidgetItem();
        top[tok[0]]->addChild(item);
        ls->setItemWidget(item, 0, new QLabel(QString::fromStdString(tok[1])));

        QTreeWidgetItem *item2 = new QTreeWidgetItem();

        item->addChild(item2);
        ls->setItemWidget(item2, 0, wid[0]);
      }
    return wid;
  }
}
