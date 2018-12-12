#pragma once
#include <utility/MemoryManager.h>
#include <utility/memory/structs.h>

//#ifndef __INTELLISENSE__
template <typename... Ts> struct simulation {
private:
	template <std::size_t I = 0, typename T, typename... Tp>
	constexpr inline typename std::enable_if<I == sizeof...(Tp), void>::type
		assign_tuple(std::tuple<Tp...> &, std::tuple<Tp..., T> &) {}

	template <std::size_t I = 0, typename T, typename... Tp>
	constexpr inline typename std::enable_if <
		I<sizeof...(Tp), void>::type assign_tuple(std::tuple<Tp...> &t, std::tuple<Tp..., T> &u) {
		std::get<I>(u) = std::get<I>(t);
		assign_tuple<I + 1, T, Tp...>(t, u);
	}

public:
	std::tuple<Ts...> functions;
	template <typename T>
	simulation<Ts..., function_call<T>> then(void(*func)(T), std::string name = "", Color col = Color::azure, bool graph = true) {
		simulation<Ts..., function_call<T>> new_simulation;
		assign_tuple<0, function_call<T>, Ts...>(functions, new_simulation.functions);
		std::get<sizeof...(Ts)>(new_simulation.functions) = function_call<T>(func, name, col, graph);
		return new_simulation;
	}

	template <typename T> simulation<Ts..., array_clear<T>> then(array_clear<T> arr) {
		simulation<Ts..., array_clear<T>> new_simulation;
		assign_tuple<0, array_clear<T>, Ts...>(functions, new_simulation.functions);
		std::get<sizeof...(Ts)>(new_simulation.functions) = arr;
		return new_simulation;
	}
};
//#endif
