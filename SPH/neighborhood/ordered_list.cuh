#pragma once
#define ELEM_NUM 12
#define SM_ID(buffer, idx) buffer[threadIdx.x + idx * blockDim.x]
#define GM_ID(buffer, paritcle_idx, idx) buffer[_params.ALLSPACE_NUMPTCLS * idx + particle_idx]

template<typename _Data = float, typename _Integer = int16_t>
class ordered_list_highest {
	struct paired_value { _Integer idx; _Data data; };

	_Data* data_ptr;
	_Integer sm_size;
	_Integer elem_count;
	_Integer inserted_elements;
	paired_value comp_val;
public:
	__device__ ordered_list_highest(_Data* shared_mem, _Data default_comp, _Integer blockSize, _Integer elements) :data_ptr(shared_mem), sm_size(blockSize), elem_count(elements) {
		inserted_elements = 0;
		comp_val = paired_value{ 0, default_comp };
	}
	__device__ ordered_list_highest(_Data* shared_mem, _Data default_comp) : ordered_list_highest(shared_mem, default_comp, blockDim.x, ELEM_NUM) {}

	__device__ void print_elements() {
		printf("[");
		for (_Integer it = 0; it < elem_count; ++it)
		{
			printf("%f,", data_ptr[threadIdx.x + sm_size * it]);
		}
		printf("]\n");
	}

	__device__ void print_comp_val() {
		printf("Comparative Value: %f@%d\n", comp_val.data, comp_val.idx);
	}

	__device__ void push(_Data new_item) {
		if (inserted_elements < elem_count)
		{
			data_ptr[threadIdx.x + sm_size * inserted_elements] = new_item;
			if (new_item < comp_val.data)
				comp_val = paired_value{ inserted_elements, new_item };
			++inserted_elements;
		}
		else
		{
			if (new_item > comp_val.data)
			{
				data_ptr[threadIdx.x + sm_size * comp_val.idx] = new_item;
				comp_val = get_min();
			}
		}
	}

	__device__ paired_value get_min() {
		_Data min = _Data{ FLT_MAX };
		_Integer min_idx = 0;
		for (_Integer it = 0; it < elem_count; ++it) {
			auto current = data_ptr[threadIdx.x + sm_size * it];
			if (current < min)
			{
				min = current;
				min_idx = it;
			}
		}
		return paired_value{ min_idx, min };
	}
	__device__  paired_value get_max() {
		_Data min = _Data{ -FLT_MAX };
		_Integer min_idx = 0;
		for (_Integer it = 0; it < elem_count; ++it) {
			auto current = data_ptr[threadIdx.x + sm_size * it];
			if (current > min)
			{
				min = current;
				min_idx = it;
			}
		}
		return paired_value{ min_idx, min };
	}
	__device__ __inline__ void swap(_Data& rhs, _Data& lhs) {
		_Data tmp = rhs;
		rhs = lhs;
		lhs = tmp;
	}

	__device__  void sort_values_descendeding() {
		for (_Integer it = 0; it < elem_count; ++it) {
			paired_value min = paired_value{ 0, _Data{-FLT_MAX} };
			for (_Integer inner = it; inner < elem_count; ++inner)
			{
				_Data current = data_ptr[threadIdx.x + sm_size * inner];
				if (current > min.data) {
					min = paired_value{ inner, current };
				}
			}
			swap(data_ptr[threadIdx.x + sm_size * min.idx], data_ptr[threadIdx.x + sm_size * it]);
		}
	}
	__device__  void sort_values_ascendeding() {
		for (_Integer it = 0; it < elem_count; ++it) {
			paired_value min = paired_value{ 0, _Data{ FLT_MAX} };
			for (_Integer inner = it; inner < elem_count; ++inner)
			{
				_Data current = data_ptr[threadIdx.x + sm_size * inner];
				if (current < min.data) {
					min = paired_value{ inner, current };
				}
			}
			swap(data_ptr[threadIdx.x + sm_size * min.idx], data_ptr[threadIdx.x + sm_size * it]);
		}
	}
};

template<typename _Data = float, typename _Integer = int16_t>
class ordered_list_lowest {
public:

	struct paired_value { _Integer idx; _Data data; };

	_Data* data_ptr;
	_Integer sm_size;
	_Integer elem_count;
	_Integer inserted_elements;
	paired_value comp_val;

	__device__ ordered_list_lowest(_Data* shared_mem, _Data default_comp, _Integer blockSize, _Integer elements) :data_ptr(shared_mem), sm_size(blockSize), elem_count(elements) {
		inserted_elements = 0;
		comp_val = paired_value{ 0, default_comp };
	}
	__device__ ordered_list_lowest(_Data* shared_mem, _Data default_comp) : ordered_list_lowest(shared_mem, default_comp, blockDim.x, ELEM_NUM) {}

	__device__ void print_elements() {
		printf("[");
		for (_Integer it = 0; it < elem_count; ++it)
		{
			printf("%f,", data_ptr[threadIdx.x + sm_size * it]);
		}
		printf("]\n");
	}

	__device__ void print_comp_val() {
		printf("Comparative Value: %f@%d\n", comp_val.data, comp_val.idx);
	}

	__device__ void insert(_Data new_item) {
		if (inserted_elements < elem_count)
		{
			data_ptr[threadIdx.x + sm_size * inserted_elements] = new_item;
			if (new_item > comp_val.data)
				comp_val = paired_value{ inserted_elements, new_item };
			++inserted_elements;
		}
		else
		{
			if (new_item < comp_val.data)
			{
				data_ptr[threadIdx.x + sm_size * comp_val.idx] = new_item;
				comp_val = get_max();
			}
		}
	}

	__device__ paired_value get_min() {
		_Data min = FLT_MAX;
		_Integer min_idx = 0;
		for (_Integer it = 0; it < elem_count; ++it) {
			float current = data_ptr[threadIdx.x + sm_size * it];
			if (current < min)
			{
				min = current;
				min_idx = it;
			}
		}
		return paired_value{ min_idx, min };
	}
	__device__  paired_value get_max() {
		_Data min = -FLT_MAX;
		_Integer min_idx = 0;
		for (_Integer it = 0; it < elem_count; ++it) {
			float current = data_ptr[threadIdx.x + sm_size * it];
			if (current > min)
			{
				min = current;
				min_idx = it;
			}
		}
		return paired_value{ min_idx, min };
	}
	__device__ __inline__ void swap(float& rhs, float& lhs) {
		float tmp = rhs;
		rhs = lhs;
		lhs = tmp;
	}

	__device__  void sort_values_descendeding() {
		for (_Integer it = 0; it < elem_count; ++it) {
			paired_value min = paired_value{ 0, -FLT_MAX };
			for (_Integer inner = it; inner < elem_count; ++inner)
			{
				_Data current = data_ptr[threadIdx.x + sm_size * inner];
				if (current > min.data) {
					min = paired_value{ inner, current };
				}
			}
			swap(data_ptr[threadIdx.x + sm_size * min.idx], data_ptr[threadIdx.x + sm_size * it]);
		}
	}
	__device__  void sort_values_ascendeding() {
		for (_Integer it = 0; it < elem_count; ++it) {
			paired_value min = paired_value{ 0, FLT_MAX };
			for (_Integer inner = it; inner < elem_count; ++inner)
			{
				_Data current = data_ptr[threadIdx.x + sm_size * inner];
				if (current < min.data) {
					min = paired_value{ inner, current };
				}
			}
			swap(data_ptr[threadIdx.x + sm_size * min.idx], data_ptr[threadIdx.x + sm_size * it]);
		}
	}
};