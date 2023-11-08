//
// Created by qiaojinxia on 2023/11/8.
//

#ifndef ANTNET_TENSOR_H
#define ANTNET_TENSOR_H

#pragma once
#include "Device.h"
#include <vector>

namespace AntNet{
    class Tensor {
        float* data{}; // 数据指针
        std::vector<size_t> shape; // 形状信息
        size_t totalSize{}; // 总元素数量
        Device* device{}; // 设备信息，指向数据存储的设备
    public:
        Tensor(const std::vector<size_t>& shape, Device* device);
        ~Tensor();
    };
}


#endif //ANTNET_TENSOR_H
