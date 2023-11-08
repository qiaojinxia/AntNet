//
// Created by qiaojinxia on 2023/11/8.
//

#include "Device.h"

namespace AntNet{

    void CPUDevice::allocate_memory(Tensor &tensor) {

    }

    void CPUDevice::deallocate_memory(Tensor &tensor) {

    }

    void CPUDevice::copy_data_to_device(Tensor &tensor) {

    }

    void CPUDevice::copy_data_to_host(Tensor &tensor) {

    }

    void CPUDevice::set_device() {

    }

    void CPUDevice::synchronize() {

    }

    void CPUDevice::matmul(const Tensor &a, const Tensor &b, Tensor &out) {

    }

    void CPUDevice::add(const Tensor &a, const Tensor &b, Tensor &out) {

    }

    void CPUDevice::subtract(const Tensor &a, const Tensor &b, Tensor &out) {

    }

    void CPUDevice::apply_activation(const Tensor &a, Tensor &out, AntNet::ActivationType type) {

    }

    void GPUDevice::allocate_memory(Tensor &tensor) {

    }

    void GPUDevice::deallocate_memory(Tensor &tensor) {

    }

    void GPUDevice::copy_data_to_device(Tensor &tensor) {

    }

    void GPUDevice::copy_data_to_host(Tensor &tensor) {

    }

    void GPUDevice::set_device() {

    }

    void GPUDevice::synchronize() {

    }

    void GPUDevice::matmul(const Tensor &a, const Tensor &b, Tensor &out) {

    }

    void GPUDevice::add(const Tensor &a, const Tensor &b, Tensor &out) {

    }

    void GPUDevice::apply_activation(const Tensor &a, Tensor &out, AntNet::ActivationType type) {

    }

    void AntNet::GPUDevice::subtract(const Tensor &a, const Tensor &b, Tensor &out) {

    }

}
