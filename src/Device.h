//
// Created by qiaojinxia on 2023/11/8.
//

#include <stdexcept>
#ifndef ANTNET_DEVICE_H
#define ANTNET_DEVICE_H
#pragma once
class Tensor; // 前向声明

namespace AntNet {

    class DeviceException : public std::runtime_error {
    public:
        explicit DeviceException(const std::string &msg) : std::runtime_error(msg) {}
    };

    using ActivationType = int;

    class Device {
    public:
        virtual ~Device() = default;

        virtual void allocate_memory(Tensor &tensor) = 0;

        virtual void deallocate_memory(Tensor &tensor) = 0;

        virtual void copy_data_to_device(Tensor &tensor) = 0;

        virtual void copy_data_to_host(Tensor &tensor) = 0;

        virtual void set_device() = 0;

        virtual void synchronize() = 0;

    public:

        virtual void matmul(const Tensor &a, const Tensor &b, Tensor &out) = 0;

        virtual void add(const Tensor &a, const Tensor &b, Tensor &out) = 0;

        virtual void subtract(const Tensor &a, const Tensor &b, Tensor &out) = 0;

        virtual void apply_activation(const Tensor &a, Tensor &out, ActivationType type) = 0;
    };

    class CPUDevice : public Device {
    public:
        ~CPUDevice() override = default;

        void allocate_memory(Tensor &tensor) override;
        void deallocate_memory(Tensor &tensor) override;
        void copy_data_to_device(Tensor &tensor) override;
        void copy_data_to_host(Tensor &tensor) override;
        void set_device() override;
        void synchronize() override;

        void matmul(const Tensor &a, const Tensor &b, Tensor &out) override;
        void add(const Tensor &a, const Tensor &b, Tensor &out) override;
        void subtract(const Tensor &a, const Tensor &b, Tensor &out) override;
        void apply_activation(const Tensor &a, Tensor &out, ActivationType type) override;
    };

    class GPUDevice : public Device {
        void allocate_memory(Tensor &tensor) override;
        void deallocate_memory(Tensor &tensor) override;
        void copy_data_to_device(Tensor &tensor) override;
        void copy_data_to_host(Tensor &tensor) override;
        void set_device() override;
        void synchronize() override;

        void matmul(const Tensor &a, const Tensor &b, Tensor &out) override;
        void add(const Tensor &a, const Tensor &b, Tensor &out) override;
        void subtract(const Tensor &a, const Tensor &b, Tensor &out) override;
        void apply_activation(const Tensor &a, Tensor &out, ActivationType type) override;
    };

}

#endif //ANTNET_DEVICE_H
