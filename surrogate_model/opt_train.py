import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import torch.multiprocessing as mp
import argparse
import logging
import pdb
import json

import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import random
from datetime import datetime
torch.backends.cudnn.enabled = False
mp.set_start_method('spawn', force=True)
torch.set_printoptions(precision=6)
# 设置随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# 设置CUDA的随机种子
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ExpDataset(Dataset):
    def __init__(self, input_json_path):
        self.input_json_path = input_json_path
        self.features, self.labels = self.load_data(self.input_json_path)

    def load_data(self, json_file):
        with open(json_file, 'r') as json_file:
            data = [json.loads(line) for line in json_file]

        df = pd.DataFrame(data)
        # Separate input features and output column
        features = df[[col for col in df.columns if col != "log10_IDT"]].values
        label = df["log10_IDT"].values
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return feature_tensor, label_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FC4AddSkip(nn.Module):
    def __init__(self, input_size=623, hidden_sizes=(512, 256, 256), output_size=1):
        super(FC4AddSkip, self).__init__()
        # 定义网络层
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.silu = nn.SiLU()
        self.skip1 = nn.Linear(input_size, hidden_sizes[0])  # 第一层的跳跃连接
        self.skip2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])  # 第二层的跳跃连接
        self.skip3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])  # 第三层的跳跃连接

    def forward(self, x):
        # pdb.set_trace()
        # 第一层
        residual1 = self.skip1(x)  # 跳跃连接
        x = self.silu(self.fc1(x) + residual1)

        # 第二层
        residual2 = self.skip2(x)  # 跳跃连接
        x = self.silu(self.fc2(x) + residual2)

        # 第三层
        residual3 = self.skip3(x)  # 跳跃连接
        x = self.silu(self.fc3(x) + residual3)

        # 第四层（无跳跃连接）
        x = self.fc4(x)  # 输出层没有跳跃连接
        return x


class PartialGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, params_tensor, mask):
        # 保存 mask 用于反向传播
        ctx.save_for_backward(mask)
        return params_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # 加载 mask
        mask, = ctx.saved_tensors
        # 只对 mask=1 的位置保留梯度，其余位置的梯度为 0
        grad_input = grad_output * mask
        return grad_input, None  # 第二个 None 是因为 mask 不需要梯度


class CustomModel(nn.Module):
    def __init__(self, pretrained_model, params_tensor, mask_tensor, param_bounds, means, stds, lnA_mean, lnA_std):
        """"
        means 和 stds 是用于 正则化 的均值和标准差张量
        lnA_mean 和 lnA_std 是用于 逆标准化 的均值和标准差张量
        """
        super(CustomModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = pretrained_model.to(device)
        # 冻结预训练模型的参数
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        self.params_tensor = torch.nn.Parameter(params_tensor.clone().to(device))  # 将参数张量作为 nn.Parameter
        self.register_buffer("mask", mask_tensor.to(device))  # 注册掩码为不可训练的张量
        self.register_buffer("param_bounds", param_bounds)   # 参数上下界
        self.register_buffer("means", means.to(device))  # 正则化使用
        self.register_buffer("stds", stds.to(device))  # 正则化使用
        self.register_buffer("lnA_mean", lnA_mean.to(device))
        self.register_buffer("lnA_std", lnA_std.to(device))
        # 打印加载的参数以验证
        # for name, param in self.pretrained_model.named_parameters():
        #     print(f"In function~~~{name}: requires_grad = {param.requires_grad}, value = {param[:5]}")

    def forward(self, batch_x):
        params_with_partial_grad = PartialGradientFunction.apply(self.params_tensor, self.mask)  # torch.Size([615])

        params_with_partial_grad = params_with_partial_grad.unsqueeze(0).repeat(batch_x.size(0), 1)  # torch.Size([16, 615])
        # 拼接 (batch_size, input_dim + params_dim)
        concatenated_input = torch.cat((batch_x, params_with_partial_grad), dim=1)  # torch.Size([32, 623])
        # pdb.set_trace()
        output = self.pretrained_model(concatenated_input)
        return output

    def regularization_loss(self, regularization_strength=0.1):
        """
        L2 regularization on original (inverse-standardized) parameters.
        Args:
            params_tensor (torch.Tensor): 当前标准化的参数张量。
            param_mean (torch.Tensor): lnA均值张量。
            param_std (torch.Tensor): lnA标准差张量。
            regularization_strength (float): 正则化强度。

        Returns:
            torch.Tensor: 正则化损失值。
        """
        params_original = self.params_tensor * self.lnA_std + self.lnA_mean  # 逆标准化
        # 计算正则化损失
        mask_applied_params = params_original[self.mask == 1]
        normalized_deviation = ((mask_applied_params - self.means) ** 2) / (self.stds ** 2)  # 用的是mech08的均值
        # pdb.set_trace()
        reg_loss = torch.sum(normalized_deviation)
        # pdb.set_trace()
        return regularization_strength * reg_loss

    def clamp_params(self):
        # 对 params_tensor 的梯度裁剪
        with torch.no_grad():
            # 获取 mask 为 1 的索引
            mask_indices = torch.where(self.mask == 1)[0]
            # print(f"mask_indices: {mask_indices}")
            for idx, param_idx in enumerate(mask_indices):
                lower_bound, upper_bound = self.param_bounds[idx]
                # print(f"original {idx}: {self.params_tensor[param_idx]}")
                lower_bound = lower_bound.clone().detach().to(self.params_tensor.device)
                upper_bound = upper_bound.clone().detach().to(self.params_tensor.device)
                # 裁剪参数
                self.params_tensor[param_idx] = self.params_tensor[param_idx].clamp(lower_bound, upper_bound)
                # pdb.set_trace()
                # print(f"after clamp: {self.params_tensor[param_idx]}")


def denormalize(tensor, mean, std, eps=1e-8):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=tensor.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=tensor.device)
    mean_clone = mean.clone().detach().to(tensor.device)  # 使用 clone().detach()
    std_clone = std.clone().detach().to(tensor.device)  # 使用 clone().detach()
    return tensor * (std_clone + eps) + mean_clone


def prepare_mask_parameters(mech_params_path, opt_config_path, num_groups=205):
    """
        Args:
            mech_params_path (str): Path to the all lnA, n ,Ea parameters.
            opt_config_path (str): contain optimizing lnA's min/max.
            num_groups (int): reactions' number. lnA, n, Ea.

        Returns:
            params_tensor (torch.Tensor): 所有参数的张量，维度为 (num_groups * 3, )。
            mask (torch.Tensor): 掩码张量，标记哪些参数需要梯度计算（1 表示可导，0 表示固定）。
            params_bounds (torch.Tensor): 参数的上下界张量，维度为 (num_groups, 2)。
            param_means (torch.Tensor): lnA 的均值张量，用于正则化。是字典，一维张量。
        """
    with open(mech_params_path, "r") as mech_file:
        mech_params = json.load(mech_file)

    # 读取 1.json, 包含min和max
    with open(opt_config_path, "r") as opt_file:
        optimization_params = [json.loads(line) for line in opt_file.readlines()]

    # 构建优化参数的映射表
    optimization_dict = {entry["lnA"]: entry for entry in optimization_params}
    params_tensor = []
    mask = []
    params_bounds = {}
    param_means = {}

    # 遍历所有组，1个组代表1个基元反应，按顺序处理
    for i in range(1, num_groups + 1):
        lnA_key = f"lnA{i}"
        n_key = f"n{i}"
        Ea_key = f"Ea{i}"

        n_value = mech_params.get(n_key, 0.0)
        Ea_value = mech_params.get(Ea_key, 0.0)

        if lnA_key in optimization_dict:
            lnA_min = optimization_dict[lnA_key]["min"]
            lnA_max = optimization_dict[lnA_key]["max"]
            param_idx = len(params_tensor)  # 当前参数的索引
            params_bounds[param_idx] = (lnA_min, lnA_max)

            lnA_random = mech_params.get(lnA_key, 0.0)  # 随机初始化优化参数, 此处先不随机，看一下效果
            params_tensor.append(lnA_random)
            mask.append(1)  # 掩码为 1，表示需要梯度
            param_means[param_idx] = mech_params.get(lnA_key, 0.0)  # 保存lnA本身作为后续优化的均值
            # print(f"  lnA (optimized): {lnA_normalized}")
        else:
            # 固定参数
            lnA_value = mech_params.get(lnA_key, 0.0)
            params_tensor.append(lnA_value)
            mask.append(0)  # 掩码为 0，表示固定
            # print(f"lnA (fixed): {lnA_normalized}")

        params_tensor.append(n_value)  # n
        mask.append(0)  # n 不需要优化，掩码为 0
        # print(f"n: {n_normalized}")

        params_tensor.append(Ea_value)  # Ea
        mask.append(0)  # Ea 不需要优化，掩码为 0
        # print(f"Ea: {Ea_normalized}")

    params_tensor = torch.tensor(params_tensor, dtype=torch.float32)
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    param_means = torch.tensor(list(param_means.values()), dtype=torch.float32)

    # 提取上下界
    bounds_min = [bounds[0] for bounds in params_bounds.values()]
    bounds_max = [bounds[1] for bounds in params_bounds.values()]

    # 转换为张量
    bounds_min_tensor = torch.tensor(bounds_min, dtype=torch.float32)
    bounds_max_tensor = torch.tensor(bounds_max, dtype=torch.float32)

    # 合并为一个张量
    params_bounds_tensor = torch.stack((bounds_min_tensor, bounds_max_tensor), dim=1)

    # print("Bounds Tensor Shape:", params_bounds_tensor.shape)
    # print("Bounds Tensor:\n", params_bounds_tensor)

    return params_tensor, mask_tensor, params_bounds_tensor, param_means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch model_2 Training')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: ", device)
    # data = 'input.json'  # 记得修改 Exp_dataset 里的字段log10_IDT,意思是数据文件、标准化值文件、函数提取名应该一致
    train_data = 'Exp_data_idt/train_data.json'
    valid_data = 'Exp_data_idt/val_data.json'
    train_dataset = ExpDataset(train_data)
    valid_dataset = ExpDataset(valid_data)
    test_dataset = ExpDataset('Exp_data_idt/test_data.json')
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    params_file = "C:/Users/admin/Desktop/241220_paper/outs/05mech_norm_input/norm_mech_02_params.json"
    # 提取关键字 'mech_01' 从路径中
    keyword = params_file.split('/')[-1].split('_')[1]  # 使用路径分割和下划线分割来提取 'mech_01'

    # 从 20 中提取均值
    params_file_08 = "C:/Users/admin/Desktop/241220_paper/outs/05mech_norm_input/norm_mech_20_params.json"

    print(f"Processing params file: {params_file}")
    optimization_path = "bound_lnA_norm.json"  # lnA的上下界
    # 用于逆标准化
    lnA_mean = 26.93702
    lnA_std = 19.28485
    lnA_mean_tensor = torch.tensor(lnA_mean).to(device)
    lnA_std_tensor = torch.tensor(lnA_std).to(device)

    pretrained_model = FC4AddSkip().to(device)
    checkpoint = torch.load('ckpt_0207.pth', map_location=device, weights_only=True)
    state_dict = checkpoint["net"]
    pretrained_model.load_state_dict(state_dict, strict=True)
    # 冻结预训练模型的参数
    for param in pretrained_model.parameters():
        param.requires_grad = False
    criterion = nn.MSELoss(reduction='none')
    mean_log_y = 2.832582
    std_log_y = 0.456879

    par, ma, par_bou, param_lnA_means = prepare_mask_parameters(
        params_file_08, optimization_path, num_groups=205)  # 从08中提取均值 param_lnA_means

    param_lnA_means = param_lnA_means.to(device) * lnA_std_tensor + lnA_mean_tensor  # 逆标准化

    params, mask, param_bounds, param_lnA_means_2 = prepare_mask_parameters(
        params_file, optimization_path, num_groups=205)

    # 用于正则化
    param_lnA_stds = torch.full((20,), 5.22)

    pretrained_model.eval()
    model = CustomModel(pretrained_model, params, mask, param_bounds,
                        param_lnA_means, param_lnA_stds,
                        lnA_mean_tensor, lnA_std_tensor)  # 前两个用于正则化，后两个用于逆标准化
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    epochs = 3000
    start_time = time.time()
    print(f"Starting training! -- >>")
    # 初始化最小损失值和对应的 epoch
    min_loss = float('inf')  # 初始化为无穷大，表示最小值尚未计算
    best_epoch = -1  # 存储最低损失对应的 epoch
    model.eval()
    train_avg_standard_loss = []
    valid_avg_standard_loss = []
    train_avg_inverse_loss = []
    valid_avg_inverse_loss = []
    for epoch in range(epochs):
        model.train()
        train_total_samples = 0
        train_total_loss = 0
        train_total_inverse_loss = 0
        idt_total_loss = 0

        for batch_index, (batch_x, batch_y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y = batch_y.unsqueeze(-1)
            outputs = model(batch_x)
            task_loss = criterion(outputs, batch_y)  # task_loss
            reg_loss = model.regularization_loss(regularization_strength=1)
            loss = task_loss.mean() + reg_loss  # task_loss 是总和
            # 计算逆标准化后的损失
            outputs_inverse = denormalize(outputs, mean_log_y, std_log_y)
            batch_y_inverse = denormalize(batch_y, mean_log_y, std_log_y)
            batch_inverse_loss = criterion(outputs_inverse, batch_y_inverse)

            train_total_samples += batch_y.size(0)
            train_total_loss += task_loss.sum()
            train_total_inverse_loss += batch_inverse_loss.sum()
            loss.backward()
            optimizer.step()
            model.clamp_params()

        avg_standard_loss_per_epoch = train_total_loss / train_total_samples

        avg_inverse_loss_per_epoch = train_total_inverse_loss / train_total_samples

        train_avg_standard_loss.append(avg_standard_loss_per_epoch.cpu().item())
        train_avg_inverse_loss.append(avg_inverse_loss_per_epoch.cpu().item())
        print(f"Training -->> Epoch: {epoch + 1},"
              f"(no reg loss)standard loss: {avg_standard_loss_per_epoch:.7f}, "
              f"inverse loss: {avg_inverse_loss_per_epoch:.7f}")

        # 验证阶段
        model.eval()
        valid_total_samples = 0
        valid_total_inverse_loss = 0
        valid_total_standardized_loss = 0

        with torch.no_grad():
            for batch_index, (batch_x, batch_y) in enumerate(valid_dataloader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                batch_y = batch_y.unsqueeze(-1)
                valid_standardized_loss = criterion(outputs, batch_y)

                outputs_inverse = denormalize(outputs, mean_log_y, std_log_y)
                batch_y_inverse = denormalize(batch_y, mean_log_y, std_log_y)
                valid_inverse_loss = criterion(outputs_inverse, batch_y_inverse)

                valid_total_standardized_loss += valid_standardized_loss.sum()
                valid_total_inverse_loss += valid_inverse_loss.sum()
                valid_total_samples += batch_y.size(0)

        # 计算并存储每个 epoch 的平均逆标准化损失
        avg_standardized_valid_loss_per_epoch = valid_total_standardized_loss / valid_total_samples
        avg_inverse_valid_loss_per_epoch = valid_total_inverse_loss / valid_total_samples

        valid_avg_standard_loss.append(avg_standardized_valid_loss_per_epoch.cpu().item())
        valid_avg_inverse_loss.append(avg_inverse_valid_loss_per_epoch.cpu().item())

        print(
            f'Valid-->> Epoch [{epoch + 1}/{epochs}], '
            f'Standardized Loss: {avg_standardized_valid_loss_per_epoch:.7f}, '
            f'Inverse Loss: {avg_inverse_valid_loss_per_epoch:.7f}')
        # 监控并记录最低损失值及对应的 epoch
        if avg_inverse_valid_loss_per_epoch < min_loss:
            min_loss = avg_inverse_valid_loss_per_epoch
            best_epoch = epoch + 1
            # 打印损失最低的 epoch 和相应的参数
            print(f"Valid-->> Lowest loss found at epoch {best_epoch}, loss: {min_loss:.7f}")

            # 获取 mask 为 1 的 params_tensor
            masked_params = model.params_tensor[model.mask == 1]
            masked_params_inverse = masked_params * lnA_std_tensor + lnA_mean_tensor  # 逆标准化

            print(f"Epoch {best_epoch}, Masked params (inverse standardized): {masked_params_inverse}")

            masked_params_inverse_list = masked_params_inverse.cpu().tolist()
            # 保存到 JSON 文件
            data_to_save = {
                "epoch": best_epoch,
                "min_avg_inverse_valid_loss": min_loss.cpu().item(),
                "masked_params_inverse": masked_params_inverse_list
            }
            # 保存为 JSON 文件
            with open('best_model_params_just_idt.json', 'w') as f:
                json.dump(data_to_save, f, indent=4)

    # 结束计时
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total prediction time: {elapsed_time:.2f} seconds")

    # 绘制avg_inverse_loss_per_epoch随epoch变化的图
    plt.plot(range(epochs), train_avg_standard_loss, marker='', linestyle='-', color='b', label='Training Loss')
    plt.plot(range(epochs), valid_avg_standard_loss, marker='', linestyle='-', color='r', label='Validation Loss')
    plt.title('Average Standard Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Average Standard Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.plot(range(epochs), train_avg_inverse_loss, marker='', linestyle='-', color='b', label='Training Loss')
    plt.plot(range(epochs), valid_avg_inverse_loss, marker='', linestyle='-', color='r', label='Validation Loss')
    plt.title(f'Average Inverse Loss per Epoch (best epoch: {best_epoch}, loss: {min_loss:.6f})')
    # plt.axhline(y=0.001, color='b', linestyle='--', label='y = 0.001')
    # plt.axhline(y=0.002, color='r', linestyle='--', label='y = 0.002')
    plt.xlabel('Epoch')
    plt.ylabel('Average Inverse Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

