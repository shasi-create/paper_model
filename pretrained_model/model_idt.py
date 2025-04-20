#from tensorboard import summary

from h5Dataset import H5Dataset
from torch.utils.data import DataLoader
import torch
torch.backends.cudnn = False
import torch.nn as nn
import torch.optim as optim
import os
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import argparse
import logging
from FC4_AddSkip import FC4AddSkip
import pdb


logging.basicConfig(level=logging.INFO,  # 设置日志级别为 INFO
                    format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
                    handlers=[
                        logging.StreamHandler(),  # 输出到控制台
                        logging.FileHandler('retry.txt')  # 输出到文件
                    ])


def denormalize(tensor, mean, std, eps=1e-8):
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean, device=tensor.device)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std, device=tensor.device)
    mean_clone = mean.clone().detach().to(tensor.device)  # 使用 clone().detach()
    std_clone = std.clone().detach().to(tensor.device)  # 使用 clone().detach()
    return tensor * (std_clone + eps) + mean_clone


def train_and_validate_model(model, train_dataloader, valid_dataloader,
                             optimizer, criterion, device, num_epochs,
                             output_mean, output_std):
    global best_mse
    # 用于记录每个 epoch 的平均逆标准化损失
    avg_inverse_train_losses = []
    avg_inverse_valid_losses = []

    avg_standardized_train_losses = []
    avg_standardized_valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_total_samples = 0
        train_total_inverse_loss = 0
        train_total_standardized_loss = 0

        for batch_index, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            batch_y = batch_y.unsqueeze(-1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            # 计算逆标准化后的损失
            outputs_inverse = denormalize(outputs, output_mean, output_std)
            batch_y_inverse = denormalize(batch_y, output_mean, output_std)
            batch_inverse_loss = criterion(outputs_inverse, batch_y_inverse)
            train_total_inverse_loss += batch_inverse_loss.sum()

            train_total_samples += batch_y.size(0)
            if loss.mean() > 10:
                print(f"batch id: ", batch_index + 1, f"loss:", loss)
                pdb.set_trace()

            if (batch_index + 1) % 50 == 0:
                standardized_loss_train = loss.mean().item()
                inverse_standardized_loss = batch_inverse_loss.mean().item()

                print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}], "
                      f"Standardized Loss: {standardized_loss_train:.4f}, "
                      f"Inverse Standardized Loss: {inverse_standardized_loss:.4f}")
            train_total_standardized_loss += loss.sum()

            loss.mean().backward()  # 反向传播是基于标准化的平均损失
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                      
            optimizer.step()  # 更新参数
                         
        # 计算并存储每个epoch的平均逆标准化损失
        avg_inverse_train_loss_per_epoch = train_total_inverse_loss / train_total_samples
        avg_inverse_train_losses.append(avg_inverse_train_loss_per_epoch)  # 存储这个 epoch 的平均损失

        avg_standardized_train_loss_per_epoch = train_total_standardized_loss / train_total_samples
        avg_standardized_train_losses.append(avg_standardized_train_loss_per_epoch)  # 存储标准化训练损失
        if avg_standardized_train_loss_per_epoch > 10:
            print("Average training loss: ", avg_standardized_train_loss_per_epoch)
            # pdb.set_trace()
        torch.cuda.empty_cache()

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

                outputs_inverse = denormalize(outputs, output_mean, output_std)
                batch_y_inverse = denormalize(batch_y, output_mean, output_std)
                valid_inverse_loss = criterion(outputs_inverse, batch_y_inverse)

                if torch.isnan(valid_inverse_loss).any():
                    print(f"NaN detected in epoch {epoch}, batch {batch_index}")

                valid_total_standardized_loss += valid_standardized_loss.sum()
                valid_total_inverse_loss += valid_inverse_loss.sum()
                valid_total_samples += batch_y.size(0)

                if (batch_index + 1) % 5 == 0:
                    standardized_loss_valid = valid_standardized_loss.mean().item()
                    inverse_standardized_loss_valid = valid_inverse_loss.mean().item()

                    print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_index + 1}], "
                          f"Standardized Loss: {standardized_loss_valid:.4f}, "
                          f"Inverse Standardized Loss: {inverse_standardized_loss_valid:.4f}")

        # 计算并存储每个 epoch 的平均逆标准化损失
        avg_inverse_valid_loss_per_epoch = valid_total_inverse_loss / valid_total_samples
        avg_inverse_valid_losses.append(avg_inverse_valid_loss_per_epoch)  # 存储这个 epoch 的平均损失
        
        avg_standardized_valid_loss_per_epoch = valid_total_standardized_loss / valid_total_samples
        avg_standardized_valid_losses.append(avg_standardized_valid_loss_per_epoch)

        # Save checkpoint.
        if avg_inverse_valid_loss_per_epoch < best_mse:
            print('Saving..')
            logging.info('Saving..')
            state = {
                'net': model.state_dict(),               # 保存模型的状态字典
                'optimizer': optimizer.state_dict(),     # 保存优化器的状态字典
                'acc': avg_inverse_valid_loss_per_epoch, # 保存当前最佳验证损失
                'epoch': epoch                           # 保存当前的训练轮数
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt_0113.pth')
            best_mse = avg_inverse_valid_loss_per_epoch

        # 清理 GPU 缓存
        torch.cuda.empty_cache()

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train--'
            f' Inverse Loss: {avg_inverse_train_loss_per_epoch:.4f}, '
            f'Standardized Loss: {avg_standardized_train_loss_per_epoch:.4f}, '
            f'Valid--'
            f' Inverse Loss: {avg_inverse_valid_loss_per_epoch:.4f}, '
            f' Standardized Loss: {avg_standardized_valid_loss_per_epoch:.4f}')

        logging.info(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train--'
            f' Inverse Loss: {avg_inverse_train_loss_per_epoch:.4f}, '
            f'Standardized Loss: {avg_standardized_train_loss_per_epoch:.4f}, '
            f'Valid--'
            f' Inverse Loss: {avg_inverse_valid_loss_per_epoch:.4f}, '
            f' Standardized Loss: {avg_standardized_valid_loss_per_epoch:.4f}')

    print("Training done.")
    logging.info('Training done.')

    return avg_inverse_train_losses, avg_inverse_valid_losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch 25856logIDT_4FC_add_lion Training')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('Using device: %s', device)
    best_mse = float('inf')

    logging.info('==> Preparing data..')
    data = '04data_log.h5'
    mean_std_file = 'param_statistics.json'
    mean_log_y = 2.832582
    std_log_y = 0.456879
    batch_size_v = 64
    train_dataset = H5Dataset(data,
                              mean_std_file_path=mean_std_file,
                              mean_output=mean_log_y, std_output=std_log_y,
                              dataset_type='train')
    valid_dataset = H5Dataset(data,
                              mean_std_file_path=mean_std_file,
                              mean_output=mean_log_y, std_output=std_log_y,
                              dataset_type='valid')
    test_dataset = H5Dataset(data,
                             mean_std_file_path=mean_std_file,
                             mean_output=mean_log_y, std_output=std_log_y,
                             dataset_type='test')
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Dataset size: {len(valid_dataset)}")
    print(f"Dataset size: {len(test_dataset)}")
    # 将生成器包装为 DataLoader
    nw = 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size_v, num_workers=nw,
                              pin_memory=False, prefetch_factor=1, shuffle=True)

    valid_loader = DataLoader(valid_dataset, batch_size=batch_size_v, num_workers=nw,
                              pin_memory=False, prefetch_factor=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size_v, num_workers=nw,
                             pin_memory=False, prefetch_factor=1, shuffle=False)
    logging.info('Using {} dataloader workers every process'.format(nw))

    logging.info('==> Building model..')
    pre_model = FC4AddSkip().to(device)
    data_iter = iter(train_loader)
    inputs, _ = next(data_iter)
    print(inputs.shape)
    #summary(pre_model, (inputs.shape[1],))

    criterion_1 = nn.MSELoss(reduction='none')  # 使用均方误差作为损失函数（回归任务）
    #optimizer_1 = Lion(pre_model.parameters(), lr=1e-4, weight_decay=1e-2)
    optimizer_1 = optim.AdamW(pre_model.parameters(), lr=1e-4, weight_decay=1e-5)
    epochs = 700

    (inverse_train_losses, inverse_valid_losses) = train_and_validate_model(
        model=pre_model, train_dataloader=train_loader,
        valid_dataloader=valid_loader, optimizer=optimizer_1,
        criterion=criterion_1, device=device, num_epochs=epochs,
        output_mean=mean_log_y, output_std=std_log_y
    )
