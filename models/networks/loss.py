"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.architecture import VGG19
from torchvision.models import inception_v3

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        # 定义用于归一化的均值和标准差
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    def forward(self, x, y):
        loss = 0
        num_channels = x.shape[1]
        for c in range(num_channels):
            x_c = x[:, c:c + 1, :, :]  # 取第 c 个通道
            y_c = y[:, c:c + 1, :, :]
            # 将单通道复制为3通道
            x_c = x_c.repeat(1, 3, 1, 1)
            y_c = y_c.repeat(1, 3, 1, 1)
            # 将图像从 [-1, 1] 归一化到 [0, 1]
            x_c = (x_c + 1) / 2
            y_c = (y_c + 1) / 2

            # 使用 ImageNet 的均值和标准差进行归一化
            x_c = (x_c - self.mean) / self.std
            y_c = (y_c - self.mean) / self.std

            # 通过 VGG 网络提取特征
            x_vgg, y_vgg = self.vgg(x_c), self.vgg(y_c)

            # 计算多层特征的损失
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        # 平均每个通道的损失
        loss /= num_channels
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class SpaceLoss(nn.Module):
    def __init__(self):
        super(SpaceLoss, self).__init__()

    def forward(self, adj_attention_map, adj_matrix):
        """
        计算生成的 adj_attention_map 和给定的 adj_matrix 之间的空间损失。
        使用 L2 损失度量两者之间的差异。
        """
        # adj_attention_map 的形状是 [16, 1, 7, 7]
        # adj_matrix 的形状是 [1, 16, 16]

        batch_size = adj_matrix.size(0)  # bs = 1
        numnodes = adj_matrix.size(1)  # numnodes = 16

        # 扩展 adj_matrix 使其具有与 adj_attention_map 相同的空间维度
        adj_matrix_expanded = adj_matrix.unsqueeze(1)  # [1, 16, 16] -> [1, 1, 16, 16]

        # 将 adj_matrix 的每个节点的关系映射到 (16, 1, 7, 7) 的空间维度上
        adj_matrix_expanded = F.interpolate(
            adj_matrix_expanded, size=(adj_attention_map.size(2), adj_attention_map.size(3)),
            mode='bilinear', align_corners=False
        )  # [1, 1, 7, 7]

        # 计算 L2 损失（均方误差）
        loss = F.mse_loss(adj_attention_map, adj_matrix_expanded)
        return loss


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super(InceptionFeatureExtractor, self).__init__()
        inception = inception_v3(pretrained=True, transform_input=False)
        # 去除 AuxLogits 和 fc 层
        inception.AuxLogits = None
        inception.fc = nn.Identity()
        self.features = inception
        self.features.eval()
        self.features.to(device)
        for param in self.features.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, x):
        if x.size(2) != 299 or x.size(3) != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        # 直接调用 Inception 的 forward 方法
        features = self.features(x)
        # 如果 forward 返回的是元组，则只取第一个（主输出）
        if isinstance(features, tuple):
            features = features[0]
        # 假设此时 features shape 为 (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)
        return features


class StatisticalMatchingLoss(nn.Module):
    def __init__(self, device='cuda', lambda_cov=10.0):
        """
        lambda_cov: 协方差损失的权重
        """
        super(StatisticalMatchingLoss, self).__init__()
        self.feature_extractor = InceptionFeatureExtractor(device=device)
        self.lambda_cov = lambda_cov
        self.device = device

    def compute_mean_cov(self, features):
        """
        计算特征的均值和协方差。features: (B, feature_dim)
        """
        # 计算均值
        mu = torch.mean(features, dim=0)  # shape: (feature_dim,)
        # 中心化数据
        centered = features - mu.unsqueeze(0)
        # 无偏协方差估计，注意：如果 batch size 很小，这里的估计可能会有噪声
        cov = torch.matmul(centered.t(), centered) / (features.size(0) - 1)
        return mu, cov

    def forward(self, fake_images, real_images):
        # 如果输入为单通道，重复 3 个通道
        if fake_images.size(1) == 1:
            fake_images = fake_images.repeat(1, 3, 1, 1)
        if real_images.size(1) == 1:
            real_images = real_images.repeat(1, 3, 1, 1)

        """
        fake_images, real_images: 形状 (B, C, H, W)
        要求图像为浮点型且范围在 [0, 1]（如有需要，可先做归一化）
        """
        # 提取特征
        fake_features = self.feature_extractor(fake_images)
        real_features = self.feature_extractor(real_images)

        # 分别计算均值与协方差
        mu_fake, cov_fake = self.compute_mean_cov(fake_features)
        mu_real, cov_real = self.compute_mean_cov(real_features)

        # 均值差：简单使用 L2 距离（均方误差）
        loss_mu = torch.mean((mu_fake - mu_real) ** 2)
        # 协方差差：采用 Frobenius 范数的平方
        loss_cov = torch.mean((cov_fake - cov_real) ** 2)

        loss = loss_mu + self.lambda_cov * loss_cov
        return loss


class EdgeTextureLoss(nn.Module):
    def __init__(self, num_classes, edge_weight=10.0, texture_weight=10.0):
        super(EdgeTextureLoss, self).__init__()
        self.num_classes = num_classes
        self.learnable_edge_extractor = LearnableEdgeExtractor(num_classes)
        self.learnable_texture_extractor = LearnableTextureExtractor(num_classes)

        # Define Sobel kernels
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x / 8.0)
        self.register_buffer('sobel_y', sobel_y / 8.0)

        # Weights for loss components
        self.edge_weight = edge_weight
        self.texture_weight = texture_weight

    def forward(self, image, real_image, mask_tensor):
        """
        Compute edge and texture losses between the generated and real images.
        Args:
            image: Generated image, [batch_size, 1, H, W].
            real_image: Ground truth image, [batch_size, 1, H, W].
            mask_tensor: One-hot encoded mask, [batch_size, num_classes, H, W].
        Returns:
            Total loss that combines edge and texture differences.
        """
        # Extract edge and texture features
        edge_generated = self.sobel_edge_detection(image, mask_tensor)  # [batch_size, num_classes, H, W]
        edge_real = self.sobel_edge_detection(real_image, mask_tensor)  # [batch_size, num_classes, H, W]
        edge_diff_per_class = torch.abs(edge_generated - edge_real)  # [batch_size, num_classes, H, W]

        texture_generated = self.texture_feature_extraction(image, mask_tensor)  # [batch_size, num_classes, 4]
        texture_real = self.texture_feature_extraction(real_image, mask_tensor)  # [batch_size, num_classes, 4]
        grayscale_diff_per_class = torch.abs(texture_generated - texture_real)  # [batch_size, num_classes, 4]

        # Learnable feature extraction
        learnable_edge_scores = self.learnable_edge_extractor(edge_diff_per_class)
        learnable_texture_scores = self.learnable_texture_extractor(grayscale_diff_per_class)

        # Compute L2 loss for both edge and texture features
        edge_loss = F.mse_loss(learnable_edge_scores, torch.zeros_like(learnable_edge_scores))
        texture_loss = F.mse_loss(learnable_texture_scores, torch.zeros_like(learnable_texture_scores))

        # Total loss as a weighted sum of edge and texture losses
        total_loss = self.edge_weight * edge_loss + self.texture_weight * texture_loss

        return total_loss

    def sobel_edge_detection(self, image, mask_tensor):
        """
        Perform Sobel edge detection for each class.
        Args:
            image: Input image, [batch_size, 1, H, W].
            mask_tensor: One-hot encoded mask, [batch_size, num_classes, H, W].
        Returns:
            Edge maps for each class, [batch_size, num_classes, H, W].
        """
        edge_x = F.conv2d(image, self.sobel_x, padding=1)
        edge_y = F.conv2d(image, self.sobel_y, padding=1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # Avoid NaN issues

        # Apply mask to extract edges for each class
        class_edges = edge * mask_tensor  # Broadcasting over classes
        return class_edges

    def texture_feature_extraction(self, image, mask_tensor):
        """
        Extract texture features (mean, std, max, min) for each class.
        Args:
            image: Input image, [batch_size, 1, H, W].
            mask_tensor: One-hot encoded mask, [batch_size, num_classes, H, W].
        Returns:
            Texture features for each class, [batch_size, num_classes, 4].
        """
        batch_size, numnodes, num_classes, height, width = mask_tensor.shape
        mask_tensor = mask_tensor.reshape(-1, num_classes, height, width)
        grayscale_features = torch.zeros(batch_size * numnodes, num_classes, 4, device=image.device)

        for i in range(num_classes):
            class_mask = mask_tensor[:, i:i + 1, :, :]  # [batch_size, 1, H, W]
            masked_image = image * class_mask

            valid_pixels = class_mask.sum(dim=[2, 3], keepdim=True) + 1e-8
            mean_gray = masked_image.sum(dim=[2, 3], keepdim=True) / valid_pixels
            std_gray = torch.sqrt(((masked_image - mean_gray) ** 2).sum(dim=[2, 3], keepdim=True) / valid_pixels + 1e-6)
            max_gray = masked_image.amax(dim=[2, 3], keepdim=True)
            min_gray = masked_image.amin(dim=[2, 3], keepdim=True)
            grayscale_features[:, i, 0] = mean_gray.squeeze()
            grayscale_features[:, i, 1] = std_gray.squeeze()
            grayscale_features[:, i, 2] = max_gray.squeeze()
            grayscale_features[:, i, 3] = min_gray.squeeze()

        return grayscale_features


class LearnableEdgeExtractor(nn.Module):
    def __init__(self, num_classes):
        super(LearnableEdgeExtractor, self).__init__()
        self.conv1 = nn.Conv2d(num_classes, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, edge_diff_per_class):
        edge_diff_per_class = edge_diff_per_class.reshape(-1, edge_diff_per_class.shape[2], edge_diff_per_class.shape[3], edge_diff_per_class.shape[4])
        """
        Learnable Edge Extractor.
        Input: [batch_size, num_classes, height, width]
        Output: [batch_size, num_classes]
        """
        x = F.relu(self.bn1(self.conv1(edge_diff_per_class)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)  # [batch_size, num_classes]
        x = self.fc(x)
        return torch.sigmoid(x)


class LearnableTextureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(LearnableTextureExtractor, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.bn1 = nn.BatchNorm1d(16)  # 修改为固定的特征维度
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)  # 修改为固定的特征维度
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(32, 64)
        self.bn3 = nn.BatchNorm1d(64)  # 修改为固定的特征维度
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, grayscale_diff_per_class):
        """
        Learnable Texture Extractor.
        Input: [batch_size, num_classes, 4]
        Output: [batch_size, num_classes]
        """
        batch_size, num_classes, _ = grayscale_diff_per_class.shape
        x = grayscale_diff_per_class.view(batch_size * num_classes, -1)  # Flatten classes
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return torch.sigmoid(x).view(batch_size, num_classes)
