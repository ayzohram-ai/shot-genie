# improved_config.py - 针对过拟合问题的改进配置

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import clip

class ImprovedConfig:
    """改进的训练配置，解决过拟合问题"""
    
    # =========================== 核心改进 ===========================
    
    # 1. 更强的正则化
    DROPOUT = 0.3                    # 增加Dropout (原来0.1)
    WEIGHT_DECAY = 0.05              # 增加权重衰减 (原来0.01)
    
    # 2. 更保守的学习率
    CLIP_LR = 5e-7                   # 降低CLIP学习率 (原来1e-6)
    REGRESSION_LR = 5e-5             # 降低回归头学习率 (原来1e-4)
    
    # 3. 早停机制
    PATIENCE = 3                     # 减少耐心值 (原来5)
    MIN_DELTA = 1e-5                 # 更小的改善阈值
    
    # 4. 数据增强
    USE_DATA_AUGMENTATION = True     # 启用数据增强
    
    # 5. 批次大小调整
    BATCH_SIZE = 16                  # 减小批次大小 (原来32)
    
    # 6. 训练轮数
    NUM_EPOCHS = 15                  # 减少训练轮数 (原来20)

class ImprovedAestheticModel(nn.Module):
    """改进的模型架构，增强泛化能力"""
    
    def __init__(self, clip_model_name="ViT-B/32", hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # 加载CLIP，部分冻结
        self.clip_model, _ = clip.load(clip_model_name, device="cpu")
        
        # 冻结CLIP的前几层
        self._freeze_clip_layers()
        
        if "ViT-B" in clip_model_name:
            clip_dim = 512
        elif "ViT-L" in clip_model_name:
            clip_dim = 768
        else:
            clip_dim = 512
        
        # 改进的回归头 - 增加BatchNorm和更多正则化
        self.regression_head = nn.Sequential(
            # 第一层
            nn.Linear(clip_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第二层
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),  # 添加BatchNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 第三层（新增）
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # 输出层
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _freeze_clip_layers(self):
        """冻结CLIP的前几层"""
        # 冻结前6层的transformer blocks
        for i, layer in enumerate(self.clip_model.visual.transformer.resblocks):
            if i < 6:  # 冻结前一半的层
                for param in layer.parameters():
                    param.requires_grad = False
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        for m in self.regression_head.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        with torch.cuda.amp.autocast():
            image_features = self.clip_model.encode_image(images)
        
        scores = self.regression_head(image_features.float())
        return scores.squeeze(-1)

class ImprovedTrainer:
    """改进的训练器，包含更好的正则化策略"""
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数 - 添加L1正则化
        self.criterion = nn.MSELoss()
        self.l1_lambda = 0.001  # L1正则化系数
        
        # 改进的优化器配置
        self._setup_optimizer()
        
        # 学习率调度器 - 使用ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2, 
            min_lr=1e-8, verbose=True
        )
        
        # 早停
        self.early_stopping = EarlyStopping(
            patience=ImprovedConfig.PATIENCE,
            min_delta=ImprovedConfig.MIN_DELTA
        )
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler()
        
        # 记录
        self.train_losses = []
        self.val_losses = []
        self.val_correlations = []
        self.lr_history = []
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 分组参数
        clip_params = []
        regression_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # 只优化需要梯度的参数
                if 'regression_head' in name:
                    regression_params.append(param)
                else:
                    clip_params.append(param)
        
        print(f"CLIP可训练参数: {sum(p.numel() for p in clip_params)}")
        print(f"回归头参数: {sum(p.numel() for p in regression_params)}")
        
        # AdamW优化器
        self.optimizer = optim.AdamW([
            {'params': clip_params, 'lr': ImprovedConfig.CLIP_LR, 
             'weight_decay': ImprovedConfig.WEIGHT_DECAY},
            {'params': regression_params, 'lr': ImprovedConfig.REGRESSION_LR, 
             'weight_decay': ImprovedConfig.WEIGHT_DECAY}
        ])
    
    def _compute_l1_loss(self):
        """计算L1正则化损失"""
        l1_loss = 0
        for param in self.model.regression_head.parameters():
            l1_loss += torch.sum(torch.abs(param))
        return l1_loss * self.l1_lambda
    
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, scores) in enumerate(self.train_loader):
            images = images.to(self.device)
            scores = scores.to(self.device)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                predictions = self.model(images)
                mse_loss = self.criterion(predictions, scores)
                l1_loss = self._compute_l1_loss()
                loss = mse_loss + l1_loss
            
            # 反向传播
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        return total_loss / num_batches
    
    def validate(self):
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, scores in self.val_loader:
                images = images.to(self.device)
                scores = scores.to(self.device)
                
                with torch.cuda.amp.autocast():
                    predictions = self.model(images)
                    loss = self.criterion(predictions, scores)
                
                total_loss += loss.item()
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(scores.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        correlation = np.corrcoef(all_preds, all_targets)[0, 1] if len(all_preds) > 1 else 0
        
        return avg_loss, correlation
    
    def train(self, num_epochs):
        """主训练循环"""
        print(f"🏋️ 开始改进训练，共{num_epochs}个epoch")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 40)
            
            # 训练
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # 验证
            val_loss, correlation = self.validate()
            self.val_losses.append(val_loss)
            self.val_correlations.append(correlation)
            
            # 记录学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            self.lr_history.append(current_lr)
            
            # 学习率调度
            self.scheduler.step(val_loss)
            
            print(f"训练损失: {train_loss:.6f}")
            print(f"验证损失: {val_loss:.6f}")
            print(f"验证相关性: {correlation:.6f}")
            print(f"学习率: {current_lr:.2e}")
            
            # 早停检查
            if self.early_stopping(val_loss):
                print(f"💡 早停触发，在epoch {epoch+1}")
                break
            
            # 保存最佳模型
            if self.early_stopping.is_best:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'correlation': correlation
                }, r'D:\clip+ava\improved_best_model.pth')
                print("✅ 保存改进的最佳模型!")
        
        print(f"🎉 训练完成! 最佳验证损失: {self.early_stopping.best_loss:.6f}")
        
        # 绘制改进的训练曲线
        self.plot_training_curves()

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience=3, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.is_best = False
    
    def __call__(self, val_loss):
        self.is_best = False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.is_best = True
        else:
            self.counter += 1
        
        return self.counter >= self.patience

def create_improved_data_loaders():
    """创建改进的数据加载器，包含数据增强"""
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import pandas as pd
    
    # 数据增强的变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
        transforms.RandomRotation(degrees=5),        # 小角度旋转
        transforms.ColorJitter(                      # 颜色抖动
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        ),
        transforms.RandomAdjustSharpness(1.2, p=0.3), # 随机锐化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 验证集不使用数据增强
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 这里需要您的数据集类
    # train_dataset = YourDataset(train_df, transform=train_transform)
    # val_dataset = YourDataset(val_df, transform=val_transform)
    
    print("💡 数据增强配置:")
    print("- 随机水平翻转")
    print("- 小角度旋转 (±5°)")
    print("- 颜色抖动")
    print("- 随机锐化")
    
    return None, None  # 占位符

# 使用示例
def retrain_with_improvements():
    """使用改进配置重新训练"""
    print("🔄 使用改进配置重新训练模型")
    print("主要改进:")
    print("✓ 增加Dropout到0.3")
    print("✓ 增加权重衰减到0.05")  
    print("✓ 降低学习率")
    print("✓ 部分冻结CLIP层")
    print("✓ 添加BatchNorm")
    print("✓ 启用梯度裁剪")
    print("✓ L1正则化")
    print("✓ 数据增强")
    print("✓ 改进的早停机制")
    
    # 创建改进的模型
    model = ImprovedAestheticModel(dropout=0.3)
    
    # 这里需要您的数据加载器
    # train_loader, val_loader = create_improved_data_loaders()
    # trainer = ImprovedTrainer(model, train_loader, val_loader)
    # trainer.train(ImprovedConfig.NUM_EPOCHS)
    
    print("\n运行这个改进版本，您应该看到:")
    print("🎯 验证损失不再持续上升")
    print("📈 验证相关性保持稳定或改善")
    print("⚡ 更快的收敛和更好的泛化")

if __name__ == "__main__":
    retrain_with_improvements()