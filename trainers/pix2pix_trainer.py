import torch.nn as nn
from models.pix2pix_model import Pix2PixModel
from torch.cuda.amp import autocast, GradScaler

class Pix2PixTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = Pix2PixModel(opt)
        if len(opt.gpu_ids) > 0:
            self.pix2pix_model = nn.DataParallel(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)
            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        self.use_fp16 = opt.use_fp16  # Option for using mixed precision (FP16)
        self.scaler = GradScaler() if self.use_fp16 else None

    def run_generator_one_step(self, data, adj_matrix=None):
        """
        Run one step of the generator.
        It takes 'data' and an optional 'adj_matrix' argument.
        """
        self.optimizer_G.zero_grad()

        # Use autocast for mixed precision training if use_fp16 is True
        if self.use_fp16:
            with autocast():
                # Call the model, passing adj_matrix if provided
                g_losses, generated = self.pix2pix_model(data, mode='generator', adj_matrix=adj_matrix)
                g_loss = sum(g_losses.values()).mean()

            # Scale the loss and backpropagate
            self.scaler.scale(g_loss).backward()

            # Step the optimizer with the scaled gradients
            self.scaler.step(self.optimizer_G)

            # Update the scaler
            self.scaler.update()
        else:
            # Without autocast, using FP32 precision
            g_losses, generated = self.pix2pix_model(data, mode='generator', adj_matrix=adj_matrix)
            g_loss = sum(g_losses.values()).mean()
            g_loss.backward()
            self.optimizer_G.step()

        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data, adj_matrix=None):
        self.optimizer_D.zero_grad()

        # Use autocast for mixed precision training if use_fp16 is True
        if self.use_fp16:
            with autocast():
                # Call the model, passing fake_images if provided
                d_losses = self.pix2pix_model(data, mode='discriminator', adj_matrix=adj_matrix)
                d_loss = sum(d_losses.values()).mean()

            # Scale the loss and backpropagate
            self.scaler.scale(d_loss).backward()

            # Step the optimizer with the scaled gradients
            self.scaler.step(self.optimizer_D)

            # Update the scaler
            self.scaler.update()
        else:
            # Without autocast, using FP32 precision
            d_losses = self.pix2pix_model(data, mode='discriminator')
            d_loss = sum(d_losses.values()).mean()
            d_loss.backward()
            self.optimizer_D.step()

        self.d_losses = d_losses

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
