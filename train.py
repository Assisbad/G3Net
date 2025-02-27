import sys
import torch
from collections import OrderedDict
from tqdm import tqdm
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from trainers.pix2pix_trainer import Pix2PixTrainer

opt = TrainOptions().parse()
print(' '.join(sys.argv))

dataloader = data.create_dataloader(opt)

trainer = Pix2PixTrainer(opt)

iter_counter = IterationCounter(opt, len(dataloader))
visualizer = Visualizer(opt)

best_fid = float('inf')


first_batch = next(iter(dataloader))
adj_matrix = first_batch.get('adj').cuda()


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)


    for i, data_i in enumerate(tqdm(dataloader, desc=f'Epoch {epoch}', ncols=100), start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()
        input_semantics, real_images = data_i['label'].cuda(), data_i['image'].cuda()

        label_map = input_semantics.long()
        bs, numnodes, c, h, w = label_map.size()
        if opt.label_nc == 1:

            input_semantics = label_map.float()
        else:
            nc = opt.label_nc + 1 if opt.contain_dontcare_label else opt.label_nc
            input_label = torch.zeros(bs, numnodes, opt.label_nc, h, w).cuda()
            input_semantics = input_label.scatter_(2, label_map, 1.0)


        fake_images, _, _ = trainer.pix2pix_model.module.generate_fake(input_semantics, real_images, compute_kld_loss=False, adj_matrix=adj_matrix)

        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i, adj_matrix)
        trainer.run_discriminator_one_step(data_i, adj_matrix)


        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([
                ('input_label', data_i['label'].view(-1, data_i['label'].size(2), data_i['label'].size(3), data_i['label'].size(4))),
                ('synthesized_image', trainer.get_latest_generated()),
                ('real_image', data_i['image'].view(-1, data_i['image'].size(2), data_i['image'].size(3), data_i['image'].size(4)))
            ])

            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print(f'Saving the latest model (epoch {epoch}, total_steps {iter_counter.total_steps_so_far})')
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print(f'Saving the model at the end of epoch {epoch}, iters {iter_counter.total_steps_so_far}')
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')