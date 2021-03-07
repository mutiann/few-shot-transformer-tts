from preprocess import MultiDatasets
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os, glob
from tqdm import tqdm
import time, datetime
from utils import find_ckpt, plot_mel
import argparse
from corpora import T1, T2, T3
import torch
import json
import traceback

def adjust_learning_rate(optimizer, step_num):
    if hp.lr_decay_type == 'noam':
        lr = hp.lr_max * hp.warmup_steps ** 0.5 * min(step_num * hp.warmup_steps ** -1.5, step_num ** -0.5)
    else:
        if step_num < hp.warmup_steps:
            lr = hp.lr_max
        else:
            lr = (0.01 ** ((step_num - hp.warmup_steps) / hp.lr_decay_steps)) * hp.lr_max
        lr = max(lr, hp.lr_min)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def linear_anneal(step, starts, ends, min_val, max_val):
    if step <= starts:
        return min_val
    if step >= ends:
        return max_val
    else:
        return min_val + ((step - starts) / (ends - starts)) * (max_val - min_val)

def synthesize(data_path, model_path, out_path, samples):
    import sys
    os.makedirs(out_path, exist_ok=True)
    for name, text, spk, lang in samples:
        cmd = sys.executable + ' synthesis.py --data_path=%s --restore_path=%s --spk=%s --lang=%s --text="%s" --out_path=%s' % (
            data_path, model_path, spk, lang, text, os.path.join(out_path, name)
        )
        print(cmd)
        os.system(cmd)

def main(args):
    subsets = args.subsets
    if subsets == 'T1':
        subsets = T1
    elif subsets == 'T2':
        subsets = T1 + ':' + T2
    elif subsets == 'T3':
        subsets = T1 + ':' + T2 + ':' + T3
    subsets = subsets.split(':')
    print("Training on datasets:", subsets)
    logdir = args.log_dir
    model_path = args.model_path
    hp.parse(args.hparams)
    open(os.path.join(logdir, 'hparams.json'), 'w').write(hp.to_json(indent=1))
    open(os.path.join(logdir, 'args.json'), 'w').write(json.dumps(vars(args), indent=1))
    torch.manual_seed(0)
    if args.ddp:
    # Not supported
        raise NotImplementedError()
    else:
        rank = local_rank = 0
        log_flag = True
        map_location = {}

    values = hp.values()
    if log_flag:
        print('Hyperparameters:\n' + '\n'.join(['  %s: %s' % (name, values[name]) for name in sorted(values)]))

    try:
        dataset = MultiDatasets(args.data_path, subsets, seed=local_rank)
    except:
        import traceback as tb
        tb.print_exc()
    global_step = 0
    m = Model().cuda()
    if args.ddp:
        example_param = list(m.parameters())[5]
        print("[%d] Model on" % local_rank, example_param.device)
        m = nn.parallel.DistributedDataParallel(
            m, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    else:
        m = nn.DataParallel(m)
    if log_flag:
        print("Using", t.cuda.device_count(), 'GPUs')

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr_max)

    if args.restore_from:
        params = t.load(args.restore_from, map_location=map_location)
        m.load_state_dict(params['model'])
        optimizer.load_state_dict(params['optimizer'])
        global_step = list(params['optimizer']['state'].values())[0]['step']
        print("[%d] Restore from" % local_rank, args.restore_from)

    params = find_ckpt(model_path, map_location=map_location)
    if params:
        m.load_state_dict(params['model'])
        optimizer.load_state_dict(params['optimizer'])
        global_step = params['step']
        print("[%d] Restore from previous run at" % local_rank, model_path, "Step=%d" % global_step)

    if log_flag:
        writer = SummaryWriter(log_dir=logdir)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
    if args.ddp:
        print("Initialized at process %d" % local_rank)

    dataset.global_step = global_step
    dataset.start()

    eval_samples = []
    for lang_ds in dataset.datasets:
        for sub_ds in lang_ds.datasets:
            eval_samples.append(sub_ds.meta[0])
            eval_samples.append(sub_ds.meta[-1])

    while True:
        tic = time.time()
        data = dataset.get_batch()
        global_step += 1
        dataset.global_step = global_step
        lr = adjust_learning_rate(optimizer, global_step)

        character, mel, mel_input, lang_ids, spk_ids, pos_text, pos_mel, _ = data
        character = character.cuda()
        mel = mel.cuda()
        mel_input = mel_input.cuda()
        lang_ids = lang_ids.cuda()
        spk_ids = spk_ids.cuda()
        pos_text = pos_text.cuda()
        pos_mel = pos_mel.cuda()
        mask = t.abs(pos_mel.ne(0).type(t.float) - 1)
        stop_tokens = t.cat([mask[:, 1:], t.ones_like(mask[:, -1:])], dim=-1)

        mel_pred, postnet_pred, attn_probs, stop_preds, attns_enc, attns_dec = \
            m.forward(character, mel_input, pos_text, pos_mel, lang_ids, spk_ids)

        mask = 1 - mask
        mel_loss = nn.MSELoss(reduction='none')(mel_pred, mel).mean(-1)
        mel_loss = (mel_loss * mask).sum() / mask.sum()
        post_mel_loss = nn.MSELoss(reduction='none')(postnet_pred, mel).mean(-1)
        post_mel_loss = (post_mel_loss * mask).sum() / mask.sum()

        stop_loss = nn.BCEWithLogitsLoss(reduction='none')(stop_preds, stop_tokens.unsqueeze(-1)).squeeze(-1)
        stop_loss = (stop_loss * mask).sum() / mask.sum()
        st_weight = linear_anneal(global_step, hp.stop_token_weight_anneal_start, hp.stop_token_weight_anneal_step,
                                  min_val=0.1, max_val=hp.stop_token_weight)
        loss = mel_loss + post_mel_loss + stop_loss * st_weight

        optimizer.zero_grad()
        # Calculate gradients
        try:
            loss.backward()
        except Exception as e:
            print("Failed, mel shape:", mel_input.shape)
            raise e
        if global_step > 1000:
            nn.utils.clip_grad_norm_(m.parameters(), 0.1)

        # Update weights
        optimizer.step()

        if global_step % hp.image_step == 0 and log_flag:
            for i, prob in enumerate(attn_probs):
                shift = max(1, prob.shape[0] // 4)
                k = 0
                for j in range(0, prob.shape[0], shift):
                    x = vutils.make_grid(prob[j].T * 255)
                    writer.add_image('Attention_%d_%d' % (i, k), x, global_step)
                    k += 1

            for i, prob in enumerate(attns_enc):
                shift = max(1, prob.shape[0] // 4)
                k = 0
                for j in range(0, prob.shape[0], shift):
                    x = vutils.make_grid(prob[j].T * 255)
                    writer.add_image('Attention_enc_%d_%d'% (i, k), x, global_step)
                    k += 1

            for i, prob in enumerate(attns_dec):
                shift = max(1, prob.shape[0] // 4)
                k = 0
                for j in range(0, prob.shape[0], shift):
                    x = vutils.make_grid(prob[j] * 255)
                    writer.add_image('Attention_dec_%d_%d' % (i, k), x, global_step)
                    k += 1

        if global_step % 100 == 0 and log_flag:
            grad_l2norms = [p.grad.data.norm(2) for p in m.parameters() if p.grad is not None]
            grad_infnorms = [p.grad.data.abs().max() for p in m.parameters() if p.grad is not None]
            writer.add_scalar('mel_loss', mel_loss, global_step)
            writer.add_scalar('post_mel_loss', post_mel_loss, global_step)
            writer.add_scalar('stop_loss', stop_loss, global_step)
            writer.add_scalar('lr', lr, global_step)
            writer.add_scalar('st_weight', st_weight, global_step)
            writer.add_scalar('grad_norm/mean_l2', sum(grad_l2norms) / len(grad_l2norms), global_step)
            writer.add_scalar('grad_norm/max_l2', max(grad_infnorms), global_step)
            writer.add_scalar('grad_norm/mean_inf', sum(grad_infnorms) / len(grad_infnorms), global_step)
            writer.add_scalar('grad_norm/max_inf', max(grad_infnorms), global_step)

            writer.add_scalars('alphas',{
                    'encoder_alpha':m.module.encoder.alpha.data,
                    'decoder_alpha':m.module.decoder.alpha.data,
                }, global_step)

        if log_flag:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            print("[%s][Step %d] Loss=%.4f Post-loss=%.4f Stop-loss=%.4f (%.3f s) Alpha=(%.4f, %.4f)" % (
                timestamp, global_step, loss.item(), post_mel_loss.item(), stop_loss.item(), time.time() - tic,
                m.module.encoder.alpha.data, m.module.decoder.alpha.data
            ))

            if global_step % hp.save_step == 0:
                print("Save to", os.path.join(model_path, 'checkpoint_transformer_%d.pth.tar' % global_step))
                t.save({'model':m.state_dict(), 'optimizer':optimizer.state_dict()},
                       os.path.join(model_path, 'checkpoint_transformer_%d.pth.tar' % global_step))
                writer.flush()
                if global_step % (hp.save_step * 2) == 0 and global_step > 60000:
                    synthesize(args.data_path,
                               os.path.join(model_path, 'checkpoint_transformer_%d.pth.tar' % global_step),
                               os.path.join(args.log_dir, 'eval_%d' % global_step), samples=eval_samples)

        elif global_step % 100 == 0:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
            print("[%s][%d][Step %d] Loss=%.4f Post-loss=%.4f Stop-loss=%.4f (%.3f s) Alpha=(%.4f, %.4f)" % (
                timestamp, local_rank, global_step, loss.item(), post_mel_loss.item(), stop_loss.item(), time.time() - tic,
                m.module.encoder.alpha.data, m.module.decoder.alpha.data
            ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-dir', dest='model_path',
        help="Directory to save checkpoints and resume (when --restore_from is not specified)",
        default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument(
        '--log-dir', '--logDir',
        help="Directory to save log and tfevents", default=".")
    parser.add_argument(
        '--data_path', required=True)
    parser.add_argument(
        '--subsets', help="Subsets of training data", required=True)
    parser.add_argument(
        '--restore_from', help='Path of checkpoint to restore', default=None)
    parser.add_argument(
        '--hparams', default='',
        help='Alternative hparams')
    parser.add_argument(
        '--ddp', help='Using DDP',
        default=False)

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)
    main(args)
