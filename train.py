from torch.utils.tensorboard import SummaryWriter
import os, glob
from utils import dict_send_to
from tqdm import tqdm
import time, datetime
import argparse
import json
import traceback
from hyperparams import hparams as hp
import torch
from torch import nn
from transformer import tacotron
import logging
from utils import infolog, checkpoint
from utils.text import language_vec_to_id
from dataloader import Feeder, FeederEval
from functools import partial
from synthesize import eval_batch, save_eval_results
import sys
import faulthandler, signal
from datetime import timedelta
if hasattr(faulthandler, 'register'):
    faulthandler.register(signal.SIGUSR1)

def main(args):
    logdir = args.log_dir
    model_dir = args.model_dir
    data_dir = args.data_dir

    hp.parse(args.hparams)
    time_id = datetime.datetime.now().strftime('%m%d_%H%M')

    torch.manual_seed(0)
    if args.ddp:
        from torch import distributed as dist
        dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=10))
        local_rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        map_location = lambda _, __:  _.cuda(local_rank)
        print("Local rank: %d, World size: %d" % (local_rank, world_size))
    else:
        local_rank = 0
        world_size = 1
        map_location = {}

    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        infolog.set_logger(os.path.join(logdir, 'outputs_%s.log' % (time_id)))
        writer = SummaryWriter(log_dir=logdir)
        open(os.path.join(logdir, 'hparams.json'), 'w').write(hp.to_json(indent=1))
        open(os.path.join(logdir, 'args.json'), 'w').write(json.dumps(vars(args), indent=1))
    else:
        logging.basicConfig(format="[%(levelname)s %(asctime)s]" + "[%d]" % local_rank + " %(message)s",
                            stream=sys.stdout, level=logging.INFO)
        writer = None
    if not torch.cuda.is_available():
        map_location = lambda _, __:  _.cpu()

    if local_rank == 0:
        values = hp.values()
        logging.info('Hyperparameters:\n' + '\n'.join(['  %s: %s' % (name, values[name]) for name in sorted(values)]))
        logging.info(' '.join(sys.argv))

    if args.eval_steps is not None:
        eval_steps = [int(s) for s in args.eval_steps.split(':')]
    else:
        eval_steps = None

    lang_to_id = json.load(open(os.path.join(data_dir, 'lang_id.json'))) if hp.multi_lingual else None
    spk_to_id = json.load(open(os.path.join(data_dir, 'spk_id.json'))) if hp.multi_speaker else None
    if os.path.exists('filter_keys.json'):
        filter_keys = json.load(open('filter_keys.json'))
    else:
        filter_keys = {}

    if args.training_languages in filter_keys:
        training_languages = filter_keys[args.training_languages]
    else:
        training_languages = args.training_languages.split(':') if args.training_languages else None
    eval_languages = args.eval_languages.split(':') if args.eval_languages else None
    warmup_languages = args.warmup_languages.split(':') if args.warmup_languages else None
    adapt_languages = args.adapt_languages.split(':') if args.adapt_languages else None

    warmup_speakers = args.warmup_speakers.split(':') if args.warmup_speakers else None
    training_speakers = args.training_speakers.split(':') if args.training_speakers else None
    adapt_speakers = args.adapt_speakers.split(':') if args.adapt_speakers else None
    eval_speakers = args.eval_speakers.split(':') if args.eval_speakers else None
    adapt_samples = args.adapt_samples.split(':') if args.adapt_samples else None
    if args.exclude_speakers in filter_keys:
        exclude_speakers = filter_keys[args.exclude_speakers]
    else:
        exclude_speakers = args.exclude_speakers.split(':') if args.exclude_speakers else None

    if args.downsample_languages:
        downsample_languages = args.downsample_languages.split(',')
        downsample_languages = [l.split(':') for l in downsample_languages]
        downsample_languages = dict([(v, float(r)) for v, r in downsample_languages])
    else:
        downsample_languages = {}

    zipfilepath = args.zipfilepath if args.zipfilepath else os.path.join(data_dir, 'mels.zip')
    train_meta = args.train_meta if args.train_meta else os.path.join(data_dir, 'metadata.train.txt')
    eval_meta = args.eval_meta if args.eval_meta else os.path.join(data_dir, 'metadata.eval.txt')
    feeder = Feeder(zipfilepath, train_meta, hparams=hp, spk_to_id=spk_to_id, lang_to_id=lang_to_id,
                    rank=local_rank, world_size=world_size,
                    adapt_lang=adapt_languages, adapt_spk=adapt_speakers,
                    train_lang=training_languages, train_spk=training_speakers, exclude_spk=exclude_speakers,
                    downsample_lang=downsample_languages, adapt_samples=adapt_samples,
                    warmup_lang=warmup_languages, warmup_spk=warmup_speakers)
    if local_rank == 0:
        feeder_eval = FeederEval(zipfilepath, eval_meta, hp, spk_to_id=spk_to_id, lang_to_id=lang_to_id,
                                 eval_lang=eval_languages, eval_spk=eval_speakers, exclude_spk=exclude_speakers,
                                 shuffle=True, keep_order=True, pick_partial=True, single=False)

    logging.info("Using %d GPUs" % torch.cuda.device_count())
    m = tacotron.Tacotron(hp)
    tacotron.initialize_variables(m)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)
    if args.ddp:
        example_param = list(m.parameters())[5]
        logging.info("Model on %s" % str(example_param.device))
        m = nn.parallel.DistributedDataParallel(m, device_ids=[local_rank], output_device=local_rank)
    else:
        m = nn.DataParallel(m)
    m.train()

    optim = torch.optim.Adam(m.parameters(), lr=hp.max_lr, eps=hp.adam_eps)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=partial(tacotron.learning_rate_schedule, hp=hp))

    global_step = None
    if args.restore_from:
        global_step = checkpoint.load_model(args.restore_from, m, optim, sched, map_location)
        logging.info("Restore from" + args.restore_from + ", step %s" % str(global_step))
    ckpt_path = checkpoint.find_ckpt(model_dir)
    if ckpt_path:
        global_step = checkpoint.load_model(ckpt_path, m, optim, sched, map_location)
        logging.info("Restore from previous run at" + model_dir + "from" + ckpt_path + ", step %s" % str(global_step))
    if global_step is None:
        global_step = 0
    if os.path.exists(os.path.join(logdir, 'feeder_%d.pth' % local_rank)):
        feeder.load_state_dict(torch.load(os.path.join(logdir, 'feeder_%d.pth' % local_rank)))

    feeder.global_step = global_step
    feeder.daemon = True
    feeder.start()

    time_window = infolog.ValueWindow(100)
    loss_window = infolog.ValueWindow(100)
    summary_windows = []
    if local_rank == 0:
        state_dict = m.state_dict()
        for var in state_dict:
            logging.info("%s %s" % (var, state_dict[var].shape))

    if hp.multi_lingual:
        id_to_lang = dict([(v, k) for k, v in lang_to_id.items()])
        counts = infolog.LookupWindow('counts', reduction='total')
        aft_losses = infolog.LookupWindow('aft_losses', reduction='avg')
        summary_windows = [counts, aft_losses]

    logging.info("Start training run")
    while True:
        tic = time.time()
        batch = feeder.get_batch()
        batch = dict_send_to(batch, device)

        try:
            outputs = m(**batch)
            losses = tacotron.compute_loss(m, batch['mel_targets'], batch['target_lengths'], outputs, hp)
            optim.zero_grad()
            losses['loss'].backward()
        except Exception as e:
            logging.error("Failed, input shape: %s, target shape: %s" %
                          (str(batch['inputs'].shape), str(batch['mel_targets'].shape)))
            traceback.print_exc()
            if args.ddp: # Not considering race condition, unsure if it is safe
                optim.zero_grad()
                torch.save(feeder.state_dict(), os.path.join(logdir, 'feeder_%d.pth' % local_rank))
                if local_rank == 0:
                    checkpoint.save_model(model_dir, m, optim, sched, global_step)
                else:
                    time.sleep(20)
                sys.exit(1)

        optim.step()
        sched.step()
        global_step += 1
        feeder.global_step = global_step

        if local_rank == 0:
            losses = dict_send_to(losses, torch.device('cpu'), detach=True)
            dur = time.time() - tic
            time_window.append(dur)
            loss_window.append(losses['mse_loss'])
            message = '[Step %d] %.3f sec/step (%.3f), lr=%.06f, loss=%.5f, mse_loss=%.5f (Ave. %.5f)' % (
                global_step, dur, time_window.average, sched.get_last_lr()[-1], losses['loss'],
                losses['mse_loss'], loss_window.average)
            logging.info(message)

            if hp.multi_lingual:
                lvs = batch['input_language_vecs'].detach().cpu().numpy()
                lang_ids = [language_vec_to_id(lv) for lv in lvs]
                langs = [id_to_lang[id] for id in lang_ids]
                counts.update(langs, [1] * len(langs))
                aft_losses.update(langs, losses['aft_losses'])

            if global_step % args.checkpoint_interval == 0:
                checkpoint.save_model(model_dir, m, optim, sched, global_step)
                logging.info("Save checkpoint to " + model_dir)
                os.system("rsync -avu %s %s" % (os.path.join(logdir, '*'), os.path.join(model_dir, 'logs')))

            if global_step % args.summary_interval == 0:
                for key in ['loss', 'mse_loss', 'l2', 'stop_loss', 'aft_loss']:
                    writer.add_scalar('losses/' + key, losses[key], global_step=global_step)
                writer.add_scalar('lr', sched.get_last_lr()[-1], global_step=global_step)
                for window in summary_windows:
                    stats = window.summary()
                    for k, v in stats:
                        writer.add_scalar(k, v, global_step=global_step)
                    window.clear()

            if (eval_steps and global_step in eval_steps) or \
                (eval_steps is None and global_step % args.checkpoint_interval == 0):
                eval_path = os.path.join(logdir, 'eval_%d' % (global_step))
                os.makedirs(eval_path, exist_ok=True)
                m.eval()
                if hasattr(m, 'module'):
                    eval_model = m.module
                else:
                    eval_model = m
                eval_model.decoder.train()
                batches = feeder_eval.fetch_data()
                logging.info('Running %d evals, to %s' % (len(batches), eval_path))
                batches = batches[:hp.max_eval_batches]

                for batch in batches:
                    try:
                        eval_tic = time.time()
                        batch = dict_send_to(batch, device)
                        results = eval_batch(eval_model, batch)
                        save_eval_results(**results, output_dir=eval_path, save_trimmed_wave=False)
                        logging.info('Finished batch in %.2f sec, samples: %s' % (
                            time.time() - eval_tic, batch['names']))
                    except:
                        traceback.print_exc()
                m.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True,
                        help="Directory to save checkpoints and resume (when --restore_from is not specified)")
    parser.add_argument('--log-dir', required=True, help="Directory to save log and tfevents")
    parser.add_argument('--data-dir', required=True, help="Directory with data and metadata")
    parser.add_argument('--zipfilepath', type=str, default=None,
                        help="Zip file of mels, use mels.zip under data-dir when not given")
    parser.add_argument('--train_meta', type=str, default=None,
                        help="Metadata file for training, use metadata.train.txt under data-dir when not given")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--adapt_languages', type=str, default=None,
                        help="Languages for adaptation, separated by colons.")
    parser.add_argument('--adapt_speakers', type=str, default=None,
                        help="Speakers for adaptation under the adapt_languages, separated by colons;"
                             " use all when not given")
    parser.add_argument('--training_languages', type=str, default=None,
                        help="Languages for training, separated by colons; use all when not given")
    parser.add_argument('--training_speakers', type=str, default=None,
                        help="Speakers for training under the training_languages, separated by colons;"
                             " use all when not given")
    parser.add_argument('--eval_languages', type=str, default=None,
                        help="Languages for eval, separated by colons; use all when not given")
    parser.add_argument('--eval_speakers', type=str, default=None,
                        help="Speakers for eval under the eval_languages, separated by colons; use all when not given")
    parser.add_argument('--warmup_languages', type=str, default=None,
                        help="Languages for warmup, separated by colons; use all when not given")
    parser.add_argument('--warmup_speakers', type=str, default=None,
                        help="Speakers for warmup under the warmup_languages, separated by colons;"
                             " use all when not given")
    parser.add_argument('--exclude_speakers', type=str, default=None,
                        help="Speakers to be excluded from training and eval, separated by colons")
    parser.add_argument('--adapt_samples', type=str, default=None,
                        help="Name of samples used for adaptation, separated by colons; use all when not given")
    parser.add_argument('--downsample_languages', type=str, default=None,
                        help="Languages to be downsampled during training, taking the form LANG:RATIO or LANG:N_SAMPLES"
                             ", separated by commas")
    parser.add_argument('--eval_steps', type=str, default=None,
                        help="Steps of checkpoints to run eval on. Run on all steps when not specified")
    parser.add_argument('--checkpoint_interval', type=int, default=10000)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--restore_from', help='Path of checkpoint to restore', default=None)
    parser.add_argument('--hparams', default='', help='Alternative hparams')
    parser.add_argument('--ddp', help='Using DDP', action='store_true')

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)
    main(args)
