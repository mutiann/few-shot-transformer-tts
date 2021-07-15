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
from concurrent.futures import ProcessPoolExecutor
from utils.transcribe import transcribe, transcribe_available
from transformer import tacotron
import logging
from utils import infolog, checkpoint
from utils.text import language_vec_to_id
from dataloader import FeederEval
from functools import partial
from synthesize import eval_batch, save_eval_results
import numpy as np
import sys
import faulthandler, signal
if hasattr(faulthandler, 'register'):
    faulthandler.register(signal.SIGUSR1)


def run_transcription(eval_path, names, existent_samples, meta_index, cer_window, step):
    if os.path.exists(os.path.join(eval_path, 'transcriptions.jsonl')):
        lines = open(os.path.join(eval_path, 'transcriptions.jsonl'), encoding='utf-8').read().splitlines()
        lines = [json.loads(l) for l in lines]
        found_names = [t['name'] for t in lines if t['DisplayText']]
        transcribe_names = set(names + [n for n in existent_samples if n not in found_names])
        logging.info("Exist transcriptions skipped: " + str(set(found_names).difference(transcribe_names)))
        prev_trans = [t for t in lines if t['name'] not in transcribe_names and t['DisplayText']]
    else:
        transcribe_names = names + existent_samples
        prev_trans = []
    trans = []
    for n in transcribe_names:
        if n + '.npy' in meta_index:
            trans.append(transcribe(wav_path=os.path.join(eval_path, n + '_trim.wav'),
                                    meta=meta_index[n + '.npy'], id_to_lang=lambda x: x.replace('_', '-')))
    trans += prev_trans
    trans.sort(key=lambda x: x['name'])

    with open(os.path.join(eval_path, 'transcriptions.jsonl'), 'w', encoding='utf-8') as fw:
        for t in trans:
            fw.write(json.dumps(t, ensure_ascii=False) + '\n')
    logging.info('[Step %d] Raw CER=%.3f' % (step, np.mean([t['cer'] for t in trans]).item()))

    keys = []
    values = []
    for t in trans:
        if 'fail' not in t:
            keys.append(t['locale'])
            values.append(t['cer'])
        else:
            logging.warn("Failed sample: " + t['name'])
    cer_window.update(keys, values)


def main(args):
    logdir = args.log_dir
    model_dir = args.model_dir
    data_dir = args.data_dir

    os.makedirs(logdir, exist_ok=True)
    hp.parse(args.hparams)
    open(os.path.join(logdir, 'hparams.json'), 'w').write(hp.to_json(indent=1))
    open(os.path.join(logdir, 'args.json'), 'w').write(json.dumps(vars(args), indent=1))
    time_id = datetime.datetime.now().strftime('%m%d_%H%M')
    logging.basicConfig(format="[%(levelname)s %(asctime)s]" + " %(message)s",
                        stream=sys.stdout, level=logging.INFO)

    torch.manual_seed(0)
    infolog.set_logger(os.path.join(logdir, 'outputs_%s.log' % (time_id)))
    writer = SummaryWriter(log_dir=logdir)
    map_location = {}
    if not torch.cuda.is_available():
        map_location = {'cuda:0': 'cpu'}

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

    eval_languages = args.eval_languages.split(':') if args.eval_languages else None
    eval_speakers = args.eval_speakers.split(':') if args.eval_speakers else None
    if args.exclude_speakers in filter_keys:
        exclude_speakers = filter_keys[args.exclude_speakers]
    else:
        exclude_speakers = args.exclude_speakers.split(':') if args.exclude_speakers else None

    zipfilepath = args.zipfilepath if args.zipfilepath else os.path.join(data_dir, 'mels.zip')
    if not os.path.exists(zipfilepath):
        zipfilepath = None
    eval_meta = args.eval_meta if args.eval_meta else os.path.join(data_dir, 'metadata.eval.txt')
    feeder_eval = FeederEval(zipfilepath, eval_meta, hp, spk_to_id=spk_to_id, lang_to_id=lang_to_id,
                                 eval_lang=eval_languages, eval_spk=eval_speakers, exclude_spk=exclude_speakers,
                                 shuffle=True, keep_order=True, pick_partial=False, single=False)
    meta_index = dict([(m['n'], m) for m in feeder_eval._metadata])
    m = tacotron.Tacotron(hp)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    m.to(device)
    m.eval()
    m.decoder.train()
    state_dict = m.state_dict()
    for var in state_dict:
        logging.info("%s %s" % (var, state_dict[var].shape))

    ckpt = []
    finished_ckpt = []

    if hp.multi_lingual:
        id_to_lang = dict([(v, k) for k, v in lang_to_id.items()])

    while True:
        if len(ckpt) == 0:
            logging.info('Scanning: %s\n' % model_dir + '\n'.join(os.listdir(model_dir)))
            for l in glob.iglob(os.path.join(model_dir, 'model.ckpt-*')):
                step = l.split('-')[-1]
                if l not in finished_ckpt and step.isnumeric():
                    if eval_steps and int(step) in eval_steps:
                        pass
                    elif int(step) < args.start_step or (eval_steps and int(step) not in eval_steps) or \
                            int(step) % args.eval_interval != 0:
                        continue
                    ckpt.append((l, int(step)))
            ckpt.sort(key=lambda x: x[-1])
        if len(ckpt) == 0:
            if args.no_wait:
                logging.info('No more ckpt, exit')
                return
            logging.info('No ckpt found, sleeping...')
            time.sleep(600)
            continue

        tic = time.time()
        ckpt_path, step = ckpt[0]
        ckpt = ckpt[1:]

        eval_path = os.path.join(logdir, 'eval_%d' % (step))
        logging.info('Evaluating %s' % ckpt_path)
        os.makedirs(eval_path, exist_ok=True)

        existent_samples = []
        for f in glob.iglob(os.path.join(eval_path, '*_trim.wav')):
            name = os.path.split(f)[-1][:-9]
            existent_samples.append(name)
        if len(existent_samples) == 0 or not args.recover_eval:
            batches = feeder_eval.fetch_data()
        else:
            logging.info("%d samples found and skipped" % len(existent_samples))
            batches = feeder_eval.fetch_data(exclude=existent_samples)
        summary_windows = []
        if zipfilepath:
            mse = infolog.LookupWindow('mse_dtw', reduction='avg')
            summary_windows.append(mse)
        cer = infolog.LookupWindow('cer', reduction='avg')
        summary_windows.append(cer)

        checkpoint.load_model(ckpt_path, m, map_location=map_location)
        logging.info('Running %d batches, to %s' % (len(batches), eval_path))
        batches = batches[:hp.max_eval_batches]
        eval_futures = []
        names = []
        executor = ProcessPoolExecutor(max_workers=5)
        evaltime = 0

        for i, batch in enumerate(batches):
            logging.info("[Batch %d] Generating " % i + str(batch['names']))
            batch = dict_send_to(batch, device)
            eval_tic = time.time()
            results = eval_batch(m, batch, use_bar=False, bar_interval=500)
            evaltime += time.time() - eval_tic

            results['mel_pre'] = results['alignments']['self'] = None
            fn = partial(save_eval_results, **results, output_dir=eval_path, save_trimmed_wave=True)
            logging.info('[Batch %d] Submit thread: ' % (i) + str(batch['names']))
            eval_futures.append(executor.submit(fn))
            names.extend(batch['names'])

            if 'input_language_vecs' in batch:
                lvs = batch['input_language_vecs'].detach().cpu().numpy()
                lang_ids = [language_vec_to_id(lv) for lv in lvs]
                langs = [id_to_lang[id] for id in lang_ids]
            else:
                langs = ['' for _ in batch['names']]
            if zipfilepath:
                mse.update(langs, infolog.calculate_mse_dtw(
                            results['mel_aft'], results['generated_lengths'],
                            batch['mel_targets'], batch['target_lengths']))
        eval_futures = [f.result() for f in eval_futures]

        if transcribe_available:
            run_transcription(eval_path, names, existent_samples, meta_index, cer, step)

        for window in summary_windows:
            stats = window.summary()
            for k, v in stats:
                writer.add_scalar(k, v, global_step=step)
            window.clear()

        logging.info('Finished eval in %.3f sec (sample generation %.3f)' % (time.time() - tic, evaltime))
        finished_ckpt.append(ckpt_path)

        os.system("rsync -avu %s %s" % (os.path.join(logdir, '*'), os.path.join(model_dir, 'eval_logs')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True, help="Directory of checkpoints")
    parser.add_argument('--log-dir', required=True, help="Directory to save log and tfevents")
    parser.add_argument('--data-dir', required=True, help="Directory with data and metadata")
    parser.add_argument('--no_wait', default=None, help='Wait when there is no ckpt available')
    parser.add_argument('--zipfilepath', type=str, default=None,
                        help="Zip file of mels, use mels.zip under data-dir when not given")
    parser.add_argument('--eval_meta', type=str, default=None,
                        help="Metadata file for eval, use metadata.eval.txt under data-dir when not given")
    parser.add_argument('--eval_languages', type=str, default=None,
                        help="Languages for eval, separated by colons; use all when not given")
    parser.add_argument('--eval_speakers', type=str, default=None,
                        help="Speakers for eval under the eval_languages, separated by colon; use all when not given")
    parser.add_argument('--exclude_speakers', type=str, default=None,
                        help="Speakers to be excluded from eval, separated by colon")
    parser.add_argument('--recover_eval', type=bool, default=None,
                        help="Whether skip the samples that are found already synthesized; "
                             "enabling this may break the MSE metrics")
    parser.add_argument('--start_step', type=int, default=50000,
                        help="Mininum step of checkpoint to run eval on")
    parser.add_argument('--eval_steps', type=str, default=None,
                        help="Steps of checkpoints to run eval on; consider all checkpoints when not specified")
    parser.add_argument('--eval_interval', type=int, default=10000,
                        help="Interval of steps to run eval on; "
                             "if step % eval_interval is not zero, the checkpoint will be skipped")
    parser.add_argument('--hparams', default='', help='Alternative hparams')

    args, unparsed = parser.parse_known_args()
    print('unparsed:', unparsed)
    main(args)
