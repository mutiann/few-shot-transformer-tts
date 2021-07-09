import torch
import time
import copy
from hyperparams import hparams as hp
import numpy as np
import logging
import tqdm
import traceback
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from utils.infolog import plot_attn, plot_mel
from utils.audio import mel2wav, save_wav, trim_silence_intervals


def eval_batch(model_eval, data, use_bar=True, bar_interval=10):
    with torch.no_grad():
        tic = time.time()
        batch = copy.copy(data)
        device = batch['inputs'].device
        batch_size = batch['inputs'].shape[0]
        target_lengths = torch.ones([batch_size], dtype=torch.int32, device=device)
        finished = torch.zeros([batch_size], dtype=torch.bool, device=device)
        mels = torch.zeros([batch_size, 0, hp.num_mels], dtype=torch.float32, device=device)  # [B, T, M]
        if 'input_spk_ids' not in batch:
            batch['input_spk_ids'] = None
        if 'input_language_vecs' not in batch:
            batch['input_language_vecs'] = None
        enc_outputs = model_eval.encoder(batch['inputs'], batch['input_lengths'],
                                         batch['input_spk_ids'], batch['input_language_vecs'])
        torch.cuda.empty_cache()
        if use_bar:
            bar = tqdm.tqdm()
        while not torch.all(finished) and mels.shape[1] < hp.max_generation_frames:
            try:
                decoder_input = torch.cat([mels, torch.zeros([batch_size, 1, hp.num_mels], device=device)], dim=1)

                mel_bef, stop_logits, align = \
                    model_eval.decoder(enc_outputs, batch['input_lengths'],
                                       decoder_input, target_lengths, leave_one=True)
                stop = stop_logits[:, -1] > 0
                mels = torch.cat([mels, mel_bef[:, -1:]], dim=1)
                finished = torch.logical_or(finished, stop)
                target_lengths = torch.where(finished, target_lengths, target_lengths + 1)

                if mels.shape[1] % bar_interval == 0 and bar_interval != -1:
                    if use_bar:
                        bar.update(bar_interval)
                    else:
                        print(mels.shape[1])
            except:
                traceback.print_exc()
                break

        mel_aft = mels + model_eval.postnet(mels, target_lengths)

        # Evade memory leakage
        for key in ['self', 'encdec']:
            for i in range(len(align[key])):
                align[key][i] = align[key][i].cpu().numpy()

        toc = time.time()
        total_length = target_lengths.sum().item()
        logging.info("Time: %.4f, Samples: %d, Length: %d, Max length: %d, Real-time Factor: %.4f" % (
            toc - tic, mels.shape[0], total_length, target_lengths.max().item(),
            (toc - tic) / total_length * 80))

        return {'names': data['names'], 'mel_pre': mels.cpu().numpy(),
                'mel_aft': mel_aft.cpu().numpy(), 'alignments': align,
                'input_lengths': list(batch['input_lengths'].cpu().numpy()),
                'generated_lengths': list(target_lengths.cpu().numpy())}


def save_eval_results(names, mel_pre, mel_aft, alignments, input_lengths, generated_lengths,
                      output_dir, save_trimmed_wave=False, n_plot_alignment=None):
    def save_i(i):
        try:
            name = names[i]
            mel = mel_aft[i][:generated_lengths[i]]
            np.save(os.path.join(output_dir, '%s.npy' % name), mel)
            wav = mel2wav(mel)
            save_wav(wav, os.path.join(output_dir, '%s.wav' % name))
            if save_trimmed_wave:
                wav_trim = trim_silence_intervals(wav)
                save_wav(wav_trim, os.path.join(output_dir, '%s_trim.wav' % name))
            plot_mel(os.path.join(output_dir, '%s_mel.png' % name), mel)

            if n_plot_alignment is None or i < n_plot_alignment:
                aligns = [a[i].transpose([0, 2, 1]) for a in alignments["encdec"]]
                plot_attn(aligns, os.path.join(output_dir, '%s_align.png' % (name)),
                          enc_length=input_lengths[i], dec_length=generated_lengths[i])

        except:
            logging.error('Fail to produce eval output: ' + names[i])
            logging.error(traceback.format_exc())

    tic = time.time()
    executor = ThreadPoolExecutor(max_workers=4)
    futures = []
    for i in range(len(names)):
        futures.append(executor.submit(partial(save_i, i=i)))
    [future.result() for future in futures]

    logging.info('[%s] Finished saving evals in %.2f secs: ' %
                 (threading.current_thread().name, time.time() - tic) + str(names))