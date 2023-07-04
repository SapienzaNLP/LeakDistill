import torch
torch.autograd.set_detect_anomaly(False)

import random, traceback
import numpy as np
import copy

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import wandb
from pathlib import Path
from spring_amr.multiplier_scheduler import MultiplierScheduler

try:
    from torch.cuda.amp import autocast

    autocast_available = True
except ImportError:
    class autocast:
        def __init__(self, enabled=True): pass

        def __enter__(self): return self

        def __exit__(self, exc_type, exc_value, exc_traceback): pass


    autocast_available = False

from torch.cuda.amp.grad_scaler import GradScaler
import transformers

from spring_amr import ROOT
from spring_amr.dataset import reverse_direction
from spring_amr.optim import RAdam
from spring_amr.evaluation import write_predictions, predict_amrs, predict_sentences, compute_bleu
from spring_amr.utils import instantiate_model_and_tokenizer, instantiate_loader, class_to_dict
from spring_amr.distil import DistilLoss

from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from torch import nn
from torch.nn import functional as F

import smatch

def replace_empty_amr(file_name):
    with open(file_name, mode='r') as file:
        content = file.read().replace('()', '(e / empty)')
    with open(file_name, mode='w') as file:
        file.write(content)

def compute_smatch(test_path, predictions_path):
    replace_empty_amr(predictions_path)
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return score[2]

def replace_gnn_in_adapters(teacher_model, student_model):
    teacher_adapters = teacher_model.get_encoder().graph_adapters
    student_adapters = student_model.get_encoder().graph_adapters
    for i, t_adapter in enumerate(teacher_adapters):
        for p in t_adapter.graph_net.parameters():
            p.requires_grad_(False)
        student_adapters[i].graph_net = t_adapter.graph_net
        student_adapters[i].use_pretrained_gnn = True

def do_train(checkpoint=None, direction='amr', split_both_decoder=False, fp16=False,
             restore_optimizer=False,
             adapter_configs=None, eval_mode=False, save_best_loss=False, add_mask_task=False,
             config=None,
             teacher_model=None, distil_loss=None,
             replace_gnn=False,
             device=None,
             seed=13
             ):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.use_deterministic_algorithms(True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=split_both_decoder,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False),
        # with_adapter=init_adapter
        adapter_configs=adapter_configs,
        output_hidden_states=teacher_model is not None
    )
    model.to(device)
    model.cur_iteration = 0
    model.main_config = config

    # Teacher-Student mode
    if teacher_model is not None:
        cloned_model = copy.deepcopy(teacher_model)
        cloned_model.to(device)
        if 'copy_decoder' in config and config['copy_decoder']:
            model.model.decoder = cloned_model.get_decoder()

            if 'fix_decoder_layers' not in config:
                for p in model.get_decoder().parameters():
                    p.requires_grad_(False)

            model.model.encoder.embed_tokens = cloned_model.get_encoder().embed_tokens
            for p in model.model.encoder.embed_tokens.parameters():
                p.requires_grad_(False)

        if 'copy_encoder' in config and config['copy_encoder']:
            adapter_config = model.model.encoder.adapter_config
            model.model.encoder.layers = cloned_model.get_encoder().layers
            model.get_encoder().adapter_config = adapter_config

        if replace_gnn:
            replace_gnn_in_adapters(teacher_model, model)

    if 'fix_encoder_layers' in config:
        for l_id in config['fix_encoder_layers']:
            for p in model.get_encoder().layers[l_id].parameters():
                p.requires_grad_(False)

    if 'fix_decoder_layers' in config:
        for l_id in config['fix_decoder_layers']:
            for p in model.get_decoder().layers[l_id].parameters():
                p.requires_grad_(False)

    print(model)
    print(model.config)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('Number of trainable parameters', num_params)

    if checkpoint is not None:
        print(f'Checkpoint restored ({checkpoint})!')

    if direction == 'both' and split_both_decoder:
        params_dir_enc = list(model.model.encoder.parameters())
        params_dir_enc_check = {id(p) for p in params_dir_enc}
        params_dir_dec = set()
        params_dir_dec |= {p for p in model.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec |= {p for p in model.rev.model.decoder.parameters() if id(p) not in params_dir_enc_check}
        params_dir_dec = list(params_dir_dec)
        optimizer = RAdam(
            [{'params': params_dir_enc, 'lr': config['learning_rate']},
             {'params': params_dir_dec, 'lr': config['learning_rate'] * 2}, ],
            weight_decay=config['weight_decay'])
    else:
        optimizer = RAdam(
            model.parameters(),
            eps=config['adam_eps'],
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'])

    if checkpoint is not None and restore_optimizer:
        optimizer.load_state_dict(torch.load(checkpoint, map_location=device)['optimizer'])

    if config['scheduler'] == 'cosine':
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    elif config['scheduler'] == 'constant':
        scheduler = transformers.get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'])
    elif config['scheduler'] == 'linear':
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config['warmup_steps'],
            num_training_steps=config['training_steps'])
    else:
        raise ValueError

    scaler = GradScaler(enabled=fp16)

    if 'reduce_datasets' in config and config['reduce_datasets']:
        reduce_train, reduce_dev = 0.3, 1
    else:
        reduce_train, reduce_dev = None, None

    if not eval_mode:
        train_loader = instantiate_loader(
            config['train'],
            tokenizer,
            snt_tokenizer,
            batch_size=config['batch_size'],
            evaluation=False,
            use_recategorization=config['use_recategorization'],
            remove_longer_than=config['remove_longer_than'],
            remove_wiki=config['remove_wiki'],
            dereify=config['dereify'],
            alignment_paths=config['align_train'] if config['align_mode'] else None,
            random_reduce=reduce_train,
            align_keep_full=config['keep_full_graph'],
            align_from_metadata=config['align_from_metadata'],
            mlm_prob_upper_bound=config['aux_mask_ratio'] if 'aux_mask_ratio' in config else 0,
            extra_nodes_mask_prob=config['extra_nodes_mask_prob'] if 'extra_nodes_mask_prob' in config else 0,
            extra_nodes_contract_prob=config['extra_nodes_contract_prob'] if 'extra_nodes_contract_prob' in config else 0
        )

    dev_gold_path = ROOT / 'data/tmp/dev-gold.txt'
    dev_pred_path = ROOT / 'data/tmp/dev-pred.txt'
    dev_loader = instantiate_loader(
        config['dev'],
        tokenizer,
        snt_tokenizer,
        batch_size=config['batch_size'],
        evaluation=True, out=dev_gold_path,
        use_recategorization=config['use_recategorization'],
        remove_wiki=config['remove_wiki'],
        dereify=config['dereify'],
        alignment_paths=config['align_dev'] if config['align_mode'] else None,
        random_reduce=reduce_dev,
        align_keep_full=config['keep_full_graph']
    )

    if direction == 'amr':
        kl_div_loss = nn.KLDivLoss(reduction='none')

        if 'double_path' in config and config['double_path']:
            if config['beta_sched_steps'] != 0:
                leak_loss_multiplier = MultiplierScheduler(config['beta_start_value'], config['beta_end_value'], scheduler_params={'type': 'linear', 'steps': config['beta_sched_steps']})
            else:
                leak_loss_multiplier = MultiplierScheduler(config['beta_start_value'], scheduler_params={'type': 'constant'})

        if config['double_path'] and config['clone_mode_for_dp']:
            # This case for experimenting with two separate models (teacher and student), instead of doing two forward paths in one model
            leaked_model = copy.deepcopy(model)
            optimizer.add_param_group({'params': leaked_model.parameters()})
            leaked_model.to(device)
        else:
            leaked_model = model

        def train_step(engine, batch):
            log_vars = {}
            model.cur_iteration += 1
            model.train()
            x, y, extra = batch

            if config['align_mode']:
                leaked_model.get_encoder().cur_edges = extra['align_graph_edges']
                leaked_model.get_encoder().not_edges = extra['align_graph_neg_edges']
                leaked_model.get_encoder().contracted_edges = extra['align_contracted_edges']
                leaked_model.get_encoder().orig_graph_data = extra['orig_graph_data']

            # For GLM
            if config.adapter['encoder']['graph_mode'] and config.adapter['encoder']['leak_mode'] \
                    and config['keep_full_graph'] and config['adapter']['encoder']['extra_nodes_as_input']:
                _, _, new_attention_mask, _ = extra['orig_graph_data'][1]
                x['attention_mask'] = new_attention_mask.int()

            model.amr_mode = True
            with autocast(enabled=fp16):
                if teacher_model is not None:
                    teacher_model.get_encoder().cur_edges = extra['align_graph_edges']
                    teacher_model.get_encoder().orig_graph_data = extra['orig_graph_data']
                    old_attention_mask = x['attention_mask']

                    if config['keep_full_graph'] and config['adapter']['encoder']['extra_nodes_as_input']:
                        _, _, new_attention_mask, _ = extra['orig_graph_data'][1]
                        x['attention_mask'] = new_attention_mask.int()
                    teacher_output = teacher_model(**x, **y)
                    x['attention_mask'] = old_attention_mask
                # The outcome model forward path
                output = model(**x, **y)
                loss = output.loss
                orig_loss = loss.item()
                # LeakDistill part
                if config['double_path']:
                    output.cls_loss = loss
                    # When extra nodes are treated as BART input we change input_ids and attention_mask
                    leaked_x = x.copy()
                    if config['keep_full_graph'] and config['adapter']['encoder']['extra_nodes_as_input']:
                        _, _, new_attention_mask, _ = extra['orig_graph_data'][1]
                        leaked_x['attention_mask'] = new_attention_mask.int()
                        leaked_x['input_ids'] = extra['leak_input_ids']
                    # Switch on adapters and leak mode
                    leaked_model.get_encoder().leak_path = True
                    # Run leak path
                    output_2 = leaked_model(**leaked_x, **y)
                    # Return to original settings
                    leaked_model.get_encoder().leak_path = False
                    # Reset loss in case of no L_nll loss
                    if config['exclude_orig_loss']:
                        loss = 0
                    log_vars['leak_loss'] = output_2.loss
                    log_vars['leak_multiplier'] = leak_loss_multiplier()
                    loss += log_vars['leak_multiplier'] * log_vars['leak_loss']
                    # KL loss calculation
                    if config['kl_multiplier'] > 0:
                        temp = config['kl_temperature']
                        target_mask = y['labels'] != model.config.pad_token_id
                        soft_pred = F.log_softmax(output.logits / temp, dim=-1)
                        soft_targets = F.softmax(output_2.logits / temp, dim=-1)
                        output.kl_div_loss = kl_div_loss(soft_pred[target_mask], soft_targets[target_mask]).sum(dim=-1).mean()
                        log_vars['kl_loss'] = config['kl_multiplier'] * output.kl_div_loss
                        loss += log_vars['kl_loss']
                        '''
                        # Reverse KL, this option is not fully explored
                        soft_pred = F.log_softmax(output_2.logits / temp, dim=-1)
                        soft_targets = F.softmax(output.logits / temp, dim=-1)
                        output.reverse_kl_div_loss = kl_div_loss(soft_pred[target_mask], soft_targets[target_mask]).sum(dim=-1).mean()
                        log_vars['rev_kl_loss'] = config['kl_multiplier'] * output.reverse_kl_div_loss
                        loss += log_vars['rev_kl_loss']
                        '''
                # Vanilla Knowledge Distilaltion
                if teacher_model is not None:
                    encoder_mask = x['attention_mask'] == 1
                    target_mask = y['labels'] != model.config.pad_token_id
                    loss = (1 - config['distil_alpha']) * distil_loss(['kl'], teacher_output, output, target_mask, encoder_mask, teacher_model)

            scaler.scale((loss / config['accum_steps'])).backward()
            if config['log_wandb']:
                log_vars.update({'tr_cls_loss': output.cls_loss})
                log_vars['orig_loss'] = orig_loss
                wandb.log(log_vars, step=engine.state.iteration)
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            if config['adapter']['encoder']['leak_mode']:
                model.get_encoder().cur_edges = extra['align_graph_edges']
                model.get_encoder().orig_graph_data = extra['orig_graph_data']

                if config['keep_full_graph'] and config['adapter']['encoder']['extra_nodes_as_input']:
                    _, _, new_attention_mask, _ = extra['orig_graph_data'][1]
                    x['attention_mask'] = new_attention_mask.int()

            model.amr_mode = True
            output = model(**x, **y)
            loss = output.loss

            return loss.item()

    elif direction == 'text':
        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            with autocast(enabled=fp16):
                loss, *_ = model.rev(**x, **y)
            scaler.scale((loss / config['accum_steps'])).backward()
            return loss.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            loss, *_ = model(**x, **y)
            return loss.item()

    elif direction == 'both':
        def train_step(engine, batch):
            model.train()
            x, y, extra = batch
            model.amr_mode = True
            with autocast(enabled=fp16):
                loss1, *_ = model(**x, **y)
            scaler.scale((loss1 / config['accum_steps'] * 0.5)).backward()
            loss1 = loss1.item()
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            with autocast(enabled=fp16):
                loss2, *_ = model.rev(**x, **y)
            scaler.scale((loss2 / config['accum_steps'] * 0.5)).backward()
            return loss1, loss2.item()

        @torch.no_grad()
        def eval_step(engine, batch):
            model.eval()
            x, y, extra = batch
            model.amr_mode = True
            loss1, *_ = model(**x, **y)
            x, y = reverse_direction(x, y)
            model.rev.amr_mode = False
            loss2, *_ = model.rev(**x, **y)
            return loss1.item(), loss2.item()

    else:
        raise ValueError

    if not eval_mode:
        trainer = Engine(train_step)
    evaluator = Engine(eval_step)

    if not eval_mode:
        @trainer.on(Events.STARTED)
        def update(engine):
            print('training started!')

        @trainer.on(Events.EPOCH_COMPLETED)
        @trainer.on(Events.ITERATION_COMPLETED(every=config['accum_steps']))
        def update(engine):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_trn_loss(engine):
            log_msg = f"training epoch: {engine.state.epoch}"
            if direction in ('amr', 'both'):
                log_msg += f" | loss_amr: {engine.state.metrics['trn_amr_loss']:.3f}"
            if direction in ('text', 'both'):
                log_msg += f" | loss_text: {engine.state.metrics['trn_text_loss']:.3f}"
            print(log_msg)

        @trainer.on(Events.EPOCH_COMPLETED)
        def run_dev_eval(engine):
            dev_loader.batch_size = config['batch_size']
            dev_loader.device = next(model.parameters()).device
            evaluator.run(dev_loader)

    if not config['best_loss']:
        if direction in ('amr', 'both'):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine, remove_align=False):
                if ('calc_smatch' not in config or not config['calc_smatch']) or ('calc_smatch' in config and config['calc_smatch'] and
                                    'smatch_iteration' in config and model.cur_iteration < config['smatch_iteration']):
                    engine.state.metrics['dev_smatch'] = 0
                else:
                    device = next(model.parameters()).device
                    dev_loader.device = device
                    graphs = predict_amrs(dev_loader, model, tokenizer,
                                          restore_name_ops=config['collapse_name_ops'],
                                          remove_align=remove_align)
                    write_predictions(dev_pred_path, tokenizer, graphs)
                    try:
                        smatch = compute_smatch(dev_gold_path, dev_pred_path)
                    except:
                        traceback.print_exc()
                        smatch = 0.
                    engine.state.metrics['dev_smatch'] = smatch

        if direction in ('text', 'both'):
            @evaluator.on(Events.EPOCH_COMPLETED)
            def smatch_eval(engine):
                device = next(model.parameters()).device
                dev_loader.device = device
                pred_sentences = predict_sentences(dev_loader, model.rev, tokenizer, beam_size=config['beam_size'])
                bleu = compute_bleu(dev_loader.dataset.sentences, pred_sentences)
                engine.state.metrics['dev_bleu'] = bleu.score

    @evaluator.on(Events.EPOCH_COMPLETED)
    def log_dev_loss(engine):
        log_msg = f"dev epoch: {trainer.state.epoch}" if not eval_mode else ""
        if direction in ('amr', 'both'):
            log_msg += f" | loss_amr: {engine.state.metrics['dev_amr_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | smatch: {engine.state.metrics['dev_smatch']:.3f}"
        if direction in ('text', 'both'):
            log_msg += f" | loss_text: {engine.state.metrics['dev_text_loss']:.3f}"
            if not config['best_loss']:
                log_msg += f" | bleu: {engine.state.metrics['dev_bleu']:.3f}"
        print(log_msg)

    if not eval_mode:
        if direction == 'amr':
            RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_amr_loss')
            RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_amr_loss')
        elif direction == 'text':
            RunningAverage(output_transform=lambda out: out).attach(trainer, 'trn_text_loss')
            RunningAverage(output_transform=lambda out: out).attach(evaluator, 'dev_text_loss')
        elif direction == 'both':
            RunningAverage(output_transform=lambda out: out[0]).attach(trainer, 'trn_amr_loss')
            RunningAverage(output_transform=lambda out: out[1]).attach(trainer, 'trn_text_loss')
            RunningAverage(output_transform=lambda out: out[0]).attach(evaluator, 'dev_amr_loss')
            RunningAverage(output_transform=lambda out: out[1]).attach(evaluator, 'dev_text_loss')

        if config['log_wandb']:
            from ignite.contrib.handlers.wandb_logger import WandBLogger
            wandb_logger = WandBLogger(init=False)

            if direction == 'amr':
                wandb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="iterations/trn_amr_loss",
                    output_transform=lambda loss: loss
                )
            elif direction == 'text':
                wandb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="iterations/trn_text_loss",
                    output_transform=lambda loss: loss
                )
            if direction == 'both':
                wandb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="iterations/trn_amr_loss",
                    output_transform=lambda loss: loss[0]
                )
                wandb_logger.attach_output_handler(
                    trainer,
                    event_name=Events.ITERATION_COMPLETED,
                    tag="iterations/trn_text_loss",
                    output_transform=lambda loss: loss[1]
                )

            if direction == 'amr':
                metric_names_trn = ['trn_amr_loss']
                metric_names_dev = ['dev_amr_loss']
                if not config['best_loss']:
                    metric_names_dev.append('dev_smatch')
            elif direction == 'text':
                metric_names_trn = ['trn_text_loss']
                metric_names_dev = ['dev_text_loss']
                if not config['best_loss']:
                    metric_names_dev.append('dev_bleu')
            elif direction == 'both':
                metric_names_trn = ['trn_amr_loss', 'trn_text_loss']
                metric_names_dev = ['dev_amr_loss', 'dev_smatch']
                if not config['best_loss']:
                    metric_names_dev.extend(['dev_text_loss', 'dev_bleu'])

            wandb_logger.attach_output_handler(
                trainer,
                event_name=Events.EPOCH_COMPLETED,
                tag="epochs",
                metric_names=metric_names_trn,
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            wandb_logger.attach_output_handler(
                evaluator,
                event_name=Events.EPOCH_COMPLETED,
                tag="epochs",
                metric_names=metric_names_dev,
                global_step_transform=lambda *_: trainer.state.iteration,
            )

            @trainer.on(Events.ITERATION_COMPLETED)
            def wandb_log_lr(engine):
                wandb.log({'lr': scheduler.get_last_lr()[0]}, step=engine.state.iteration)

        if config['save_checkpoints']:
            if direction in ('amr', 'both'):
                if config['best_loss']:
                    prefix = 'best-loss-amr'
                    score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
                else:
                    prefix = 'best-smatch'
                    score_function = lambda x: evaluator.state.metrics['dev_smatch']
            else:
                if config['best_loss']:
                    prefix = 'best-loss-text'
                    score_function = lambda x: 1 / evaluator.state.metrics['dev_amr_loss']
                else:
                    prefix = 'best-bleu'
                    score_function = lambda x: evaluator.state.metrics['dev_bleu']

            to_save = {'model': model, 'optimizer': optimizer}
            if config['log_wandb']:
                where_checkpoints = str(wandb_logger.run.dir)
            else:
                root = ROOT / 'runs'
                try:
                    root.mkdir()
                except:
                    pass
                where_checkpoints = root / str(len(list(root.iterdir())))
                try:
                    where_checkpoints.mkdir()
                except:
                    pass
                where_checkpoints = str(where_checkpoints)

            print(where_checkpoints)
            handler = ModelCheckpoint(
                where_checkpoints,
                prefix,
                n_saved=1,
                create_dir=True,
                score_function=score_function,
                global_step_transform=global_step_from_engine(trainer),
            )
            evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler, to_save)
            if save_best_loss:
                handler_2 = ModelCheckpoint(
                    where_checkpoints,
                    'best-loss-amr',
                    n_saved=1,
                    create_dir=True,
                    score_function=lambda x: 1 / evaluator.state.metrics['dev_amr_loss'],
                    global_step_transform=global_step_from_engine(trainer),
                )
                evaluator.add_event_handler(Events.EPOCH_COMPLETED, handler_2, to_save)

    if torch.cuda.is_available():
        model.cuda()
    device = next(model.parameters()).device
    if not eval_mode:
        train_loader.device = device
        trainer.run(train_loader, max_epochs=config['max_epochs'])
    else:
        dev_loader.device = device
        smatch_eval(evaluator)


def init_parser(default_config=ROOT / 'configs/config.yaml'):
    parser = ArgumentParser(
        description="AMR parser",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=Path, default=default_config,
                        help='Use the following config for hparams.')
    parser.add_argument('--checkpoint', type=str,
                        help='Warm-start from a previous fine-tuned checkpoint.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device. 'cpu', 'cuda', 'cuda:<n>'.")
    parser.add_argument('--beamsize', type=int, default='5')
    parser.add_argument('--teacher-checkpoint', type=str, default='')
    parser.add_argument('--seed', type=int, default='13')

    return parser


def parser_and_config(project_name="",
                      default_config=ROOT / 'configs/config.yaml',
                      runs_dir=str(ROOT / 'runs/'),
                      force_default_config=True,
                      skip_wandb=False,
                      func_add_args=None):

    parser = init_parser(default_config)
    if func_add_args is not None:
        func_add_args(parser)
    args, unknown = parser.parse_known_args()

    if args.fp16 and autocast_available:
        raise ValueError('You\'ll need a newer PyTorch version to enable fp16 training.')

    if force_default_config:
        args.config = Path(default_config)

    with args.config.open() as y:
        config = yaml.load(y, Loader=yaml.FullLoader)

    if config['log_wandb'] and not skip_wandb:
        wandb.init(
            project=project_name if project_name != '' else config['wandb_project'],
            config=config,
            dir=runs_dir)
        config = wandb.config

    print(config)

    return args, config

def run_with_teacher(
                     student_checkpoint=None,
                     student_config=None,
                     replace_gnn=False,
                     runs_dir=str(ROOT / 'runs/'),
                     ):
    '''
    This function is used for vanilla Knowledge Distillation
    :param teacher_checkpoint:
    :param student_checkpoint:
    :param student_config:
    :param replace_gnn:
    :param runs_dir:
    :return:
    '''
    args, config = parser_and_config(
         default_config=ROOT / 'configs/config_leak.yaml',
         force_default_config=True,
         skip_wandb=True
    )

    teacher_model, _, _ = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=args.teacher_checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=False,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False),
        adapter_configs=config['adapter'],
        output_hidden_states=True
    )
    # Freeze teacher's parameters
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    teacher_model.to(args.device)

    args, config = parser_and_config(
        default_config=student_config,
        runs_dir=runs_dir
    )

    do_train(
        config=config,
        checkpoint=student_checkpoint,
        direction='amr',  # args.direction,
        split_both_decoder=False,
        fp16=args.fp16,
        restore_optimizer=False,
        adapter_configs=config['adapter'],
        save_best_loss=False,
        add_mask_task=config['aux_mask_task'] if 'aux_mask_task' in config else False,
        teacher_model=teacher_model,
        distil_loss=DistilLoss(config['distil_temp']).to(args.device),
        replace_gnn=replace_gnn,
        device=args.device,
        seed=args.seed
    )

def default_run(
        default_config=ROOT / 'configs/config.yaml',
        runs_dir=str(ROOT / 'runs/'),
        project_name="Adapters",
        checkpoint=None,
        eval_mode=False,
        restore_optimizer=True
):
    args, config = parser_and_config(project_name=project_name, default_config=default_config, runs_dir=runs_dir)

    do_train(
        config=config,
        checkpoint=checkpoint,
        direction='amr',  # args.direction,
        split_both_decoder=False,
        fp16=args.fp16,
        restore_optimizer=restore_optimizer,
        adapter_configs=config['adapter'],
        save_best_loss=False,
        add_mask_task=config['aux_mask_task'] if 'aux_mask_task' in config else False,
        eval_mode=eval_mode,
        seed=args.seed
    )