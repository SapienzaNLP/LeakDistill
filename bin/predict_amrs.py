import sys, os
sys.path.append('.')
from spring_amr.train_utils import *
from spring_amr.penman import encode


def func_add_args(parser):
    parser.add_argument('--snt-to-tok', action='store_false',
                        help='Text transformation to the format being trained on. Set it to false if the text is already tokenized')
    parser.add_argument('--datasets', type=str,
                        help='Path to AMR files', default='')
    parser.add_argument('--gold-path', type=str, default=ROOT / 'data/tmp/test-gold.txt')
    parser.add_argument('--pred-path', type=str, default=ROOT / 'data/tmp/test-pred.txt')


args, config = parser_and_config(project_name="",
                                 default_config=ROOT / 'configs/config_leak_distill.yaml',
                                 skip_wandb=True,
                                 func_add_args=func_add_args
                                 )

device = torch.device(args.device)
model, tokenizer, snt_tokenizer = instantiate_model_and_tokenizer(
        config['model'],
        checkpoint=args.checkpoint,
        additional_tokens_smart_init=config['smart_init'],
        dropout=config['dropout'],
        attention_dropout=config['attention_dropout'],
        from_pretrained=config['warm_start'],
        init_reverse=False,
        penman_linearization=config['penman_linearization'],
        collapse_name_ops=config['collapse_name_ops'],
        use_pointer_tokens=config['use_pointer_tokens'],
        raw_graph=config.get('raw_graph', False),
        adapter_configs=config['adapter']
    )
model.amr_mode = True
model.to(device)
model.main_config = config

gold_path = Path(args.gold_path)
pred_path = Path(args.pred_path)

loader = instantiate_loader(
    config['test'] if not args.datasets else args.datasets,
    tokenizer,
    snt_tokenizer,
    batch_size=config['batch_size'],
    evaluation=True,
    out=gold_path,
    use_recategorization=config['use_recategorization'],
    remove_wiki=config['remove_wiki'],
    dereify=config['dereify'],
    alignment_paths=None, #config['align_test'] if 'align_test' in config else None, # This is used when files contains ::tok
    align_from_metadata=config['align_from_metadata'],
    snt_to_tok=args.snt_to_tok
)

loader.device = device

print(args)

graphs = predict_amrs(
        loader,
        model,
        tokenizer,
        beam_size=args.beamsize,# config['beam_size'],
        restore_name_ops=config['collapse_name_ops']
    )

pieces = [encode(g) for g in graphs]
pred_path.write_text('\n\n'.join(pieces))

score = compute_smatch(gold_path, pred_path)
print(f'Smatch: {score:.5f}')
