import argparse

def init_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    arg_parser.add_argument('--cuda', action='store_true', help='use gpu')
    arg_parser.add_argument('--encode_dependency', action='store_true', help='encode dependency')
    arg_parser.add_argument('--encode_entity', action='store_true', help='encode entity')
    arg_parser.add_argument('--lang', choices=['python', 'lambda_dcs', 'wikisql', 'prolog'], default='python')
    arg_parser.add_argument('--mode', choices=['train', 'self_train', 'train_decoder', 'train_semi', 'log_semi', 'test',
                                               'sample'], default='train', help='run mode')
    arg_parser.add_argument('--model_name', choices=['transformer', 'rnn', 'table', 'sketch'], default='rnn',
                            help='model name')
    arg_parser.add_argument('--curriculum', action='store_true', help='curriculum_learning')
    arg_parser.add_argument('--curriculum_step', default=10, type=int, help='curriculum_step')

    arg_parser.add_argument('--lstm', choices=['lstm', 'lstm_with_dropout', 'parent_feed'], default='lstm')

    arg_parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    arg_parser.add_argument('--glove_embed_path', type=str)
    arg_parser.add_argument('--glove_300_embed_path', type=str)

    arg_parser.add_argument('--lr_scheduler', action='store_true', help='use learning rate scheduler')
    arg_parser.add_argument('--lr_scheduler_gammar', default=0.5, type=float,
                            help='decay rate of learning rate scheduler')

    arg_parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    arg_parser.add_argument('--unsup_batch_size', default=10, type=int)
    arg_parser.add_argument('--beam_size', default=1, type=int, help='beam size for beam search')
    arg_parser.add_argument('--sample_size', default=5, type=int, help='sample size')
    arg_parser.add_argument('--pos_tag_embed_size', default=10, type=int, help='size of pos tag embeddings')
    arg_parser.add_argument('--entity_embed_size', default=5, type=int, help='size of entity tag embeddings')
    arg_parser.add_argument('--embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--col_embed_size', default=300, type=int, help='size of word embeddings')
    arg_parser.add_argument('--parent_feeding', action='store_true', help='enable parent feeding')
    arg_parser.add_argument('--encode_sketch', action='store_true', help='encode sketch')
    arg_parser.add_argument('--sketch_history', action='store_true', help='use sketch history')
    arg_parser.add_argument('--column_pointer', action='store_true', help='use column pointer')
    arg_parser.add_argument('--loss_epoch_threshold', default=20, type=int, help='loss epoch threshold')
    arg_parser.add_argument('--sketch_loss_coefficient', default=0.2, type=float, help='sketch loss coefficient')
    arg_parser.add_argument('--sentence_features', action='store_true', help='use sentence features')
    arg_parser.add_argument('--stanford_tokenized', action='store_true', help='use stanford tokenization')

    arg_parser.add_argument('--bert_model', default='grappa-large', type=str, help='bert_model')
    arg_parser.add_argument('--encoder_dim', default=1024, type=int, help='size of encoder_dim')

    arg_parser.add_argument('--action_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--field_embed_size', default=64, type=int, help='size of word embeddings')
    arg_parser.add_argument('--type_embed_size', default=32, type=int, help='size of word embeddings')
    arg_parser.add_argument('--hidden_size', default=300, type=int, help='size of LSTM hidden states')
    arg_parser.add_argument('--att_vec_size', default=300, type=int, help='size of attentional vector')
    arg_parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
    arg_parser.add_argument('--word_dropout', default=0.2, type=float, help='word dropout rate')
    arg_parser.add_argument('--kl_anneal', default=False, action='store_true')
    arg_parser.add_argument('--alpha', default=0.1, type=float)


    # readout layer
    arg_parser.add_argument('--no_query_vec_to_action_map', default=False, action='store_true')
    arg_parser.add_argument('--readout', default='identity', choices=['identity', 'non_linear'])
    arg_parser.add_argument('--query_vec_to_action_diff_map', default=False, action='store_true')

    # supervised attention
    arg_parser.add_argument('--sup_attention', default=False, action='store_true')

    # wikisql
    arg_parser.add_argument('--column_att', choices=['dot_prod', 'affine'], default='affine')
    arg_parser.add_argument('--answer_prune', dest='answer_prune', action='store_true')
    arg_parser.add_argument('--no_answer_prune', dest='answer_prune', action='store_false')
    arg_parser.set_defaults(answer_prune=True)

    # parent information switch and input feeding
    arg_parser.add_argument('--no_parent_production_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_field_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_field_type_embed', default=False, action='store_true')
    arg_parser.add_argument('--no_parent_state', default=False, action='store_true')
    arg_parser.add_argument('--no_input_feed', default=False, action='store_true')
    arg_parser.add_argument('--no_copy', default=False, action='store_true')

    arg_parser.add_argument('--asdl_file', type=str)
    arg_parser.add_argument('--pos_tag', type=str, help='path of the pos tag dictionary', default='')
    arg_parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary',
                            default="/home/v-zezhan/Seq2Tree/data/vocab.bin")
    arg_parser.add_argument('--table_vocab', type=str, help='path of the serialized table vocabulary',
                            default='/home/v-zezhan/Seq2Tree/data/table_vocab.bin')
    arg_parser.add_argument('--train_src', type=str, help='path to the training source file')
    arg_parser.add_argument('--unlabeled_file', type=str, help='path to the training source file')
    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    arg_parser.add_argument('--test_file', type=str, help='path to the test target file')
    arg_parser.add_argument('--prior_lm_path', type=str, help='path to the prior LM')

    # self-training
    arg_parser.add_argument('--load_decode_results', default=None, type=str)

    # semi-supervised learning arguments
    arg_parser.add_argument('--load_decoder', default=None, type=str)
    arg_parser.add_argument('--load_src_lm', default=None, type=str)

    arg_parser.add_argument('--baseline', choices=['mlp', 'src_lm', 'src_lm_and_linear'], default='mlp')
    arg_parser.add_argument('--prior', choices=['lstm', 'uniform'])
    arg_parser.add_argument('--load_prior', type=str, default=None)
    arg_parser.add_argument('--clip_learning_signal', type=float, default=None)
    arg_parser.add_argument('--begin_semisup_after_dev_acc', type=float, default=0.,
                            help='begin semi-supervised learning after'
                                 'we have reached certain dev performance')

    arg_parser.add_argument('--decode_max_time_step', default=40, type=int, help='maximum number of time steps used '
                                                                                 'in decoding and sampling')
    arg_parser.add_argument('--unsup_loss_weight', default=1., type=float, help='loss of unsupervised learning weight')

    arg_parser.add_argument('--valid_metric', default='sp_acc', choices=['nlg_bleu', 'sp_acc'],
                            help='metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int)
    arg_parser.add_argument('--log_every', default=10, type=int, help='every n iterations to log training statistics')

    arg_parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    arg_parser.add_argument('--toy', action='store_true',
                            help='If set, use small data; used for fast debugging.')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='save decoding results to file')
    arg_parser.add_argument('--patience', default=5, type=int, help='training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int)
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='if specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int)
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False)
    arg_parser.add_argument('--verbose', action='store_true', default=False)
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=False,
                            help='only evaluate the top prediction in validation')

    arg_parser.add_argument('--train_opt', default="reinforce", type=str, choices=['reinforce', 'st_gumbel'])
    arg_parser.add_argument('--dataset', default="/home/v-zezhan/Seq2Tree/data", type=str)

    arg_parser.add_argument('--epoch', default=50, type=int, help='Maximum Epoch')
    arg_parser.add_argument('--save', default='', type=str,
                            help="Path to save the checkpoint and logs of epoch")
    arg_parser.add_argument('--decoder_heads', type=int, default=4, help='num heads in Transformer Decoder')
    arg_parser.add_argument('--encoder_heads', type=int, default=4, help='num heads in Transformer Encoder')

    arg_parser.add_argument('--task_num', type=int, default=10)
    arg_parser.add_argument('--task_path', type=str, default="data/spider_cl_data_0724/task_{}/{}")
    arg_parser.add_argument('--use_grappa', action="store_true")
    arg_parser.add_argument('--vocab_path', type=str, default='./vocab.pkl')
    arg_parser.add_argument('--cl_name', type=str, default="none")
    arg_parser.add_argument('--semi_name', type=str, default="none")
    arg_parser.add_argument('--stu_sampling_name', type=str, default="none")
    arg_parser.add_argument('--tec_sampling_name', type=str, default="none")
    arg_parser.add_argument('--candidate_size', type=int, default=50)
    arg_parser.add_argument('--memory_size', type=int, default=20)
    arg_parser.add_argument('--warm_boot_epoch', type=int, default=15)
    arg_parser.add_argument('--device', type=str, default="0")
    arg_parser.add_argument('--model', type=str, default="naive")
    arg_parser.add_argument('--eval_epoch', type=int, default=20)
    arg_parser.add_argument('--k_mediod_max_epoch', type=int, default=20)
    arg_parser.add_argument('--st_rate', type=float, default=0.3)
    arg_parser.add_argument('--thread_num', type=int, default=6)
    arg_parser.add_argument('--st_epoch', type=int, default=10)
    arg_parser.add_argument('--stu_semi', action="store_true")
    arg_parser.add_argument('--tec_semi', action="store_true")
    arg_parser.add_argument('--stu_cl', action="store_true")
    arg_parser.add_argument('--tec_cl', action="store_true")
    arg_parser.add_argument('--bert_lr', type=float, default=2e-5)
    arg_parser.add_argument('--wo_stu_sql', action="store_true")
    arg_parser.add_argument('--wo_tec_sql', action="store_true")

    args = arg_parser.parse_args()
    return args