import argparse


def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-lingual Task-Oriented Dialog")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="multilingual_dst.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--trans_lang", type=str, default="de", help="Choose a language to transfer")
    parser.add_argument("--task", default="sc", choices=('sc', 'dst'), help="Choose a task")
    parser.add_argument("--domain", default="book", choices=('book', 'dvd', 'music'), help="Choose a domain")
    parser.add_argument("--loss_func", default="kl", choices=('kl', 'mse', 'cos', 'mix'), help="Choose a loss function")
    # binarize data
    parser.add_argument("--ontology_class_path", type=str, default="data/dst/dst_data/ontology_classes.json",
                        help="Path of ontology classes")
    parser.add_argument("--ontology_mapping_path", type=str, default="data/dst/dst_data/ontology-mapping.json",
                        help="Path of ontology mapping")

    # model parameters
    parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate for lstm")
    parser.add_argument("--kl1", type=float, default=0.5, help="kl1")
    parser.add_argument("--embed_size", type=int, default=768, help="Embedding size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size (should be same as embedding size)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_length", type=int, default=256, help="max_length")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--early_stop", type=int, default=10,
                        help="No improvement after several epoch, we stop training")

    # number of classes for slots and request
    parser.add_argument("--food_class", type=int, default=76, help="the number of classes for food slot (include none)")
    parser.add_argument("--price_range_class", type=int, default=5,
                        help="the number of classes for price range slot (include none)")
    parser.add_argument("--area_class", type=int, default=7, help="the number of classes for area slot (include none)")
    parser.add_argument("--request_class", type=int, default=7, help="the number of classes for request")

    # mix languages training
    parser.add_argument("--mix_train", default=True, action="store_true", help="Mix language training")
    parser.add_argument("--multi_view", default=False, action="store_true", help="multi view training")
    parser.add_argument("--mapping_for_mix", type=str, default="dict/en2de.dict",
                        help="mapping for mix language training")
    parser.add_argument("--ptm_folder", type=str, default="xlm-roberta-base")
    parser.add_argument("--dynamic_mix", action="store_true")
    parser.add_argument("--model_type", choices=('split', 'concat'))
    parser.add_argument("--dynamic_ratio", default=1.0,type=float)
    parser.add_argument("--dev_ratio", default=0.8, type=float)
    parser.add_argument("--lock_embedding", action="store_true")
    parser.add_argument("--saliency", action="store_true")

    parser.add_argument("--food_emb_file", type=str, default="")
    parser.add_argument("--price_emb_file", type=str, default="")
    parser.add_argument("--area_emb_file", type=str, default="")
    parser.add_argument("--request_emb_file", type=str, default="")
    parser.add_argument("--train_resample_ratio", type=int, default=1)
    params = parser.parse_args()

    return params