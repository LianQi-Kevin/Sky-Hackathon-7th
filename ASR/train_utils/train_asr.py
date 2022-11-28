import yaml
import nemo
import uuid
import time
import nemo.collections.asr as nemo_asr
import pytorch_lightning as pl
import argparse
import os


def train_asr_nemo(train_manifest, valid_manifest, config_path, pretrained_model, epochs, output_path):
    # nemo version
    print("nemo version: {}".format(nemo.__version__))

    # File path exists
    assert os.path.exists(train_manifest), "train manifest {} not found".format(train_manifest)
    assert os.path.exists(valid_manifest), "valid manifest {} not found".format(valid_manifest)
    assert os.path.exists(config_path), "config file {} not found".format(config_path)
    assert os.path.exists(pretrained_model), "pretrained model {} not found".format(pretrained_model)

    # load pretrained model
    citrinet = nemo_asr.models.EncDecCTCModel.restore_from(pretrained_model)
    print("Successful load pretrained model")

    # load config file
    with open(config_path) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    # set dataset path
    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['validation_ds']['manifest_filepath'] = valid_manifest
    citrinet.setup_training_data(train_data_config=params['model']['train_ds'])
    citrinet.setup_validation_data(val_data_config=params['model']['validation_ds'])

    # set train config
    trainer = pl.Trainer(gpus=1, max_epochs=epochs)

    # training
    print("Start Training Model")
    trainer.fit(citrinet)
    print("Finish Traing Model")

    # save model
    citrinet.save_to(output_path)
    print("\n Nemo model save to {}".format(output_path))


def make_args():
    parser = argparse.ArgumentParser("train nemo asr models")
    parser.add_argument("--pretrained_model", "-m", type=str, default="./models/stt_zh_citrinet_512.nemo",
                        help="pretrained model file path")
    parser.add_argument("--model_config", "-c", type=str, default="./configs/citrinet_512_zh.yaml",
                        help="pretrained model config file path")
    parser.add_argument("--train_manifest", "-t", type=str, default="./train.json",
                        help="train manifest file path")
    parser.add_argument("--valid_manifest", "-v", type=str, default="./val.json",
                        help="validation manifest file path")
    parser.add_argument("--epochs", "-e", type=int, default=300, help="train model epochs")
    parser.add_argument("--output", "-o", type=str,
                        default="{}_{}.nemo".format(time.strftime("%Y%m%d", time.localtime()), uuid.uuid4()),
                        help="model output name")
    return parser.parse_args()


if __name__ == '__main__':
    args = make_args()
    train_asr_nemo(train_manifest=args.train_manifest,
                   valid_manifest=args.valid_manifest,
                   config_path=args.model_config,
                   pretrained_model=args.pretrained_model,
                   epochs=args.epochs,
                   output_path=args.output)
