import argparse
import json
import os

# Load configuration values

parser = argparse.ArgumentParser('MOA classification')
parser.add_argument('--config', type=str, required=True, help='path to config file')
parser.add_argument('--repeat', default=False, action='store_true', help='repeat experiment only for input features')
args = parser.parse_args()

with open(args.config) as f:
    config = json.load(f)


def experiment(data_dir, feat_type, output_dir, shuffle):
    if shuffle:
        s = "--shuffle"
    else:
        s = ""
    command = f"cd moa-classification/ ; python cp_resnet_train_pred.py --data_dir {data_dir} --feat_type '{feat_type}' --model_pred_dir {output_dir}/{feat_type} {s} --file_indicator _{feat_type}"
    os.system(command)


# Run cross validation experiments with the ResNet model on the input features.

cp_data_dir = config["output_folder"] + "/model_data"
model_pred_dir = config["output_folder"] + "/predictions/"
features = ["cellprofiler", "CNN", "dino"]

if args.repeat:
    repeats = len([x for x in os.listdir(model_pred_dir) if x.find("all_performance_metrics_resnet") != -1])
    repeats = str(repeats).zfill(3)
    os.system(f"mv {model_pred_dir}/all_performance_metrics_resnet.csv {model_pred_dir}/all_performance_metrics_resnet.{repeats}.csv")
    os.system(f"mv {model_pred_dir}/moa_precision_recall_resnet.csv {model_pred_dir}/moa_precision_recall_resnet.{repeats}.csv")
    experiment(cp_data_dir, "dino", model_pred_dir, shuffle=False)
else:
    for f in features:
        experiment(cp_data_dir, f, model_pred_dir, shuffle=False)
        experiment(cp_data_dir, f, model_pred_dir, shuffle=True)

