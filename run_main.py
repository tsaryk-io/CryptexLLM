import argparse
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from utils.metrics import get_loss_function, get_metric_function
from models import Autoformer, DLinear, TimeLLM

from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
os.environ["TOKENIZERS_PARALLELISM"] = "false" # mute the warning about tokenizers parallelism

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

# MLFlow integration
try:
    from utils.mlflow_integration import setup_mlflow_for_timellm
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLFlow not available. Install mlflow to enable experiment tracking.")

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model_comment', type=str, required=True, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension') # LLama7b:4096; GPT2-small:768; BERT-base:768

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function for training')
parser.add_argument('--metric', type=str, default='MAE', help='metric for evaluation')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)
parser.add_argument('--models_dir', default='/mnt/nfs/models', help='directory to save trained models')
parser.add_argument('--num_tokens', type=int, default=1000, help='number of tokens for mapping layer')
parser.add_argument('--enable_mlflow', action='store_true', help='enable MLFlow experiment tracking')
parser.add_argument('--mlflow_experiment', type=str, default='TimeLLM-Cryptex', help='MLFlow experiment name')

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

# Initialize MLFlow tracker if enabled
mlflow_tracker = None
if args.enable_mlflow and MLFLOW_AVAILABLE and accelerator.is_local_main_process:
    try:
        # Load external data configuration if available
        external_data_config = None
        if os.path.exists('external_data_config.json'):
            with open('external_data_config.json', 'r') as f:
                external_data_config = json.load(f)
        
        mlflow_tracker = setup_mlflow_for_timellm(args, external_data_config)
        accelerator.print("MLFlow experiment tracking enabled")
    except Exception as e:
        accelerator.print(f"MLFlow initialization failed: {e}")
        mlflow_tracker = None

for ii in range(args.itr):
    # setting record of experiments
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
        args.task_name,
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.des, ii)

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.model == 'Autoformer':
        model = Autoformer.Model(args).float()
    elif args.model == 'DLinear':
        model = DLinear.Model(args).float()
    else:
        model = TimeLLM.Model(args).float()

    path = os.path.join(args.checkpoints,
                        setting + '-' + args.model_comment)  # unique checkpoint saving path
    args.content = load_content(args)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    time_now = time.time()

    train_steps = len(train_loader)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

    if args.lradj == 'COS':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
    else:
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=args.pct_start,
                                            epochs=args.train_epochs,
                                            max_lr=args.learning_rate)

    criterion = get_loss_function(args.loss, args.llm_model)
    metric_func = get_metric_function(args.metric)

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    # Variables to track final metrics
    final_test_loss = None
    final_test_metric = None

    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []

        model.train()
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):
            iter_count += 1
            model_optim.zero_grad()

            batch_x = batch_x.float().to(accelerator.device)
            batch_y = batch_y.float().to(accelerator.device)
            batch_x_mark = batch_x_mark.float().to(accelerator.device)
            batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                accelerator.device)
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                accelerator.device)

            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                    loss = criterion(outputs, batch_y)
                    
                    # Handle adaptive loss (returns dict) vs regular loss (returns tensor)
                    if isinstance(loss, dict):
                        total_loss = loss['combined_loss']
                        train_loss.append(total_loss.item())
                        loss = total_loss  # Use total loss for backward pass
                    else:
                        train_loss.append(loss.item())
            else:
                if args.output_attention:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                
                # Handle adaptive loss (returns dict) vs regular loss (returns tensor)
                if isinstance(loss, dict):
                    total_loss = loss['combined_loss']
                    train_loss.append(total_loss.item())
                    loss = total_loss  # Use total loss for backward pass
                else:
                    train_loss.append(loss.item())

            if (i + 1) % 100 == 0:
                # Handle loss display for both adaptive and regular loss
                loss_value = loss.item() if hasattr(loss, 'item') else loss['combined_loss'].item()
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_value))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if args.use_amp:
                scaler.scale(loss).backward()
                scaler.step(model_optim)
                scaler.update()
            else:
                accelerator.backward(loss)
                model_optim.step()

            if args.lradj == 'TST':
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                scheduler.step()

        accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        vali_loss, vali_metric = vali(args, accelerator, model, vali_data, vali_loader, criterion, metric_func)
        test_loss, test_metric = vali(args, accelerator, model, test_data, test_loader, criterion, metric_func)
        
        # Store the final test metrics (will be overwritten each epoch, so final values are from last epoch)
        final_test_loss = test_loss
        final_test_metric = test_metric
        
        # Log training progress to MLFlow
        if mlflow_tracker:
            mlflow_tracker.log_training_progress(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=vali_loss,
                test_loss=test_loss,
                metric_value=test_metric,
                metric_name=args.metric
            )
        
        accelerator.print(
            "Epoch: {0} | {6} Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} {5} Metric: {4:.7f}".format(
                epoch + 1, train_loss, vali_loss, test_loss, test_metric, args.metric, args.loss))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

        if args.lradj != 'TST':
            if args.lradj == 'COS':
                scheduler.step()
                accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
            else:
                if epoch == 0:
                    args.learning_rate = model_optim.param_groups[0]['lr']
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

        else:
            accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

    # Added model saving code here
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        # Create models directory if it doesn't exist
        models_path = args.models_dir
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        
        # Create a descriptive model name
        model_name = f"{args.model_id}.pth"
        model_path = os.path.join(models_path, model_name)
        
        # Save model state dict
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), model_path)
        accelerator.print(f'Model saved to {model_path}')
        
        # Log model to MLFlow
        if mlflow_tracker:
            try:
                mlflow_tracker.log_model(
                    model=unwrapped_model,
                    model_name="time_llm_model",
                    artifacts={"model_weights": model_path}
                )
                accelerator.print("Model logged to MLFlow")
            except Exception as e:
                accelerator.print(f"Failed to log model to MLFlow: {e}")

    # Existing cleanup code
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')

# After all training and evaluation, print the final metrics for the launcher to capture
accelerator.wait_for_everyone()
if accelerator.is_local_main_process:
    if final_test_loss is not None and final_test_metric is not None:
        metrics = {
            "model_id": args.model_id,
            args.loss.lower(): final_test_loss,
            args.metric.lower(): final_test_metric
        }
        
        # Log final metrics to MLFlow
        if mlflow_tracker:
            try:
                final_metrics = {
                    "final_test_loss": final_test_loss,
                    f"final_test_{args.metric.lower()}": final_test_metric
                }
                mlflow_tracker.log_metrics(final_metrics)
                
                # End MLFlow run
                mlflow_tracker.end_run("FINISHED")
                accelerator.print("MLFlow run completed successfully")
            except Exception as e:
                accelerator.print(f"Error ending MLFlow run: {e}")
        
        # Print metrics in a parseable format (/!\ Make sure the formatting agrees with the parsing in launch_experiment.py)
        print(f"FINAL_METRICS:{json.dumps(metrics)}")
