## SSCL-Text2SQL
Code for our AAAI-23 accepted paper: [Learn from Yesterday: A Semi-Supervised Continual Learning Method for Supervision-Limited Text-to-SQL Task Streams](https://arxiv.org/abs/2211.11226)

### Running Code

#### 1. Preprocess data before training
We have provided IRNet readable form of data in data. You can also run our script to convert the raw task streams into this form.
```bash
cd preprocess
sh run_me_spider.sh
sh run_me_wikisql.sh
```

#### 2. Training SFNet on task streams

Execute the following command for training on Spider task stream.
```bash
sh train_sfnet_spider.sh
```
Execute the following command for training on WikiSQL task stream.
```bash
sh train_sfnet_wikisql.sh
```
The model's loss and metrics are logged in `./saved_model/{your_run_id}/log`.
You can view them by running the following command to start TensorBoard.
```bash
tensorboard --logdir=./saved_model/{your_run_id}/log
```
Then, open your browser and enter `localhost:6006` in the address bar to get to the TensorBoard page.
