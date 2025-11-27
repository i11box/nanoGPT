# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = '/inspire/ssd/project/video-generation/public/hyr/out-webnovel'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

tensorboard_log = True # override via command line if you like
tensorboard_log_dir = '/inspire/ssd/project/video-generation/public/hyr/logs/out-webnovel'
use_diffusion_forcing = True

dataset = 'webnovel'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 12
n_head = 8
n_embd = 512
dropout = 0.1

learning_rate = 8e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 8e-5 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
